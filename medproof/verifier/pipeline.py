"""Online verifier entry point (spec §4 Path B, §5).

Input: PublicCase (sources + question), the target answer text. Nothing else.
This module must never import load_case.load_private or read case.yaml / answer banks / twins.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from medproof import llm
from medproof.llm import Meter
from medproof.schema import (Citation, ClaimStatus, OmissionFinding, Overall, PublicCase, Severity, Trap,
                             VerifierRun)
from medproof.source_index import SourceIndex
from medproof.verifier import check as chk
from medproof.verifier import claims as cl
from medproof.verifier import evidence as ev
from medproof.verifier import verdict as vd

RUNS_DIR = Path(os.environ.get("MEDPROOF_RUNS_DIR", "runs"))

VARIANTS = {
    # name: (global sweep, chronology)
    "haiku": (True, True),
    "haiku-nosweep": (False, True),   # ablation: P1 without global contradiction sweep
    "haiku-nochrono": (True, False),  # ablation: P1 without chronology stage
}


class OmissionConfirm(BaseModel):
    lines_show_it: bool = Field(description="The quoted lines really say what the description claims")
    answer_acknowledges: bool = Field(description="The answer already mentions, hedges or flags this")
    changes_bottom_line: bool = Field(description="Accounting for it would change or qualify the answer's conclusion")
    reason: str


CONFIRM_OMISSION_SYSTEM = """You double-check a reviewer's finding about a SYNTHETIC clinical record. Be strict: say lines_show_it=true only if the quoted lines, read literally, show what the finding says. Say answer_acknowledges=true if the answer mentions it in any form, including hedges or "documentation conflicts". No medical advice."""


async def _confirm_omission(item: ev.SweepItem, answer: str, index: SourceIndex, meter: Meter) -> OmissionFinding:
    cits: list[Citation] = []
    for i, lid in enumerate(item.line_ids):
        c = chk.validate_citation(index, lid, item.quotes[i] if i < len(item.quotes) else "")
        if c:
            cits.append(c)
    finding = OmissionFinding(description=item.description, kind=item.kind, citations=cits,
                              material=item.material, related_claim_id=item.related_claim_id)
    if not (item.material and item.kind == "contradiction"):
        return finding
    if not any(c.quote_verified for c in cits):
        return finding  # unconfirmed: no real line carries it
    lines = "\n".join(f"{c.file} [{c.line_id}] {c.quote}" for c in cits)
    user = f"FINDING: {item.description}\n\nLINES:\n{lines}\n\nANSWER:\n<<<\n{answer}\n>>>"
    r = await llm.check(CONFIRM_OMISSION_SYSTEM, user, OmissionConfirm, label="confirm_omission", meter=meter,
                        max_tokens=800)
    oc: OmissionConfirm | None = r.parsed  # type: ignore[assignment]
    finding.confirmed = bool(oc and oc.lines_show_it and not oc.answer_acknowledges and oc.changes_bottom_line)
    return finding


async def averify(case: PublicCase, answer: str, variant: str = "haiku", target_run_id: str | None = None,
                  target_stop_reason: str | None = None, save: bool = True) -> VerifierRun:
    if variant == "haiku-laya":
        raise NotImplementedError("Laya prioritization (P2) is optional and not implemented yet")
    if variant not in VARIANTS:
        raise ValueError(f"unknown verifier variant {variant!r}; choose from {sorted(VARIANTS)}")
    use_sweep, use_chrono = VARIANTS[variant]

    t0 = time.perf_counter()
    meter = Meter()
    stages: dict = {"variant": variant, "case_id": case.case_id, "source_sha256": case.source_sha256}

    def finish(overall: Overall, claim_verdicts=(), omissions=(), traps=(), review=(), explanation="") -> VerifierRun:
        run = VerifierRun(
            target_run_id=target_run_id, case_id=case.case_id, verifier_variant=variant, overall=overall,
            claim_verdicts=list(claim_verdicts), omissions=list(omissions), traps_found=list(traps),
            explanation=explanation, needs_review_reasons=list(review),
            total_model_calls=meter.n_calls, input_tokens=int(meter.total("input_tokens")),
            output_tokens=int(meter.total("output_tokens")),
            cache_read_input_tokens=int(meter.total("cache_read_input_tokens")),
            cache_creation_input_tokens=int(meter.total("cache_creation_input_tokens")),
            estimated_cost_usd=meter.cost_usd, latency_ms=int((time.perf_counter() - t0) * 1000),
            models_used=meter.models, fake_backend=meter.any_fake,
        )
        if save:
            run.raw_model_outputs_path = str(_save(run, stages, meter))
        return run

    # Refusal or empty answer: record, do not interpret.
    if target_stop_reason == "refusal" or not answer.strip():
        return finish(Overall.TARGET_REFUSAL, explanation="Target refused or returned no answer; nothing to verify.")

    index = SourceIndex(case)
    stages["index"] = {"lines": len(index.lines), "malformed": index.malformed, "short_chart": index.is_short}
    target_cites, target_missing = index.extract_citations(answer)
    stages["target_citations"] = {"resolved": [f"{l.file}:{l.line_id}" for l in target_cites],
                                  "nonexistent": target_missing}

    # Stage 1
    claim_list, r_claims = await cl.extract_claims(case.question, answer, meter)
    if claim_list is None or not claim_list.claims:
        reason = "claim extraction failed" if claim_list is None else "no check-worthy claims found in answer"
        return finish(*_insufficient(reason))
    stages["claims"] = claim_list.model_dump(mode="json")

    # Stages 2-5 (per claim, parallel) and stage 4 global sweep, concurrently
    packs = {c.claim_id: ev.gather(c, index) for c in claim_list.claims}
    stages["evidence_packs"] = {k: {"cited": [l.line_id for l in p.cited], "unresolved": p.unresolved_citations,
                                    "context": [l.line_id for l in p.context],
                                    "retrieved": [l.line_id for l in p.retrieved]} for k, p in packs.items()}
    claim_jobs = [chk.verify_claim(c, packs[c.claim_id], index, meter, enable_chronology=use_chrono, answer=answer)
                  for c in claim_list.claims]
    sweep_job = ev.global_sweep(case.question, claim_list, index, meter) if use_sweep else None
    gathered = await asyncio.gather(*claim_jobs, *([sweep_job] if sweep_job else []))
    claim_verdicts = list(gathered[: len(claim_jobs)])

    # Deterministic fairness guard: an answer that explicitly reports conflicting documentation
    # is not "silently" contradicting; a CONFLICTED claim inside it is reported as a minor note.
    if claim_list.answer_reports_conflict:
        for v in claim_verdicts:
            if v.status == ClaimStatus.CONFLICTED:
                v.severity = Severity.minor
                v.traps = [t for t in v.traps if t != Trap.SILENT_CONTRADICTION]
                v.checks.append("answer_reports_conflict_guard")

    omissions: list[OmissionFinding] = []
    if sweep_job:
        sweep, _, full = gathered[-1]
        stages["sweep"] = {"full_chart": full, "result": sweep.model_dump(mode="json") if sweep else None}
        if sweep:
            omissions = list(await asyncio.gather(*[_confirm_omission(i, answer, index, meter) for i in sweep.items]))

    overall, traps, review, explanation = vd.assemble(claim_verdicts, omissions)
    return finish(overall, claim_verdicts, omissions, traps, review, explanation)


def _insufficient(reason: str):
    overall, traps, review, explanation = vd.assemble([], [], no_claims_reason=reason)
    return overall, (), (), traps, review, explanation


def _save(run: VerifierRun, stages: dict, meter: Meter) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    tid = run.target_run_id or f"adhoc_{run.case_id}_{stamp}"
    d = RUNS_DIR / "verifier" / tid / run.verifier_variant
    d.mkdir(parents=True, exist_ok=True)
    path = d / "raw.json"
    path.write_text(json.dumps({
        "saved_utc": stamp,
        "verifier_run": run.model_dump(mode="json"),
        "stages": stages,
        "model_calls": [c.to_log() for c in meter.calls],
    }, indent=2, default=str))
    return path


def verify(case: PublicCase, answer: str, variant: str = "haiku", target_run_id: str | None = None,
           target_stop_reason: str | None = None, save: bool = True) -> VerifierRun:
    """Synchronous entry point used by the CLI and the benchmark (WORKPLAN Step 0 contract)."""
    return asyncio.run(averify(case, answer, variant, target_run_id, target_stop_reason, save))
