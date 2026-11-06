"""Stages 3 and 5: narrow evidence checks, chronology, confirmation, and citation validation."""

from __future__ import annotations

import difflib
import re
from enum import Enum

from pydantic import BaseModel, Field

from medproof import llm
from medproof.llm import Meter
from medproof.schema import Citation, ClaimStatus, ClaimVerdict, Severity, SourceLine, Trap
from medproof.source_index import SourceIndex
from medproof.verifier.claims import Certainty, ClaimKind, ExtractedClaim, Importance
from medproof.verifier.evidence import EvidencePack

# ---------------------------------------------------------------- citation validation (1.6)

_PUNCT = re.compile(r"[^\w.\s/-]")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", _PUNCT.sub(" ", s.lower())).strip()


def quote_matches(quote: str, source_text: str, threshold: float = 0.85) -> bool:
    q, s = _norm(quote), _norm(source_text)
    if not q:
        return False
    # every number in the quote must literally appear in the source (no fuzzy values)
    src_nums = set(_NUM.findall(s))
    if any(n not in src_nums for n in _NUM.findall(q)):
        return False
    if q in s:
        return True
    # best-window fuzzy match, tolerant of small whitespace/punctuation/ellipsis differences
    n = len(q)
    if n > len(s):
        return difflib.SequenceMatcher(None, q, s).ratio() >= threshold
    best = 0.0
    step = max(1, n // 8)
    for i in range(0, len(s) - n + 1, step):
        best = max(best, difflib.SequenceMatcher(None, q, s[i:i + n]).ratio())
        if best >= threshold:
            return True
    return False


def validate_citation(index: SourceIndex, line_id: str, quote: str) -> Citation | None:
    """A citation the verifier can stand behind, or None if the line does not exist.

    The quote shown to users is always the verbatim source text; `quote_verified` records
    whether the checker's own quote actually matched that line (if not, the line may not
    carry the meaning the checker asserted).
    """
    line = index.resolve(line_id)
    if line is None:
        return None
    ok = quote_matches(quote, line.text) if quote else False
    return Citation(file=line.file, line_id=line.line_id, quote=line.text[:240], quote_verified=ok)


# ---------------------------------------------------------------- narrow claim check (stage 3)


class EvidenceRole(str, Enum):
    supports = "supports"
    contradicts = "contradicts"
    context = "context"


class EvidenceRef(BaseModel):
    line_id: str = Field(description="Exactly as printed, e.g. L004")
    quote: str = Field(description="Short verbatim excerpt of that line")
    role: EvidenceRole


class ClaimCheck(BaseModel):
    status: ClaimStatus
    evidence: list[EvidenceRef]
    reason: str = Field(description="One or two plain-English sentences naming the decisive lines")
    uncertainty: str = Field(description="What remains unknown; empty if nothing")
    suspected_treated_as_confirmed: bool
    different_patient_or_encounter: bool
    traps: list[Trap]
    material: bool = Field(description="True if the claim being wrong would change the answer's bottom line")
    confidence: str = Field(description="high | medium | low")


CHECK_SYSTEM = """You check ONE claim from an AI answer against lines from a SYNTHETIC clinical record. Judge only faithfulness to the record; never give medical advice.

Labels:
- SUPPORTED: the lines state this, at the certainty the claim uses. A hedged claim ("may", "possibly", "cannot be ruled out") is SUPPORTED if the record leaves it open and the claim does not overstate. A claim that correctly reports that sources disagree, or correctly says something is not documented, is SUPPORTED.
- CONTRADICTED: a line states the opposite, or the timeline makes the claim impossible (e.g. a cause that starts after the effect it is said to have caused).
- NOT_ESTABLISHED: nothing shown establishes it, or the record only suspects, plans or orders what the claim states as fact (e.g. "suspected X" reported as "confirmed X", "follow-up recommended" reported as "appointment scheduled"). NOT_ESTABLISHED is not the same as CONTRADICTED.
- CONFLICTED: different lines disagree on this point and the answer asserts one side as fact without acknowledging the other anywhere.

A claim that a specific document says something ("the 03-04 note states X") is SUPPORTED if that document says it, even if other documents disagree. The full answer is shown for context: if it acknowledges a conflict or uncertainty elsewhere, do not penalise this claim for not repeating it. Judge only the claim, not omissions elsewhere in the answer.

Answer these narrow questions while deciding:
1. Does any shown line support, contradict, or fail to establish the claim?
2. Does the claim treat a suspected/preliminary/pending item as confirmed?
3. Is there an unresolved conflict between lines on this point?
4. Are the lines about the same patient and encounter as the claim?
5. If the claim is causal, does the cause precede the effect it is said to explain? A drug started after a lab abnormality cannot explain that abnormality's initial onset, though it might affect later values. Do not conclude more than the timeline shows.

Trap labels (use only those that apply): TEMPORAL_CAUSALITY, SILENT_CONTRADICTION, UNJUSTIFIED_CERTAINTY, ANCHORING (an early preliminary label kept despite later evidence), SUMMARY_HALLUCINATION (invented detail), ENTITY_BLEED (fact from another patient/encounter), STALE_COPY_FORWARD (older value repeated in a newer note).

Cite only line IDs that appear below, with verbatim quotes."""


def _claim_header(claim: ExtractedClaim) -> str:
    return (
        f"CLAIM {claim.claim_id} ({claim.kind.value}, {claim.certainty.value}"
        f"{', bottom line of the answer' if claim.is_final_conclusion else ''}):\n{claim.text}\n"
        f"Lines the answer cited for it: {', '.join(claim.cited_lines) or 'none'}"
    )


def _lines_block(pack: EvidencePack, index: SourceIndex, full_chart: bool) -> str:
    if full_chart:
        return "COMPLETE CHART:\n" + index.render_full()
    parts = []
    if pack.cited:
        parts.append("CITED BY THE ANSWER:\n" + SourceIndex.render(pack.cited))
    if pack.context:
        parts.append("NEIGHBOURING CONTEXT:\n" + SourceIndex.render(pack.context))
    if pack.retrieved:
        parts.append("OTHER RELEVANT LINES FROM THE WHOLE CHART (not cited by the answer):\n"
                     + SourceIndex.render(pack.retrieved))
    if pack.unresolved_citations:
        parts.append(f"NOTE: the answer cited {', '.join(pack.unresolved_citations)}, which do not exist in the chart.")
    return "\n\n".join(parts) or "(no relevant lines found)"


def needs_full_chart(claim: ExtractedClaim, index: SourceIndex) -> bool:
    """Absence, abstention and conflict claims can only be judged against the whole record."""
    return index.is_short and claim.kind in (ClaimKind.absence, ClaimKind.abstention, ClaimKind.conflict_report)


async def run_check(claim: ExtractedClaim, pack: EvidencePack, index: SourceIndex, meter: Meter,
                    full_chart: bool = False, label: str | None = None, answer: str = "") -> llm.LLMResult:
    user = f"{_claim_header(claim)}\n\n{_lines_block(pack, index, full_chart)}"
    if answer:
        user += f"\n\nFULL ANSWER (context only; judge the claim above):\n<<<\n{answer}\n>>>"
    return await llm.check(CHECK_SYSTEM, user, ClaimCheck, label=label or f"check:{claim.claim_id}",
                           meter=meter, max_tokens=2000)


# ---------------------------------------------------------------- chronology (stage 5)


class EffectScope(str, Enum):
    initial_onset = "initial_onset"
    persistence_or_later = "persistence_or_later"
    unspecified = "unspecified"


class TemporalExtraction(BaseModel):
    applicable: bool = Field(description="True only if the claim is about whether one event caused or explains another")
    claim_asserts_causation: bool = Field(description="True if the claim says the cause DID cause/explain the effect; "
                                          "False if it denies, doubts, or says it cannot")
    cause_line_id: str | None = Field(description="Line documenting when the proposed cause FIRST started")
    effect_line_id: str | None = Field(description="EARLIEST line documenting the effect")
    effect_scope: EffectScope = Field(description="Which part of the effect the claim attributes to the cause")


TEMPORAL_SYSTEM = """You extract event anchors for a timeline check. Do not judge the claim.
If the claim attributes an effect to a cause, give:
- cause_line_id: the line that documents when the proposed cause FIRST began (first dose, first exposure, start);
- effect_line_id: the EARLIEST line that documents the effect (first abnormal value, first symptom);
- effect_scope: initial_onset if the claim is about how the effect began or simply says X caused Y; persistence_or_later if it only claims X kept it going or affected later values; unspecified if unclear.
- claim_asserts_causation: false if the claim says the cause did NOT / cannot / is unlikely to explain the effect.
Use only line IDs shown. If the claim is not causal, set applicable=false."""

_DENIAL = re.compile(r"\b(cannot|can't|could not|couldn't|did not|didn't|does not|doesn't|is not|isn't|not|no|"
                     r"unlikely|unable|ruled? out|predates?)\b", re.I)


def denies_causation(text: str) -> bool:
    """Deterministic guard: negated or ordering-based causal claims are never treated as assertions."""
    return bool(_DENIAL.search(text))


def temporal_judgement(index: SourceIndex, cause: SourceLine, effect: SourceLine,
                       scope: EffectScope, certainty: Certainty) -> tuple[ClaimStatus | None, str]:
    """Deterministic: compare the source-of-record timestamps of the two anchor lines."""
    if not cause.timestamp or not effect.timestamp:
        return None, "Timeline check skipped: an anchor line has no timestamp."
    c, e = cause.timestamp, effect.timestamp
    if c[:10] == e[:10] and (len(c) == 10 or len(e) == 10):
        return None, f"Same calendar day ({c[:10]}) without times on both lines; order cannot be established."
    note = " (date inherited from note header)" if cause.timestamp_inherited or effect.timestamp_inherited else ""
    if c > e:
        if scope == EffectScope.persistence_or_later:
            return None, (f"The cause began {c} after the effect first appeared {e}{note}; the claim concerns only "
                          "persistence/later values, which the timeline neither confirms nor rules out.")
        if certainty == Certainty.hedged and scope == EffectScope.unspecified:
            return None, f"Cause began {c}, after the effect first appeared {e}{note}; claim is hedged, flagged for context only."
        return ClaimStatus.CONTRADICTED, (
            f"{cause.file} [{cause.line_id}] shows the proposed cause began {c}, after the effect was first "
            f"documented {e} in {effect.file} [{effect.line_id}]{note}. It cannot explain the initial onset; "
            "any effect on later values is not established by the record."
        )
    return None, f"Cause ({c}) precedes effect ({e}){note}: temporally possible, but timing alone does not prove causation."


def _is_causal(claim: ExtractedClaim) -> bool:
    if claim.kind == ClaimKind.causal_attribution:
        return True
    return bool(re.search(r"\b(caus|explain|due to|because|led to|result|induc|trigger|attribut)", claim.text, re.I))


async def run_temporal(claim: ExtractedClaim, pack: EvidencePack, index: SourceIndex,
                       meter: Meter) -> tuple[ClaimStatus | None, str, list[SourceLine], llm.LLMResult | None]:
    if not _is_causal(claim):
        return None, "", [], None
    timeline = sorted([l for l in pack.all_lines() if l.timestamp], key=lambda l: l.timestamp or "")
    # widen with every dated line mentioning the claim's entities, so the earliest onset is visible
    for ent in claim.entities[:4]:
        timeline += [l for l in index.search(ent, k=8) if l.timestamp]
    seen, ordered = set(), []
    for l in sorted(timeline, key=lambda l: (l.timestamp or "", l.file, l.line_no)):
        if (l.file, l.line_id) not in seen:
            seen.add((l.file, l.line_id))
            ordered.append(l)
    user = f"{_claim_header(claim)}\n\nDATED LINES IN CHRONOLOGICAL ORDER:\n" + "\n".join(
        f"{l.timestamp}  {l.file} [{l.line_id}] {l.text}" for l in ordered)
    r = await llm.check(TEMPORAL_SYSTEM, user, TemporalExtraction, label=f"temporal:{claim.claim_id}",
                        meter=meter, max_tokens=1000)
    te: TemporalExtraction | None = r.parsed  # type: ignore[assignment]
    if te is None or not te.applicable or not te.cause_line_id or not te.effect_line_id:
        return None, "", [], r
    cause, effect = index.resolve(te.cause_line_id), index.resolve(te.effect_line_id)
    if cause is None or effect is None:
        return None, "Timeline anchors cited non-existent lines; ignored.", [], r
    status, why = temporal_judgement(index, cause, effect, te.effect_scope, claim.certainty)
    if status == ClaimStatus.CONTRADICTED and (not te.claim_asserts_causation or denies_causation(claim.text)):
        # The claim denies the causal link; the source timestamps prove it.
        return ClaimStatus.SUPPORTED, (f"Timeline is consistent with the claim: {cause.file} [{cause.line_id}] ({cause.timestamp}) "
                      f"comes after {effect.file} [{effect.line_id}] ({effect.timestamp})."), [cause, effect], r
    return status, why, [cause, effect], r


# ---------------------------------------------------------------- confirmation (second opinion)

_GROUP = {
    ClaimStatus.SUPPORTED: "pos",
    ClaimStatus.CONTRADICTED: "neg",
    ClaimStatus.CONFLICTED: "neg",
    ClaimStatus.NOT_ESTABLISHED: "unest",
}


def statuses_agree(a: ClaimStatus, b: ClaimStatus) -> bool:
    return _GROUP[a] == _GROUP[b]


# ---------------------------------------------------------------- per-claim orchestration


def _is_material(claim: ExtractedClaim, chk: ClaimCheck | None) -> bool:
    if claim.importance == Importance.low and not claim.is_final_conclusion:
        return False
    return claim.is_final_conclusion or claim.importance == Importance.high or bool(chk and chk.material)


def _citations(index: SourceIndex, refs: list[EvidenceRef], want_roles: set[EvidenceRole]) -> list[Citation]:
    out: list[Citation] = []
    for ref in refs:
        if ref.role not in want_roles:
            continue
        c = validate_citation(index, ref.line_id, ref.quote)
        if c and all((c.file, c.line_id) != (o.file, o.line_id) for o in out):
            out.append(c)
    return out


async def verify_claim(claim: ExtractedClaim, pack: EvidencePack, index: SourceIndex, meter: Meter,
                       enable_chronology: bool = True, answer: str = "") -> ClaimVerdict:
    import asyncio

    full = needs_full_chart(claim, index)
    jobs = [run_check(claim, pack, index, meter, full_chart=full, answer=answer)]
    if enable_chronology:
        jobs.append(run_temporal(claim, pack, index, meter))
    results = await asyncio.gather(*jobs)
    r_check: llm.LLMResult = results[0]
    t_status, t_why, t_lines, _ = results[1] if enable_chronology else (None, "", [], None)

    checks = ["narrow_check" + ("(full chart)" if full else "")]
    review: list[str] = []
    chk: ClaimCheck | None = r_check.parsed  # type: ignore[assignment]

    if chk is None:
        return ClaimVerdict(
            claim_id=claim.claim_id, claim_text=claim.text, status=ClaimStatus.NOT_ESTABLISHED,
            severity=Severity.material if _is_material(claim, None) else Severity.minor,
            why="Checker output could not be obtained.", needs_review=True,
            needs_review_reasons=[f"check failed: {r_check.parse_error or r_check.stop_reason}"], checks=checks,
        )

    status, why, uncertainty = chk.status, chk.reason, chk.uncertainty
    traps = list(dict.fromkeys(chk.traps))
    if chk.suspected_treated_as_confirmed and Trap.UNJUSTIFIED_CERTAINTY not in traps and Trap.ANCHORING not in traps:
        traps.append(Trap.UNJUSTIFIED_CERTAINTY)
    if chk.different_patient_or_encounter and Trap.ENTITY_BLEED not in traps:
        traps.append(Trap.ENTITY_BLEED)

    roles = {EvidenceRole.contradicts, EvidenceRole.supports} if status != ClaimStatus.SUPPORTED else {EvidenceRole.supports}
    cits = _citations(index, chk.evidence, roles) or _citations(index, chk.evidence, set(EvidenceRole))
    invalid = [e.line_id for e in chk.evidence if index.resolve(e.line_id) is None]
    if invalid:
        review.append(f"checker cited non-existent lines {invalid}")
    if status != ClaimStatus.SUPPORTED and cits and not any(c.quote_verified for c in cits):
        review.append("checker quotes did not match the cited source lines")

    # chronology: deterministic comparison of source timestamps
    if enable_chronology and t_why:
        checks.append("chronology")
        uncertainty = (uncertainty + " " if uncertainty else "") + (t_why if t_status is None else "")
    if t_status == ClaimStatus.CONTRADICTED:
        if status == ClaimStatus.SUPPORTED:
            # deterministic evidence overrides the model; recorded for audit, does not block the verdict
            checks.append("chronology_overrode_narrow_check")
        status, why = ClaimStatus.CONTRADICTED, t_why
        traps = [Trap.TEMPORAL_CAUSALITY] + [t for t in traps if t != Trap.TEMPORAL_CAUSALITY]
        for l in t_lines:
            if all((l.file, l.line_id) != (c.file, c.line_id) for c in cits):
                cits.append(Citation(file=l.file, line_id=l.line_id, quote=l.text[:240], quote_verified=True))
        uncertainty = uncertainty or "Contribution to later values is not established by the record."

    if t_status == ClaimStatus.SUPPORTED and status != ClaimStatus.SUPPORTED:
        # deterministic evidence (timestamps) proves the denial; recorded for audit
        checks.append("chronology_overrode_narrow_check")
        status, why = ClaimStatus.SUPPORTED, t_why
        for l in t_lines:
            if all((l.file, l.line_id) != (c.file, c.line_id) for c in cits):
                cits.append(Citation(file=l.file, line_id=l.line_id, quote=l.text[:240], quote_verified=True))

    material = _is_material(claim, chk)

    # second opinion for material negative findings (independent call, full chart when short)
    if material and status != ClaimStatus.SUPPORTED and t_status not in (ClaimStatus.CONTRADICTED, ClaimStatus.SUPPORTED):
        checks.append("confirmation")
        r2 = await run_check(claim, pack, index, meter, full_chart=index.is_short,
                             label=f"confirm:{claim.claim_id}", answer=answer)
        chk2: ClaimCheck | None = r2.parsed  # type: ignore[assignment]
        if chk2 is None:
            review.append("confirmation check failed")
        elif (status == ClaimStatus.NOT_ESTABLISHED and chk2.status == ClaimStatus.SUPPORTED and not full
              and index.is_short and chk2.confidence == "high"):
            # the first pass saw only retrieved lines; the second saw the whole chart and found support
            checks.append("full_chart_found_support")
            status, why, uncertainty, traps = ClaimStatus.SUPPORTED, chk2.reason, chk2.uncertainty, []
            cits = _citations(index, chk2.evidence, {EvidenceRole.supports}) or cits
        elif not statuses_agree(status, chk2.status):
            review.append(f"checkers disagree: {status.value} vs {chk2.status.value}")
        else:
            for c in _citations(index, chk2.evidence, set(EvidenceRole)):
                if all((c.file, c.line_id) != (o.file, o.line_id) for o in cits):
                    cits.append(c)

    if status in (ClaimStatus.CONTRADICTED, ClaimStatus.CONFLICTED) and not any(c.quote_verified for c in cits):
        review.append("no valid citation carries the asserted contradiction")

    if status == ClaimStatus.SUPPORTED:
        traps = []

    if pack.unresolved_citations:
        # Reported, but a correct claim with a bad citation is not by itself a material error.
        checks.append("target_citation_validation")
        uncertainty = (uncertainty + " " if uncertainty else "") + (
            f"The answer cites line(s) {', '.join(pack.unresolved_citations)} that do not exist in the chart.")
        if status != ClaimStatus.SUPPORTED and Trap.SUMMARY_HALLUCINATION not in traps:
            traps.append(Trap.SUMMARY_HALLUCINATION)

    return ClaimVerdict(
        claim_id=claim.claim_id, claim_text=claim.text, status=status,
        severity=Severity.material if material else Severity.minor,
        citations=cits, why=why, uncertainty=uncertainty.strip(), traps=traps,
        needs_review=bool(review), needs_review_reasons=review, checks=checks,
    )
