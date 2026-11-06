"""Path A: the target AI's blind first attempt (WORKPLAN 2.4, spec sections 4 and 7).

- One model (default `claude-sonnet-5-5`) for every case and twin.
- Fresh conversation per call: a single user message, no history, no system prompt.
- The same neutral template (spec section 7) for every case. It never contains the
  key, traps, expected answers or verification instructions; the runner only ever
  receives `PublicCase` objects and cannot load case files itself.
- Model default configuration: no effort, thinking, temperature or output format is set.
  No server-side fallbacks (spec section 9: no silent cross-model fallback).
- The FIRST response is saved untouched as a `TargetRun`. No retries of a completed
  response, no edits, and an existing first run for the same case version is never
  overwritten. Refusals are saved with `stop_reason == "refusal"`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

from medproof import llm
from medproof.runs import DEFAULT_ROOT, load_target_runs, model_short, save_target_raw, save_target_run
from medproof.schema import PublicCase, TargetRun

DEFAULT_TARGET_MODEL = "claude-sonnet-5-5"
TARGET_MAX_TOKENS = 16000

PROMPT_TEMPLATE = (
    "You are reviewing a SYNTHETIC clinical record for documentation accuracy.\n"
    "Answer the user's question using the supplied source documents only.\n"
    "If evidence is absent, contradictory or tentative, say so.\n"
    "Cite relevant file/line IDs. Provide a concise answer and evidence summary.\n"
    "Do not invent missing details.\n"
    "\n"
    "SOURCES:\n"
    "{sources}\n"
    "\n"
    "QUESTION:\n"
    "{question}"
)


def format_sources(case: PublicCase) -> str:
    """All source files with their line ids, in the case's declared order."""
    return "\n\n".join(f"=== FILE: {name} ===\n{text.rstrip()}" for name, text in case.sources.items())


def build_prompt(case: PublicCase) -> str:
    return PROMPT_TEMPLATE.format(sources=format_sources(case), question=case.question)


def request_config(model: str) -> dict:
    return {
        "model": model,
        "max_tokens": TARGET_MAX_TOKENS,
        "system": None,
        "messages": "single user message (fresh conversation)",
        "effort": "not set (model default)",
        "thinking": "not set (model default)",
        "output_format": "not set (plain text answer)",
        "fallbacks": "none (spec section 9: no silent cross-model fallback)",
        "retries_of_completed_response": 0,
        "anthropic_sdk_version": anthropic.__version__,
    }


@dataclass
class RunOutcome:
    case_id: str
    status: str  # "saved" | "refusal" | "skipped_existing" | "dry_run" | "error"
    detail: str = ""
    path: Optional[Path] = None


def _existing_first_run(case: PublicCase, model: str, runs_root: Path) -> Optional[TargetRun]:
    for run in load_target_runs(runs_root):
        if (
            run.case_id == case.case_id
            and run.model == model
            and run.first_attempt
            and run.case_version_sha256 == case.source_sha256
        ):
            return run
    return None


async def run_one(case: PublicCase, model: str, is_twin: bool, runs_root: Path = DEFAULT_ROOT) -> RunOutcome:
    existing = _existing_first_run(case, model, runs_root)
    if existing is not None:
        return RunOutcome(case.case_id, "skipped_existing", f"first run already saved: {existing.run_id}")
    prompt = build_prompt(case)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        # anthropic.omit drops the system field entirely: the spec template is one user message.
        result = await llm.call(model, anthropic.omit, prompt, effort=None, schema=None,
                                max_tokens=TARGET_MAX_TOKENS, label=f"target:{case.case_id}")
    except Exception as exc:  # no response exists, so nothing is saved and no attempt is consumed
        return RunOutcome(case.case_id, "error", f"{type(exc).__name__}: {exc}")
    wall_ms = int((time.perf_counter() - t0) * 1000)
    run = TargetRun(
        run_id=f"{started:%Y%m%dT%H%M%SZ}_{case.case_id}_{model_short(model)}_first",
        case_id=case.case_id,
        case_version_sha256=case.source_sha256,
        model=model,
        timestamp_utc=started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        exact_prompt=prompt,
        system_prompt="",
        answer_raw=result.text or "",
        first_attempt=True,
        stop_reason=result.stop_reason,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms or wall_ms,
        request_config=request_config(model),
    )
    path = save_target_run(run, runs_root)
    # Everything TargetRun has no field for (answering model, stop details, cache usage,
    # cost, full provider response) is kept verbatim next to it.
    save_target_raw(run.run_id, {**result.to_log(), "raw_response": result.raw}, runs_root)
    status = "refusal" if result.stop_reason == "refusal" else "saved"
    detail = f"stop_reason={result.stop_reason}"
    if result.model and result.model != model:
        detail += f"; WARNING response model {result.model} != requested {model}"
    return RunOutcome(case.case_id, status, detail, path)


def run_targets(
    cases: list[tuple[PublicCase, bool]],
    model: str = DEFAULT_TARGET_MODEL,
    runs_root: Path = DEFAULT_ROOT,
    dry_run: bool = False,
) -> list[RunOutcome]:
    """Run each (PublicCase, is_twin) once, sequentially, each in a fresh conversation."""
    runs_root = Path(runs_root)
    outcomes: list[RunOutcome] = []
    if dry_run:
        out_dir = runs_root / "dry_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        for case, _ in cases:
            path = out_dir / f"{case.case_id}.prompt.txt"
            path.write_text(build_prompt(case) + "\n", encoding="utf-8")
            outcomes.append(RunOutcome(case.case_id, "dry_run", "prompt written; no API call", path))
        return outcomes
    for case, is_twin in cases:
        outcomes.append(asyncio.run(run_one(case, model, is_twin, runs_root)))
    return outcomes
