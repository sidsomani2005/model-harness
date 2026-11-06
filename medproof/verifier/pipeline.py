"""Online verifier entry point (WORKPLAN Step 0 contract).

STUB on main so Person 2's runner/benchmark/report flow runs end to end.
The real implementation lives on branch `p1-verifier` and replaces this file.
Contract: receives only PublicCase + answer text; never the key, traps, answer bank or twins.
"""

from __future__ import annotations

from medproof.schema import Overall, PublicCase, VerifierRun


def verify(case: PublicCase, answer: str, variant: str = "haiku", target_run_id: str | None = None,
           target_stop_reason: str | None = None, save: bool = True) -> VerifierRun:
    if target_stop_reason == "refusal" or not answer.strip():
        overall = Overall.TARGET_REFUSAL
    else:
        overall = Overall.INSUFFICIENT_TO_VERIFY
    return VerifierRun(target_run_id=target_run_id, case_id=case.case_id, verifier_variant=variant,
                       overall=overall, explanation="STUB verifier: not a real verdict.")
