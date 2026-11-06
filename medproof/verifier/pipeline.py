"""Online verifier entry point.

STEP 0 STUB. Person 1 (verifier lane, WORKPLAN item 1.8) replaces the body.
The benchmark lane calls `verify()` as a black box. It receives ONLY a
`PublicCase` and the answer text, never a case key, trap, rubric or twin label.
"""

from __future__ import annotations

import time
from typing import Optional

from medproof.schema import Overall, PublicCase, VerifierRun


def verify(
    case: PublicCase,
    answer: str,
    variant: str = "haiku",
    target_run_id: Optional[str] = None,
) -> VerifierRun:
    start = time.perf_counter()
    return VerifierRun(
        target_run_id=target_run_id or f"adhoc__{case.case_id}",
        verifier_variant=variant,
        overall=Overall.INSUFFICIENT_TO_VERIFY,
        case_id=case.case_id,
        explanation="Stub verifier: no checks were run.",
        needs_review_reasons=["verifier not implemented (Step 0 stub)"],
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
