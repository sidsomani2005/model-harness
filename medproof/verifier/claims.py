"""Stage 1: extract atomic, check-worthy claims from the target answer (Haiku, low effort).

The extractor sees only the question and the answer, never the chart and never the key.
Few-shot examples are deliberately from a NON-medical domain (building inspection) so the
prompt cannot leak any case's trap.
"""

from __future__ import annotations

import os
from enum import Enum

from pydantic import BaseModel, Field

from medproof import llm
from medproof.llm import Meter

MAX_CLAIMS = int(os.environ.get("MAX_CLAIMS_PER_ANSWER", "8"))


class ClaimKind(str, Enum):
    fact_from_source = "fact_from_source"      # a value/event the record states
    causal_attribution = "causal_attribution"  # X caused / explains Y
    temporal = "temporal"                      # ordering, timing, recency ("latest", "before")
    status = "status"                          # confirmed vs suspected / active vs resolved
    absence = "absence"                        # "the record has no X" / "none documented"
    conflict_report = "conflict_report"        # answer says sources disagree
    abstention = "abstention"                  # answer says it cannot be determined
    plan_or_followup = "plan_or_followup"      # scheduled/planned items, prescriptions
    other = "other"


class Certainty(str, Enum):
    definitive = "definitive"
    hedged = "hedged"


class Importance(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class ExtractedClaim(BaseModel):
    claim_id: str = Field(description="c1, c2, ... in order of appearance")
    text: str = Field(description="One atomic proposition, faithful to the answer's wording and certainty")
    kind: ClaimKind
    certainty: Certainty
    importance: Importance
    is_final_conclusion: bool = Field(description="True if this is (part of) the answer's bottom line")
    entities: list[str] = Field(description="Key nouns: drugs, labs, diagnoses, tests, dates, people")
    cited_lines: list[str] = Field(description="Line IDs the answer cites for this claim, verbatim, e.g. ['L007']")


class ClaimList(BaseModel):
    claims: list[ExtractedClaim]
    answer_abstains: bool = Field(description="The answer's bottom line is that it cannot be determined")
    answer_reports_conflict: bool = Field(description="The answer explicitly says sources disagree")


SYSTEM = """You split an answer into atomic, checkable claims. You do not judge whether claims are true.

Rules:
- One proposition per claim. "The roof was inspected on 4 May and passed" -> two claims.
- Keep the answer's own certainty. "may have contributed" is hedged; "caused" is definitive. Never strengthen or weaken a claim.
- Keep negations exactly ("no leaks were found" stays negative).
- Prefer claims that matter to the bottom line: conclusions, causes, timing, status (confirmed vs suspected), stated absences, plans, and any claim that sources conflict.
- Skip pure restatements of the question, generic disclaimers, and advice to consult someone.
- Copy cited line IDs exactly as the answer wrote them; [] if none for that claim.
- Return at most {max_claims} claims, most important first.

Example (building inspection, not the domain you will see):
Question: Did the 4 May inspection find water damage, and what caused it?
Answer: The 4 May inspection [R004] found staining on the ceiling of unit 3B. This was caused by the roof leak reported on 9 May [W012]. No other units were affected. The inspector's report is pending sign-off.
Claims:
c1 fact_from_source definitive high: "The 4 May inspection found staining on the ceiling of unit 3B." cited [R004]
c2 causal_attribution definitive high final: "The staining was caused by the roof leak reported on 9 May." cited [W012]
c3 absence definitive medium: "No other units were affected." cited []
c4 status definitive low: "The inspector's report is pending sign-off." cited []
"""


def _user(question: str, answer: str) -> str:
    return f"QUESTION:\n{question}\n\nANSWER TO DECOMPOSE:\n<<<\n{answer}\n>>>"


async def extract_claims(question: str, answer: str, meter: Meter) -> tuple[ClaimList | None, llm.LLMResult]:
    r = await llm.check(SYSTEM.format(max_claims=MAX_CLAIMS), _user(question, answer), ClaimList,
                        label="claims", meter=meter, max_tokens=3000)
    cl: ClaimList | None = r.parsed  # type: ignore[assignment]
    if cl is None:
        return None, r
    order = {Importance.high: 0, Importance.medium: 1, Importance.low: 2}
    kept = sorted(cl.claims, key=lambda c: (not c.is_final_conclusion, order[c.importance]))[:MAX_CLAIMS]
    # keep original reading order for display, after pruning by priority
    keep_ids = {c.claim_id for c in kept}
    cl.claims = [c for c in cl.claims if c.claim_id in keep_ids]
    return cl, r
