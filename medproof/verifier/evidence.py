"""Stages 2 and 4: evidence discovery per claim, and the global cross-document sweep.

Retrieval deliberately goes beyond the lines the target cited: a citation can be locally
faithful ("[P007] NKDA") and globally misleading (allergy list documents anaphylaxis).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from medproof import llm
from medproof.llm import Meter
from medproof.schema import SourceLine
from medproof.source_index import SourceIndex, dedupe
from medproof.verifier.claims import ClaimList, ExtractedClaim

MAX_PACK_LINES = 18


class EvidencePack(BaseModel):
    claim_id: str
    cited: list[SourceLine] = Field(default_factory=list)
    unresolved_citations: list[str] = Field(default_factory=list)
    context: list[SourceLine] = Field(default_factory=list)  # cited +/- 2 neighbours and file headers
    retrieved: list[SourceLine] = Field(default_factory=list)  # BM25 over the WHOLE chart

    def all_lines(self) -> list[SourceLine]:
        return dedupe(self.cited + self.context + self.retrieved)


def gather(claim: ExtractedClaim, index: SourceIndex, k: int = 6) -> EvidencePack:
    cited: list[SourceLine] = []
    unresolved: list[str] = []
    for ref in claim.cited_lines:
        line = index.resolve(ref)
        if line:
            cited.append(line)
        else:
            unresolved.append(ref)

    context: list[SourceLine] = []
    for line in cited:
        context += index.file_header(line.file)
        context += index.neighbors(line, 2)

    seen = {(l.file, l.line_id) for l in cited + context}
    retrieved = index.search(claim.text, k=k, exclude=seen)
    seen |= {(l.file, l.line_id) for l in retrieved}
    for ent in claim.entities[:4]:
        retrieved += index.search(ent, k=3, exclude=seen)
        seen |= {(l.file, l.line_id) for l in retrieved}

    pack = EvidencePack(claim_id=claim.claim_id, cited=dedupe(cited), unresolved_citations=unresolved,
                        context=dedupe(context), retrieved=dedupe(retrieved))
    # cap size: cited lines always kept; trim retrieved first, then context
    budget = MAX_PACK_LINES - len(pack.cited)
    pack.context = pack.context[: max(0, budget // 2)]
    pack.retrieved = pack.retrieved[: max(0, budget - len(pack.context))]
    return pack


# ---------------------------------------------------------------- global sweep


class SweepItem(BaseModel):
    description: str = Field(description="What the chart shows that the answer does not account for")
    kind: str = Field(description="contradiction | uncertainty | negative_finding | status_caveat | other")
    line_ids: list[str] = Field(description="Line IDs in the chart that show it, exactly as printed")
    quotes: list[str] = Field(description="Short verbatim quote from each listed line, same order")
    material: bool = Field(description="True only if accounting for it would change or qualify the answer's bottom line")
    related_claim_id: str | None = Field(description="Claim it bears on, if any")


class SweepResult(BaseModel):
    items: list[SweepItem]


SWEEP_SYSTEM = """You audit an answer about a SYNTHETIC clinical record for what it leaves out.

You are given the question, the answer's extracted claims, and chart lines. List ONLY evidence in the chart that the answer fails to account for AND that would change or qualify its bottom line, for example:
- another entry that contradicts a claim (two sources disagree and the answer picks one silently);
- the record marks something as suspected, preliminary, pending or unconfirmed while the answer treats it as established;
- a later or earlier entry that changes the timeline the answer relies on;
- a documented negative finding or missing result the answer ignores.

Do not list things the answer already mentions, hedges, or flags as uncertain. Do not list trivia. Do not give medical advice or judge treatment. If nothing material is omitted, return an empty list. Every line ID and quote must be copied exactly from the chart lines shown."""


def _sweep_lines(index: SourceIndex, question: str, claims: ClaimList) -> tuple[str, bool]:
    if index.is_short:
        return index.render_full(), True
    cand: list[SourceLine] = []
    cand += index.search(question, k=10)
    for c in claims.claims:
        cand += index.search(c.text, k=5)
    cand += index.cue_lines()
    return SourceIndex.render(dedupe(cand)[:80]), False


async def global_sweep(question: str, answer_claims: ClaimList, index: SourceIndex,
                       meter: Meter) -> tuple[SweepResult | None, llm.LLMResult, bool]:
    chart, full = _sweep_lines(index, question, answer_claims)
    claims_txt = "\n".join(
        f"{c.claim_id} [{c.certainty.value}{', bottom line' if c.is_final_conclusion else ''}] {c.text}"
        for c in answer_claims.claims
    )
    stance = []
    if answer_claims.answer_abstains:
        stance.append("The answer says the question cannot be determined.")
    if answer_claims.answer_reports_conflict:
        stance.append("The answer explicitly reports conflicting documentation.")
    user = (
        f"QUESTION:\n{question}\n\nANSWER CLAIMS:\n{claims_txt}\n{' '.join(stance)}\n\n"
        f"CHART LINES ({'complete chart' if full else 'candidate lines from the whole chart'}):\n{chart}"
    )
    r = await llm.check(SWEEP_SYSTEM, user, SweepResult, label="global_sweep", meter=meter, max_tokens=3000)
    return r.parsed, r, full  # type: ignore[return-value]
