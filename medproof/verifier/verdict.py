"""Stage 6: conservative verdict assembly.

Principles (spec §5 stage 6, §14):
- ERROR_DETECTED needs a material finding that is cited by real lines, quote-verified,
  and not under review (checkers agreed, or the deterministic timeline proved it).
- Any doubt about a material claim without a defensible error -> INSUFFICIENT_TO_VERIFY.
- Caveats, hedges and minor side remarks never cause ERROR_DETECTED.
- NO_ISSUE_FOUND means no material issue was found, not that the answer is proven correct.
"""

from __future__ import annotations

from medproof.schema import (
    ClaimStatus, ClaimVerdict, OmissionFinding, Overall, Severity, Trap,
)


def _is_error(v: ClaimVerdict) -> bool:
    if v.severity != Severity.material or v.needs_review:
        return False
    if not any(c.quote_verified for c in v.citations) and v.status != ClaimStatus.NOT_ESTABLISHED:
        return False
    if v.status in (ClaimStatus.CONTRADICTED, ClaimStatus.CONFLICTED):
        return True
    # NOT_ESTABLISHED is an error only when it is unjustified certainty or invented detail,
    # i.e. the claim asserts as fact what the record does not establish.
    if v.status == ClaimStatus.NOT_ESTABLISHED:
        return any(t in v.traps for t in (Trap.UNJUSTIFIED_CERTAINTY, Trap.SUMMARY_HALLUCINATION, Trap.ANCHORING))
    return False


def assemble(claims: list[ClaimVerdict], omissions: list[OmissionFinding], *,
             no_claims_reason: str | None = None) -> tuple[Overall, list[Trap], list[str], str]:
    """Return (overall, traps_found, needs_review_reasons, explanation)."""
    if no_claims_reason:
        return Overall.INSUFFICIENT_TO_VERIFY, [], [no_claims_reason], f"Could not verify: {no_claims_reason}."

    errors = [v for v in claims if _is_error(v)]
    omission_errors = [o for o in omissions if o.material and o.confirmed and o.kind == "contradiction"]

    traps: list[Trap] = []
    for v in errors:
        traps += [t for t in v.traps if t not in traps]
    if omission_errors and Trap.SILENT_CONTRADICTION not in traps:
        traps.append(Trap.SILENT_CONTRADICTION)

    review = [f"{v.claim_id}: {r}" for v in claims if v.severity == Severity.material for r in v.needs_review_reasons]
    review += [f"omission not confirmed: {o.description}" for o in omissions
               if o.material and not o.confirmed and o.kind == "contradiction"]

    if errors or omission_errors:
        overall = Overall.ERROR_DETECTED
    elif review or any(v.severity == Severity.material and v.status == ClaimStatus.NOT_ESTABLISHED for v in claims):
        overall = Overall.INSUFFICIENT_TO_VERIFY
        if not review:
            review.append("a material claim is not established by the record, but no definite error was shown")
    else:
        overall = Overall.NO_ISSUE_FOUND

    return overall, traps, review, explain(overall, claims, errors, omission_errors, omissions)


def _cite(v_cits) -> str:
    return "; ".join(f"{c.file} [{c.line_id}] \"{c.quote}\"" for c in v_cits[:3])


def explain(overall: Overall, claims: list[ClaimVerdict], errors: list[ClaimVerdict],
            omission_errors: list[OmissionFinding], omissions: list[OmissionFinding]) -> str:
    lines: list[str] = []
    if overall == Overall.ERROR_DETECTED:
        lines.append("Error detected.")
        for v in errors:
            trap = f" [{', '.join(t.value for t in v.traps)}]" if v.traps else ""
            lines.append(f"- Claim \"{v.claim_text}\" is {v.status.value}{trap}: {v.why}")
            if v.citations:
                lines.append(f"  Evidence: {_cite(v.citations)}")
            if v.uncertainty:
                lines.append(f"  Still unknown: {v.uncertainty}")
        for o in omission_errors:
            lines.append(f"- The answer does not account for: {o.description}")
            if o.citations:
                lines.append(f"  Evidence: {_cite(o.citations)}")
    elif overall == Overall.INSUFFICIENT_TO_VERIFY:
        lines.append("Insufficient to verify: the checks could not defensibly confirm or refute a material claim.")
        for v in claims:
            if v.severity == Severity.material and (v.needs_review or v.status != ClaimStatus.SUPPORTED):
                lines.append(f"- \"{v.claim_text}\": {v.status.value}. {v.why} "
                             f"{'(' + '; '.join(v.needs_review_reasons) + ')' if v.needs_review_reasons else ''}".rstrip())
    else:
        lines.append("No material issue found. This is not a guarantee of correctness.")
        notes = [v for v in claims if v.status != ClaimStatus.SUPPORTED]
        for v in notes:
            lines.append(f"- Minor: \"{v.claim_text}\" is {v.status.value}. {v.why}")
    warn = [o for o in omissions if not (o.material and o.confirmed and o.kind == "contradiction")]
    for o in warn[:3]:
        lines.append(f"- Note ({o.kind}): {o.description}")
    return "\n".join(lines)
