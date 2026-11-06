from medproof.schema import Citation, ClaimStatus, ClaimVerdict, OmissionFinding, Overall, Severity, Trap
from medproof.verifier.verdict import assemble

CIT = Citation(file="lab_results.txt", line_id="L004", quote="potassium 5.9", quote_verified=True)


def cv(status, severity=Severity.material, review=False, traps=(), cits=(CIT,)):
    return ClaimVerdict(claim_id="c1", claim_text="x", status=status, severity=severity, citations=list(cits),
                        traps=list(traps), needs_review=review, needs_review_reasons=["disagree"] if review else [])


def test_cited_material_contradiction_is_error():
    overall, traps, _, text = assemble([cv(ClaimStatus.CONTRADICTED, traps=[Trap.TEMPORAL_CAUSALITY])], [])
    assert overall == Overall.ERROR_DETECTED and traps == [Trap.TEMPORAL_CAUSALITY]
    assert "L004" in text


def test_checker_disagreement_is_insufficient_not_error():
    overall, _, review, _ = assemble([cv(ClaimStatus.CONTRADICTED, review=True)], [])
    assert overall == Overall.INSUFFICIENT_TO_VERIFY and review


def test_contradiction_without_verified_quote_is_not_error():
    bad = Citation(file="lab_results.txt", line_id="L004", quote="...", quote_verified=False)
    overall, _, _, _ = assemble([cv(ClaimStatus.CONTRADICTED, cits=[bad])], [])
    assert overall != Overall.ERROR_DETECTED


def test_not_established_is_distinct_from_contradicted():
    # plain NOT_ESTABLISHED (e.g. retrieval could not find support) -> insufficient, not an error
    overall, _, _, _ = assemble([cv(ClaimStatus.NOT_ESTABLISHED, cits=())], [])
    assert overall == Overall.INSUFFICIENT_TO_VERIFY
    # NOT_ESTABLISHED stated with unjustified certainty (confirmed by both checkers) -> error
    overall, traps, _, _ = assemble([cv(ClaimStatus.NOT_ESTABLISHED, cits=(), traps=[Trap.UNJUSTIFIED_CERTAINTY])], [])
    assert overall == Overall.ERROR_DETECTED and traps == [Trap.UNJUSTIFIED_CERTAINTY]


def test_minor_issues_never_fail_an_answer():
    overall, _, _, text = assemble([cv(ClaimStatus.SUPPORTED), cv(ClaimStatus.CONTRADICTED, severity=Severity.minor)], [])
    assert overall == Overall.NO_ISSUE_FOUND
    assert "not a guarantee" in text


def test_unconfirmed_omission_does_not_fail():
    o = OmissionFinding(description="allergy conflict", kind="contradiction", material=True, confirmed=False)
    overall, _, review, _ = assemble([cv(ClaimStatus.SUPPORTED)], [o])
    assert overall == Overall.INSUFFICIENT_TO_VERIFY and review


def test_confirmed_omission_is_silent_contradiction():
    o = OmissionFinding(description="allergy list documents anaphylaxis", kind="contradiction", material=True,
                        confirmed=True, citations=[CIT])
    overall, traps, _, _ = assemble([cv(ClaimStatus.SUPPORTED)], [o])
    assert overall == Overall.ERROR_DETECTED and Trap.SILENT_CONTRADICTION in traps


def test_uncertainty_notes_do_not_fail():
    o = OmissionFinding(description="potassium trend after day 4 unknown", kind="uncertainty", material=True, confirmed=False)
    overall, _, _, _ = assemble([cv(ClaimStatus.SUPPORTED)], [o])
    assert overall == Overall.NO_ISSUE_FOUND
