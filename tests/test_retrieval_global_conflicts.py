from medproof.verifier.claims import Certainty, ClaimKind, ExtractedClaim, Importance
from medproof.verifier.evidence import gather


def _claim(text, cited, kind=ClaimKind.fact_from_source, entities=()):
    return ExtractedClaim(claim_id="c1", text=text, kind=kind, certainty=Certainty.definitive,
                          importance=Importance.high, is_final_conclusion=True,
                          entities=list(entities), cited_lines=list(cited))


def test_nkda_citation_still_retrieves_documented_allergy(index):
    """The answer cites only the NKDA line; retrieval must surface the allergy-list anaphylaxis entry."""
    pack = gather(_claim("The patient has no known drug allergies.", ["P007"], entities=["drug allergies", "NKDA"]), index)
    ids = {l.line_id for l in pack.all_lines()}
    assert "P007" in {l.line_id for l in pack.cited}
    assert "A003" in ids, ids


def test_retrieval_crosses_files_for_timeline(index):
    pack = gather(_claim("Spironolactone caused the initial potassium rise.", ["M004"],
                         kind=ClaimKind.causal_attribution, entities=["spironolactone", "potassium"]), index)
    ids = {l.line_id for l in pack.all_lines()}
    assert "L004" in ids, ids  # first elevated potassium, in a different file than the citation


def test_nonexistent_citation_is_reported(index):
    pack = gather(_claim("Potassium was 5.9.", ["L099", "L4"]), index)
    assert pack.unresolved_citations == ["L099"]
    assert [l.line_id for l in pack.cited] == ["L004"]
