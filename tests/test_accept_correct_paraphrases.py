"""Fairness: correct answers must not be flagged regardless of phrasing or citation format."""
import pytest

from medproof.schema import Overall
from medproof.verifier.check import quote_matches

from .conftest import live


@pytest.mark.parametrize("text", [
    "per [L004] and [M004]", "lab_results.txt:L004, medication_admin.txt:M004", "see L4 and M4", "(L004; M004)",
])
def test_citation_formats_resolve(index, text):
    found, missing = index.extract_citations(text)
    assert {l.line_id for l in found} == {"L004", "M004"} and not missing


@pytest.mark.parametrize("quote", [
    "potassium 5.9 mmol/L", "Potassium 5.9 mmol/L (ref 3.5-5.0)", "2026-03-02 06:10 potassium 5.9", "potassium  5.9 mmol/l",
])
def test_quote_matching_tolerates_formatting(index, quote):
    assert quote_matches(quote, index.resolve("L004").text)


def test_quote_matching_rejects_wrong_value(index):
    assert not quote_matches("potassium 4.1 mmol/L", index.resolve("L004").text)


CORRECT_VARIANTS = [
    "No. The first high potassium (5.9 mmol/L) was drawn 2026-03-02 06:10 [L004]; spironolactone's first dose was "
    "2026-03-03 09:00 [M004], so it cannot explain the initial onset. Whether it contributed to the later values is not "
    "established. Allergies: the allergy list documents penicillin anaphylaxis [A003], but the 03-04 note states NKDA "
    "[P007]; this documentation conflict is unresolved.",
    "- Allergy status: conflicting. Penicillin anaphylaxis on the allergy list (A003) and in the day-2 note (P003), "
    "but 'NKDA' in the day-4 note (P007).\n- Spironolactone and potassium: the potassium was already elevated on "
    "03/02 (L004) before spironolactone started 03/03 (M004). It cannot account for the initial rise; the record does "
    "not establish its role afterwards.",
]


@live
@pytest.mark.parametrize("answer", CORRECT_VARIANTS)
def test_live_correct_variants_not_flagged(fixture_case, answer):
    from medproof.verifier.pipeline import verify
    run = verify(fixture_case, answer, target_run_id="live_paraphrase")
    assert run.overall != Overall.ERROR_DETECTED, run.explanation


@live
def test_live_wrong_answer_flagged(fixture_case):
    from medproof.verifier.pipeline import verify
    run = verify(fixture_case, "Spironolactone [M004] caused the initial potassium rise. The patient has no known "
                               "drug allergies [P007].", target_run_id="live_wrong")
    assert run.overall == Overall.ERROR_DETECTED, run.explanation
