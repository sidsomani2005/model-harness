from medproof.schema import ClaimStatus
from medproof.verifier.check import EffectScope, temporal_judgement
from medproof.verifier.claims import Certainty


def test_drug_after_onset_cannot_explain_initial_onset(index):
    drug, k_first = index.resolve("M004"), index.resolve("L004")  # spironolactone 03-03; K 5.9 on 03-02
    status, why = temporal_judgement(index, drug, k_first, EffectScope.initial_onset, Certainty.definitive)
    assert status == ClaimStatus.CONTRADICTED
    assert "M004" in why and "L004" in why
    # must not over-reach into "not the cause of persistence"
    assert "not established" in why


def test_unscoped_definitive_cause_is_treated_as_onset(index):
    status, _ = temporal_judgement(index, index.resolve("M004"), index.resolve("L004"),
                                   EffectScope.unspecified, Certainty.definitive)
    assert status == ClaimStatus.CONTRADICTED


def test_persistence_claim_is_not_contradicted(index):
    status, why = temporal_judgement(index, index.resolve("M004"), index.resolve("L004"),
                                     EffectScope.persistence_or_later, Certainty.hedged)
    assert status is None
    assert "persistence" in why


def test_cause_before_effect_is_possible_not_proven(index):
    # furosemide 03-01 21:00 precedes K rise 03-02: plausible, but timing is not proof
    status, why = temporal_judgement(index, index.resolve("M003"), index.resolve("L004"),
                                     EffectScope.initial_onset, Certainty.definitive)
    assert status is None
    assert "does not prove" in why


def test_same_day_without_times_is_undecidable(index):
    # A003 has only a date; L004 a date+time on another day, so construct a same-day case
    a = index.resolve("A003").model_copy(update={"timestamp": "2026-03-02"})
    status, why = temporal_judgement(index, a, index.resolve("L004"), EffectScope.initial_onset, Certainty.definitive)
    assert status is None and "cannot be established" in why


def test_denial_claims_are_recognised():
    from medproof.verifier.check import denies_causation
    assert denies_causation("Spironolactone cannot explain the initial onset of the elevated potassium.")
    assert denies_causation("The potassium rise predates spironolactone, so it is not the initial cause.")
    assert not denies_causation("Spironolactone caused the initial potassium rise.")


async def test_timeline_proves_a_denial(index):
    """A claim that the drug CANNOT explain onset is SUPPORTED when timestamps show drug after onset."""
    from medproof import llm
    from medproof.llm import Meter
    from medproof.schema import ClaimStatus
    from medproof.verifier.check import run_temporal
    from medproof.verifier.claims import ClaimKind, ExtractedClaim, Importance
    from medproof.verifier.evidence import gather
    llm.use_fake_backend(lambda req: {"applicable": True, "claim_asserts_causation": False, "about_patient_events": True, "cause_line_id": "M004",
                                      "effect_line_id": "L004", "effect_scope": "initial_onset"})
    c = ExtractedClaim(claim_id="c1", text="Spironolactone cannot explain the initial potassium rise.",
                       kind=ClaimKind.causal_attribution, certainty=Certainty.definitive, importance=Importance.high,
                       is_final_conclusion=True, entities=["spironolactone"], cited_lines=["M004"])
    status, why, lines, _ = await run_temporal(c, gather(c, index), index, Meter())
    assert status == ClaimStatus.SUPPORTED and {l.line_id for l in lines} == {"M004", "L004"}


def test_general_facts_and_reports_are_not_timeline_checked():
    from medproof.verifier.check import is_patient_causal_claim
    assert not is_patient_causal_claim("Spironolactone is a well-known cause of hyperkalemia.")
    assert not is_patient_causal_claim("The treating physician documented that the hyperkalemia was likely secondary to spironolactone.")
    assert is_patient_causal_claim("The record supports spironolactone as the explanation for the onset of this patient's elevated potassium.")
    assert is_patient_causal_claim("Spironolactone caused the initial potassium rise.")
