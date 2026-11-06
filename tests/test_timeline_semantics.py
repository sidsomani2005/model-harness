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
