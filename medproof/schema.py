"""Shared data contracts (WORKPLAN Step 0, spec §8).

Changing anything here affects both lanes: coordinate before editing.

Invariant: `PublicCase` is the ONLY case object the online verifier and the
target runner receive. It carries no key, trap, rubric, answer-bank or twin
label fields, and `extra="forbid"` stops anyone from bolting them on.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Overall(str, Enum):
    NO_ISSUE_FOUND = "NO_ISSUE_FOUND"
    ERROR_DETECTED = "ERROR_DETECTED"
    INSUFFICIENT_TO_VERIFY = "INSUFFICIENT_TO_VERIFY"
    TARGET_REFUSAL = "TARGET_REFUSAL"


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"
    CONFLICTED = "CONFLICTED"


class Trap(str, Enum):
    TEMPORAL_CAUSALITY = "TEMPORAL_CAUSALITY"
    SILENT_CONTRADICTION = "SILENT_CONTRADICTION"
    UNJUSTIFIED_CERTAINTY = "UNJUSTIFIED_CERTAINTY"
    ANCHORING = "ANCHORING"
    SUMMARY_HALLUCINATION = "SUMMARY_HALLUCINATION"
    ENTITY_BLEED = "ENTITY_BLEED"
    STALE_COPY_FORWARD = "STALE_COPY_FORWARD"


class Severity(str, Enum):
    material = "material"
    minor = "minor"


# ---------------------------------------------------------------- case inputs


class SourceLine(BaseModel):
    file: str
    line_id: str
    text: str
    timestamp: str | None = None  # ISO-8601, as written or inherited from the note header
    timestamp_inherited: bool = False
    line_no: int = 0  # 0-based physical position inside the file


class PublicCase(BaseModel):
    """What the verifier and target runner may see. Nothing else."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    question: str
    sources: dict[str, str]  # file name -> raw text with [ID] line prefixes
    source_sha256: str


# ---------------------------------------------------------------- outputs


class Citation(BaseModel):
    file: str
    line_id: str
    quote: str
    quote_verified: bool = True  # quote matched the source text (exactly or after normalization)


class ClaimVerdict(BaseModel):
    claim_id: str
    claim_text: str
    status: ClaimStatus
    severity: Severity = Severity.minor
    citations: list[Citation] = Field(default_factory=list)
    why: str = ""
    uncertainty: str = ""
    traps: list[Trap] = Field(default_factory=list)
    needs_review: bool = False
    needs_review_reasons: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)  # which stages contributed (audit trail)


class TargetRun(BaseModel):
    run_id: str
    case_id: str
    case_version_sha256: str
    model: str
    timestamp_utc: str
    exact_prompt: str
    system_prompt: str = ""
    answer_raw: str
    first_attempt: bool = True
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    request_config: dict = Field(default_factory=dict)


class OmissionFinding(BaseModel):
    """Material evidence in the chart that the answer did not account for (global sweep)."""

    description: str
    kind: str  # contradiction | uncertainty | negative_finding | status_caveat | other
    citations: list[Citation] = Field(default_factory=list)
    material: bool = False
    confirmed: bool = False
    related_claim_id: str | None = None


class VerifierRun(BaseModel):
    target_run_id: str | None = None
    case_id: str
    verifier_variant: str
    overall: Overall
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    omissions: list[OmissionFinding] = Field(default_factory=list)
    traps_found: list[Trap] = Field(default_factory=list)
    explanation: str = ""
    needs_review_reasons: list[str] = Field(default_factory=list)
    total_model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: int = 0
    models_used: list[str] = Field(default_factory=list)
    fake_backend: bool = False  # True only in offline unit tests; never report these as results
    raw_model_outputs_path: str | None = None
