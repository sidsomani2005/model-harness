"""Shared data contracts (WORKPLAN Step 0; spec section 8).

`PublicCase` is the ONLY case object the online verifier, the baselines and the
target runner ever receive. It carries no key, trap, rubric or twin metadata, and
`extra="forbid"` stops anyone from bolting them on.

The `TargetRun`, `ClaimVerdict` and `VerifierRun` fields required by spec section 8
are kept. Optional fields are added where spec section 1 (outputs) or section 4
(Path A logging) asks for more; every addition has a default so section-8 JSON
validates unchanged. Changing this file affects both lanes: coordinate first.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

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
    # verifier-side enrichments (source_index); optional for everyone else
    timestamp: Optional[str] = None  # ISO-8601, as written or inherited from the note header
    timestamp_inherited: bool = False
    line_no: int = 0  # 0-based position inside the file


class PublicCase(BaseModel):
    """Everything the online path may see: id, question, source text, and its hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    question: str
    sources: dict[str, str]  # file name (e.g. "lab_results.txt") -> full file text
    source_sha256: str


# ---------------------------------------------------------------- outputs


class Citation(BaseModel):
    file: str
    line_id: str
    quote: str = ""
    quote_verified: bool = True  # verifier: the checker's quote matched the source line


class ClaimVerdict(BaseModel):
    claim_id: str
    claim_text: str
    status: ClaimStatus
    severity: Severity = Severity.material
    citations: list[Citation] = Field(default_factory=list)
    why: str = ""
    uncertainty: str = ""
    # verifier audit trail
    traps: list[Trap] = Field(default_factory=list)
    needs_review: bool = False
    needs_review_reasons: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)


class OmissionFinding(BaseModel):
    """Material evidence in the chart that the answer did not account for (verifier global sweep)."""

    description: str
    kind: str  # contradiction | uncertainty | negative_finding | status_caveat | other
    citations: list[Citation] = Field(default_factory=list)
    material: bool = False
    confirmed: bool = False
    # confirmed | rejected (second check disagreed) | unverifiable (no real line carries it)
    # | failed (second check errored) | not_checked (not a material contradiction)
    confirm_status: str = "not_checked"
    related_claim_id: Optional[str] = None


class VerifierRun(BaseModel):
    # spec section 8
    target_run_id: Optional[str] = None
    verifier_variant: str
    overall: Overall
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    total_model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: int = 0
    raw_model_outputs_path: Optional[str] = None
    # additions (spec section 1 outputs 3, 5, 6)
    case_id: Optional[str] = None
    traps_found: list[Trap] = Field(default_factory=list)
    explanation: str = ""
    needs_review_reasons: list[str] = Field(default_factory=list)
    omissions: list[OmissionFinding] = Field(default_factory=list)
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    models_used: list[str] = Field(default_factory=list)
    fake_backend: bool = False  # True only in offline unit tests; never report these as results


class TargetRun(BaseModel):
    # spec section 8
    run_id: str
    case_id: str
    case_version_sha256: str
    model: str
    timestamp_utc: str
    exact_prompt: str
    answer_raw: str
    first_attempt: bool = True
    stop_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    # additions required by spec section 4, Path A step 5
    exact_system_prompt: Optional[str] = None
    request_config: dict[str, Any] = Field(default_factory=dict)
    response_model: Optional[str] = None
    stop_details: Optional[dict[str, Any]] = None
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    estimated_cost_usd: Optional[float] = None
    is_twin: bool = False
