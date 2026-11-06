"""Async Anthropic wrapper used for EVERY model call (target runner, verifier, baselines).

STEP 0 STUB. Person 1 (verifier lane, WORKPLAN item 1.1) owns and implements this
file. The signature and the `LLMResult` fields below are the shared contract;
callers in the benchmark lane depend on them exactly as documented.

Contract:
- `model`: exact model id, e.g. "claude-sonnet-5-5".
- `system`: system prompt string, or None to send no system prompt.
- `user`: the single user message text (fresh conversation every call).
- `effort`: "low" | "medium" | "high" | ... sent as `output_config.effort`;
  None means do NOT send effort (the model's default configuration).
- `schema`: a pydantic BaseModel subclass for structured output, or None for text.
  When given, `LLMResult.parsed` is a validated instance (or None if the model
  refused / was truncated / output failed validation).
- `max_tokens`: passed through unchanged.
- Never retries a completed response; retries only transient transport/5xx/429
  errors before any response exists. Never falls back to a different model.
- Refusals (`stop_reason == "refusal"`) and truncation (`"max_tokens"`) are
  returned in the result, never raised or hidden.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class LLMResult(BaseModel):
    model: str                                # model id requested
    response_model: Optional[str] = None      # model id reported by the API response
    text: Optional[str] = None                # concatenated text blocks
    parsed: Optional[Any] = None              # validated schema instance when schema given
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    stop_reason: Optional[str] = None
    stop_details: Optional[dict[str, Any]] = None
    latency_ms: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)  # full provider response (model_dump)


async def call(
    model: str,
    system: Optional[str],
    user: str,
    *,
    effort: Optional[str] = None,
    schema: Optional[type[BaseModel]] = None,
    max_tokens: int = 1024,
) -> LLMResult:
    raise NotImplementedError(
        "medproof.llm.call is a Step 0 stub; Person 1 implements it (WORKPLAN 1.1)."
    )
