"""Single entry point for every Anthropic call (target runner, verifier, baselines).

Responsibilities:
- structured output via `output_config.format` (schema from a pydantic model);
- usage + cost metering from the provider's `usage` block (never estimated);
- concurrency cap (MAX_PARALLEL_CHECKS) and SDK-level retries with backoff;
- refusal / truncation surfaced in `stop_reason`, never hidden;
- an explicit offline fake backend for unit tests, flagged on every result.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
from pydantic import BaseModel, ValidationError

# USD per million tokens, Anthropic first-party list prices (see README for sources).
# cache_write = 5-minute TTL write rate (1.25x input). Haiku 5.5 has a higher tier
# for prompts > 100k tokens.
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
}
LONG_CONTEXT_PRICES: dict[str, tuple[int, dict[str, float]]] = {
    "claude-haiku-5-5": (
        100_000,
        {"input": 0.50, "output": 2.50, "cache_read": 0.05, "cache_write": 0.625},
    ),
}

VERIFY_MODEL = os.environ.get("VERIFY_MODEL", "claude-haiku-5-5")
HAIKU_EFFORT = os.environ.get("HAIKU_EFFORT", "low")


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float:
    """Cost of one call from its measured usage. Thinking tokens are included in output_tokens."""
    if model not in PRICES:
        raise KeyError(f"No price configured for model {model!r}; add it to PRICES")
    rates = PRICES[model]
    if model in LONG_CONTEXT_PRICES:
        threshold, long_rates = LONG_CONTEXT_PRICES[model]
        if input_tokens + cache_read_input_tokens + cache_creation_input_tokens > threshold:
            rates = long_rates
    return (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_read_input_tokens * rates["cache_read"]
        + cache_creation_input_tokens * rates["cache_write"]
    ) / 1_000_000


@dataclass
class LLMResult:
    """Contract shared with the benchmark lane: `model` is the id requested, `response_model`
    the id the API reported (they differ only if the provider routed elsewhere)."""

    model: str
    text: str | None = None
    parsed: Any = None
    stop_reason: str | None = None
    response_model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    parse_error: str | None = None
    stop_details: dict | None = None
    fake: bool = False
    label: str = ""
    request: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.stop_reason in ("end_turn", "stop_sequence") and self.parse_error is None

    def model_dump(self, exclude: set[str] | None = None, **_: Any) -> dict:
        """pydantic-style dump (benchmark lane contract): every field, JSON-safe."""
        out = {
            "model": self.model, "response_model": self.response_model, "text": self.text,
            "parsed": self.parsed.model_dump(mode="json") if isinstance(self.parsed, BaseModel) else self.parsed,
            "stop_reason": self.stop_reason, "stop_details": self.stop_details,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "latency_ms": self.latency_ms, "cost_usd": self.cost_usd, "parse_error": self.parse_error,
            "fake": self.fake, "label": self.label, "request": self.request, "raw": self.raw,
        }
        return {k: v for k, v in out.items() if k not in (exclude or set())}

    def to_log(self) -> dict:
        return {
            "label": self.label,
            "model": self.model,
            "response_model": self.response_model,
            "stop_reason": self.stop_reason,
            "stop_details": self.stop_details,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "parse_error": self.parse_error,
            "fake": self.fake,
            "request": self.request,
            "text": self.text,
            "parsed": self.parsed.model_dump(mode="json") if isinstance(self.parsed, BaseModel) else self.parsed,
        }


class Meter:
    """Accumulates every call made during one verification (or one target run)."""

    def __init__(self) -> None:
        self.calls: list[LLMResult] = []

    def add(self, r: LLMResult) -> LLMResult:
        self.calls.append(r)
        return r

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def total(self, attr: str) -> float:
        return sum(getattr(c, attr) for c in self.calls)

    @property
    def cost_usd(self) -> float:
        return round(self.total("cost_usd"), 8)

    @property
    def any_fake(self) -> bool:
        return any(c.fake for c in self.calls)

    @property
    def models(self) -> list[str]:
        return sorted({c.response_model or c.model for c in self.calls})


# ---------------------------------------------------------------- backends

FakeHandler = Callable[[dict], "str | dict | BaseModel"]
_fake_handler: FakeHandler | None = None
_client: anthropic.AsyncAnthropic | None = None
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def use_fake_backend(handler: FakeHandler | None) -> None:
    """Route calls to `handler(request) -> json-able output` (tests only). None restores the real API."""
    global _fake_handler
    _fake_handler = handler


_client_loop: asyncio.AbstractEventLoop | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """One async client per event loop (each sync verify() call runs its own loop)."""
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client_loop is not loop:
        # SDK retries 408/409/429/5xx and connection errors with exponential backoff.
        _client = anthropic.AsyncAnthropic(max_retries=4, timeout=120.0)
        _client_loop = loop
    return _client


async def aclose() -> None:
    """Close the async client bound to the current loop (call before the loop ends)."""
    global _client, _client_loop
    if _client is not None and _client_loop is asyncio.get_running_loop():
        await _client.close()
        _client, _client_loop = None, None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore, _semaphore_loop
    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(int(os.environ.get("MAX_PARALLEL_CHECKS", "5")))
        _semaphore_loop = loop
    return _semaphore


def _parse(schema: type[BaseModel] | None, text: str) -> tuple[BaseModel | None, str | None]:
    if schema is None:
        return None, None
    try:
        return schema.model_validate_json(text), None
    except ValidationError as e:
        return None, f"schema validation failed: {e.errors()[:3]}"
    except ValueError as e:
        return None, f"invalid JSON: {e}"


async def call(
    model: str,
    system: str | None,
    user: str,
    *,
    schema: type[BaseModel] | None = None,
    effort: str | None = None,
    max_tokens: int = 2048,
    label: str = "",
    meter: Meter | None = None,
    extra: dict[str, Any] | None = None,
) -> LLMResult:
    """One fresh, single-turn Messages call. Returns an LLMResult; never raises on refusal/truncation.

    - `system=None` sends no system prompt; `effort=None` sends no effort (model default config).
    - Only transient transport/429/5xx errors are retried (by the SDK, before any response exists);
      a completed response is never retried here and there is no fallback to another model.
    """
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system is not None:
        request["system"] = system
    output_config: dict[str, Any] = {}
    if effort:
        output_config["effort"] = effort
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": anthropic.transform_schema(schema)}
    if output_config:
        request["output_config"] = output_config
    if extra:
        request.update(extra)
    log_request = {k: v for k, v in request.items() if k != "output_config"} | {
        "effort": effort,
        "schema": schema.__name__ if schema else None,
    }

    if _fake_handler is not None:
        t0 = time.perf_counter()
        out = _fake_handler({**log_request, "schema_cls": schema, "label": label})
        if isinstance(out, BaseModel):
            text = out.model_dump_json()
        elif isinstance(out, dict):
            text = json.dumps(out)
        else:
            text = str(out)
        parsed, err = _parse(schema, text)
        result = LLMResult(
            model=model, text=text, parsed=parsed, stop_reason="end_turn", parse_error=err,
            latency_ms=int((time.perf_counter() - t0) * 1000), fake=True, label=label,
            request=log_request,
        )
        return meter.add(result) if meter else result

    async with _get_semaphore():
        t0 = time.perf_counter()
        resp = await _get_client().messages.create(**request)
        latency_ms = int((time.perf_counter() - t0) * 1000)

    u = resp.usage
    in_tok = u.input_tokens or 0
    out_tok = u.output_tokens or 0
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    answered_model = resp.model or model  # reported by the API
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    stop_details = getattr(resp, "stop_details", None)

    parsed, err = (None, None)
    if resp.stop_reason == "refusal":
        err = "refusal"
    elif resp.stop_reason == "max_tokens":
        err = "truncated (max_tokens)"
    else:
        parsed, err = _parse(schema, text)

    result = LLMResult(
        model=model,
        response_model=answered_model,
        text=text,
        parsed=parsed,
        stop_reason=resp.stop_reason,
        stop_details=stop_details.model_dump() if hasattr(stop_details, "model_dump") else stop_details,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_input_tokens=cr,
        cache_creation_input_tokens=cw,
        latency_ms=latency_ms,
        cost_usd=cost_usd(answered_model if answered_model in PRICES else model, in_tok, out_tok, cr, cw),
        parse_error=err,
        label=label,
        request=log_request,
        raw=resp.model_dump(mode="json"),
    )
    return meter.add(result) if meter else result


async def check(
    system: str,
    user: str,
    schema: type[BaseModel],
    *,
    label: str,
    meter: Meter | None,
    max_tokens: int = 2048,
    effort: str | None = None,
) -> LLMResult:
    """Convenience wrapper for verifier checks: Haiku, low effort, structured output.

    One retry if the structured output failed to parse (truncation or schema miss),
    with a larger token budget. Both attempts are metered.
    """
    eff = effort or HAIKU_EFFORT
    r = await call(VERIFY_MODEL, system, user, schema=schema, effort=eff,
                   max_tokens=max_tokens, label=label, meter=meter)
    if r.parsed is None and r.stop_reason != "refusal":
        r = await call(VERIFY_MODEL, system, user, schema=schema, effort=eff,
                       max_tokens=max_tokens * 2, label=label + ":retry", meter=meter)
    return r
