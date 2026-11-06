import pytest

from medproof import llm
from medproof.llm import LLMResult, Meter, cost_usd


def test_haiku_list_price():
    # under the 100k-prompt tier: $0.10 / $0.50 per MTok
    assert cost_usd("claude-haiku-5-5", 50_000, 10_000) == pytest.approx((50_000 * 0.10 + 10_000 * 0.50) / 1e6)


def test_opus_and_sonnet_list_price():
    assert cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.00)
    assert cost_usd("claude-opus-5-5", 0, 1_000_000) == pytest.approx(20.00)
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.00)


def test_cache_tokens_priced_separately():
    c = cost_usd("claude-haiku-5-5", 1000, 0, cache_read_input_tokens=10_000, cache_creation_input_tokens=2000)
    assert c == pytest.approx((1000 * 0.10 + 10_000 * 0.01 + 2000 * 0.125) / 1e6)


def test_haiku_long_context_tier():
    assert cost_usd("claude-haiku-5-5", 150_000, 1000) == pytest.approx((150_000 * 0.50 + 1000 * 2.50) / 1e6)


def test_unknown_model_refuses_to_guess():
    with pytest.raises(KeyError):
        cost_usd("some-other-model", 1, 1)


def test_meter_sums_calls():
    m = Meter()
    for i in range(3):
        m.add(LLMResult(model="claude-haiku-5-5", text="", parsed=None, stop_reason="end_turn",
                        input_tokens=100, output_tokens=10, cost_usd=0.001))
    assert m.n_calls == 3
    assert m.total("input_tokens") == 300
    assert m.cost_usd == pytest.approx(0.003)
    assert not m.any_fake


async def test_fake_backend_is_flagged_and_free():
    llm.use_fake_backend(lambda req: "hello")
    m = Meter()
    r = await llm.call("claude-haiku-5-5", "s", "u", meter=m)
    assert r.fake and r.cost_usd == 0 and m.any_fake
