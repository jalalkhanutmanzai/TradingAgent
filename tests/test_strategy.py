import json

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

from config.settings import LLMConfig
from indicators.technical import add_all
from strategy.llm_evaluator import (
    FALLBACK_BETA,
    LLMSignalEvaluator,
    LLMSignalOutput,
    build_market_snapshot,
)
from strategy.rule_based import RuleBasedEvaluator
from strategy.signal import Action, TradeSignal
from tests.conftest import fake_anthropic_client, fake_parsed_response

_REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


# ---- TradeSignal ----------------------------------------------------------------

def test_signal_valid_buy_and_metrics():
    s = TradeSignal(symbol="X", action=Action.BUY, entry=100, stop_loss=95, take_profit=110, confidence=0.7)
    assert s.risk_per_unit == 5
    assert s.reward_risk == pytest.approx(2.0)


@pytest.mark.parametrize("action,entry,sl,tp", [
    (Action.BUY, 100, 105, 110),   # stop above entry
    (Action.BUY, 100, 95, 99),     # target below entry
    (Action.SELL, 100, 95, 90),    # stop below entry
    (Action.SELL, 100, 105, 101 + 10),  # target above entry
    (Action.BUY, 100, None, 110),  # missing stop
])
def test_signal_rejects_bad_geometry(action, entry, sl, tp):
    with pytest.raises(ValidationError):
        TradeSignal(symbol="X", action=action, entry=entry, stop_loss=sl, take_profit=tp)


def test_hold_needs_no_prices():
    s = TradeSignal.hold("X", "nothing to do")
    assert not s.is_actionable and s.reward_risk == 0


# ---- Rule-based evaluator ---------------------------------------------------------

def test_rule_based_buy_in_uptrend(ohlcv_uptrend):
    df = add_all(ohlcv_uptrend)
    # Uptrend is strong enough to push RSI > 70; relax the filter for this check.
    sig = RuleBasedEvaluator(rsi_overbought=101).evaluate(df, "BTC/USDT")
    assert sig.action is Action.BUY
    assert sig.stop_loss < sig.entry < sig.take_profit


def test_rule_based_sell_in_downtrend(ohlcv_downtrend):
    df = add_all(ohlcv_downtrend)
    sig = RuleBasedEvaluator(rsi_oversold=-1).evaluate(df, "BTC/USDT")
    assert sig.action is Action.SELL
    assert sig.take_profit < sig.entry < sig.stop_loss


def test_rule_based_hold_when_not_warmed_up(ohlcv_flat):
    df = add_all(ohlcv_flat.head(10))
    assert RuleBasedEvaluator().evaluate(df, "BTC/USDT").action is Action.HOLD


def test_rule_based_hold_without_indicators(ohlcv_flat):
    assert RuleBasedEvaluator().evaluate(ohlcv_flat, "BTC/USDT").action is Action.HOLD


# ---- LLM evaluator (mocked client) ------------------------------------------------

def _llm(result=None, error=None, **cfg):
    client, messages = fake_anthropic_client(result=result, error=error)
    return LLMSignalEvaluator(LLMConfig(**cfg), client=client), messages


def test_snapshot_is_json_serialisable(ohlcv_flat):
    snap = build_market_snapshot(add_all(ohlcv_flat), "BTC/USDT")
    json.dumps(snap)
    assert snap["symbol"] == "BTC/USDT"
    assert len(snap["recent_closes"]) == 20
    assert "rsi" in snap["latest"] and "atr" in snap["latest"]


def test_llm_buy_signal_and_request_shape(ohlcv_flat):
    df = add_all(ohlcv_flat)
    close = float(df["close"].iloc[-1])
    out = LLMSignalOutput(action="BUY", entry=close, stop_loss=close * 0.97,
                          take_profit=close * 1.06, confidence=0.72, rationale="trend up")
    ev, messages = _llm(result=fake_parsed_response(out))
    sig = ev.evaluate(df, "BTC/USDT")

    assert sig.action is Action.BUY and sig.source == "llm"
    assert sig.confidence == pytest.approx(0.72)
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_format"] is LLMSignalOutput
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert json.loads(call["messages"][0]["content"])["symbol"] == "BTC/USDT"


def test_llm_without_fallbacks_omits_beta(ohlcv_flat):
    out = LLMSignalOutput(action="HOLD", entry=None, stop_loss=None, take_profit=None,
                          confidence=0.3, rationale="mixed")
    ev, messages = _llm(result=fake_parsed_response(out), use_fallbacks=False)
    assert ev.evaluate(add_all(ohlcv_flat), "BTC/USDT").action is Action.HOLD
    assert "betas" not in messages.calls[0] and "fallbacks" not in messages.calls[0]


def test_llm_invalid_levels_fail_closed(ohlcv_flat):
    out = LLMSignalOutput(action="BUY", entry=100, stop_loss=110, take_profit=120,
                          confidence=0.9, rationale="bad stop")
    ev, _ = _llm(result=fake_parsed_response(out))
    sig = ev.evaluate(add_all(ohlcv_flat), "BTC/USDT")
    assert sig.action is Action.HOLD and "invalid" in sig.rationale


def test_llm_confidence_is_clamped(ohlcv_flat):
    out = LLMSignalOutput(action="SELL", entry=100, stop_loss=103, take_profit=94,
                          confidence=1.7, rationale="x")
    ev, _ = _llm(result=fake_parsed_response(out))
    assert ev.evaluate(add_all(ohlcv_flat), "BTC/USDT").confidence == 1.0


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_llm_bad_stop_reason_holds(ohlcv_flat, stop_reason):
    ev, _ = _llm(result=fake_parsed_response(None, stop_reason=stop_reason))
    assert ev.evaluate(add_all(ohlcv_flat), "BTC/USDT").action is Action.HOLD


def test_llm_missing_parsed_output_holds(ohlcv_flat):
    ev, _ = _llm(result=fake_parsed_response(None))
    assert ev.evaluate(add_all(ohlcv_flat), "BTC/USDT").action is Action.HOLD


@pytest.mark.parametrize("error", [
    anthropic.RateLimitError("rl", response=httpx2.Response(429, request=_REQ), body=None),
    anthropic.InternalServerError("ise", response=httpx2.Response(500, request=_REQ), body=None),
    anthropic.BadRequestError("bad", response=httpx2.Response(400, request=_REQ), body=None),
    anthropic.APIConnectionError(request=_REQ),
])
def test_llm_api_errors_fail_closed(ohlcv_flat, error):
    ev, _ = _llm(error=error)
    sig = ev.evaluate(add_all(ohlcv_flat), "BTC/USDT")
    assert sig.action is Action.HOLD and sig.source == "llm"


def test_llm_empty_frame_holds(ohlcv_flat):
    ev, messages = _llm()
    assert ev.evaluate(ohlcv_flat.iloc[0:0], "BTC/USDT").action is Action.HOLD
    assert messages.calls == []
