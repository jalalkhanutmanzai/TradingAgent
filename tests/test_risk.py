import pytest

from config.settings import RiskConfig
from risk.engine import RiskEngine
from strategy.signal import Action, TradeSignal


def buy(entry=100.0, sl=98.0, tp=104.0, conf=0.7):
    return TradeSignal(symbol="X", action=Action.BUY, entry=entry, stop_loss=sl, take_profit=tp, confidence=conf)


def sell(entry=100.0, sl=102.0, tp=96.0, conf=0.7):
    return TradeSignal(symbol="X", action=Action.SELL, entry=entry, stop_loss=sl, take_profit=tp, confidence=conf)


def test_fixed_fractional_sizing(risk_config):
    eng = RiskEngine(risk_config, 10_000)
    d = eng.evaluate(buy())
    assert d.approved
    # 1% of 10k = 100 risk; 2 per unit -> 50 units; notional 5000 <= 50% cap
    assert d.quantity == pytest.approx(50)
    assert d.risk_amount == pytest.approx(100)
    assert d.notional == pytest.approx(5000)


def test_sell_sizing(risk_config):
    d = RiskEngine(risk_config, 10_000).evaluate(sell())
    assert d.approved and d.quantity == pytest.approx(50)


def test_notional_cap_limits_tight_stops(risk_config):
    # stop 0.1 away -> risk sizing wants 1000 units (100k notional); cap is 50% of 10k.
    d = RiskEngine(risk_config, 10_000).evaluate(buy(sl=99.9, tp=100.5))
    assert d.approved
    assert d.notional == pytest.approx(5_000)
    assert d.risk_amount < 100


@pytest.mark.parametrize("risk_pct", [0.01, 0.015, 0.02])
def test_risk_never_exceeds_limit(risk_pct):
    cfg = RiskConfig(risk_per_trade=risk_pct, max_position_pct=1.0)
    eng = RiskEngine(cfg, 12_345.67, qty_step=0.001)
    for sl in (99.3, 97.77, 90.01, 50.5):
        d = eng.evaluate(buy(sl=sl, tp=100 + 3 * (100 - sl)))
        assert d.approved
        assert d.risk_amount <= 12_345.67 * risk_pct + 1e-9


def test_qty_step_rounds_down(risk_config):
    eng = RiskEngine(risk_config, 10_000, qty_step=0.1)
    d = eng.evaluate(buy(sl=97.0, tp=106.0))  # raw qty 33.33...
    assert d.quantity == pytest.approx(33.3)


def test_rejects_hold(risk_config):
    d = RiskEngine(risk_config, 10_000).evaluate(TradeSignal.hold("X", "flat"))
    assert not d.approved and d.quantity == 0


def test_rejects_low_confidence(risk_config):
    d = RiskEngine(risk_config, 10_000).evaluate(buy(conf=0.4))
    assert not d.approved and any("confidence" in r for r in d.reasons)


def test_rejects_poor_reward_risk(risk_config):
    d = RiskEngine(risk_config, 10_000).evaluate(buy(tp=101.0))  # 0.5 R:R
    assert not d.approved and any("reward/risk" in r for r in d.reasons)


def test_max_open_positions(risk_config):
    eng = RiskEngine(risk_config, 10_000)
    eng.register_open()
    eng.register_open()
    assert not eng.evaluate(buy()).approved
    eng.register_close()
    assert eng.evaluate(buy()).approved


def test_drawdown_halt_latches(risk_config):
    eng = RiskEngine(risk_config, 10_000)
    eng.update_equity(11_000)  # new peak
    eng.update_equity(9_950)   # 9.55% dd -> still ok
    assert not eng.halted and eng.evaluate(buy()).approved
    eng.update_equity(9_900)   # 10% dd -> halt
    assert eng.halted
    d = eng.evaluate(buy())
    assert not d.approved and "halted" in d.reasons[0]
    eng.update_equity(10_500)  # recovery does not auto-resume
    assert eng.halted
    eng.reset_halt()
    assert not eng.halted and eng.peak_equity == 10_500


def test_uses_current_equity_by_default(risk_config):
    eng = RiskEngine(risk_config, 10_000)
    eng.update_equity(9_500)
    assert eng.evaluate(buy()).risk_amount == pytest.approx(95)


def test_stop_triggered():
    assert RiskEngine.stop_triggered(Action.BUY, 95, 94.9)
    assert not RiskEngine.stop_triggered(Action.BUY, 95, 95.1)
    assert RiskEngine.stop_triggered(Action.SELL, 105, 105)
    assert not RiskEngine.stop_triggered(Action.HOLD, 105, 200)


def test_invalid_construction(risk_config):
    with pytest.raises(ValueError):
        RiskEngine(risk_config, 0)
    with pytest.raises(ValueError):
        RiskEngine(risk_config, 100, qty_step=-1)
