import ccxt
import pytest

from config.settings import RiskConfig
from execution.executor import ExecutionError, LiveExecutor, PaperExecutor
from risk.engine import RiskEngine
from strategy.signal import Action, TradeSignal


def buy(symbol="X", entry=100.0, sl=98.0, tp=104.0, conf=0.7):
    return TradeSignal(symbol=symbol, action=Action.BUY, entry=entry, stop_loss=sl, take_profit=tp, confidence=conf)


def sell(symbol="X", entry=100.0, sl=102.0, tp=96.0, conf=0.7):
    return TradeSignal(symbol=symbol, action=Action.SELL, entry=entry, stop_loss=sl, take_profit=tp, confidence=conf)


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(risk_per_trade=0.01, max_drawdown=0.10, max_position_pct=0.5,
                      min_reward_risk=1.5, min_confidence=0.55, max_open_positions=2)


@pytest.fixture
def engine(risk_config) -> RiskEngine:
    return RiskEngine(risk_config, 10_000)


# ---- PaperExecutor: opening -----------------------------------------------------

def test_paper_open_position_fills_at_entry_and_registers(engine):
    ex = PaperExecutor(engine)
    order = ex.open_position(buy(), quantity=10)

    assert order.side is Action.BUY and order.quantity == 10 and order.price == pytest.approx(100.0)
    assert ex.positions["X"].entry_price == pytest.approx(100.0)
    assert engine.open_positions == 1


def test_paper_open_hold_signal_rejected(engine):
    ex = PaperExecutor(engine)
    with pytest.raises(ValueError):
        ex.open_position(TradeSignal.hold("X", "flat"), quantity=10)


def test_paper_open_nonpositive_quantity_rejected(engine):
    ex = PaperExecutor(engine)
    with pytest.raises(ValueError):
        ex.open_position(buy(), quantity=0)


def test_paper_duplicate_symbol_rejected(engine):
    ex = PaperExecutor(engine)
    ex.open_position(buy(), quantity=10)
    with pytest.raises(ExecutionError):
        ex.open_position(buy(), quantity=5)


def test_paper_invalid_close_price_rejected(engine):
    ex = PaperExecutor(engine)
    ex.open_position(buy(), quantity=10)
    with pytest.raises(ExecutionError):
        ex.close_position("X", price=0.0)


# ---- PaperExecutor: closing / PnL -----------------------------------------------

def test_paper_close_buy_profit_updates_equity(engine):
    ex = PaperExecutor(engine)
    ex.open_position(buy(entry=100.0), quantity=10)
    order = ex.close_position("X", price=110.0, reason="manual")

    assert order.side is Action.SELL and order.price == pytest.approx(110.0)
    assert "X" not in ex.positions
    assert engine.open_positions == 0
    assert engine.equity == pytest.approx(10_000 + 100.0)  # +10 units * 10 gain


def test_paper_close_buy_loss_updates_equity(engine):
    ex = PaperExecutor(engine)
    ex.open_position(buy(entry=100.0), quantity=10)
    ex.close_position("X", price=95.0)

    assert engine.equity == pytest.approx(10_000 - 50.0)  # 10 units * -5


def test_paper_close_sell_profit_updates_equity(engine):
    ex = PaperExecutor(engine)
    ex.open_position(sell(entry=100.0), quantity=10)
    ex.close_position("X", price=90.0)

    assert engine.equity == pytest.approx(10_000 + 100.0)  # short gains as price falls


def test_paper_fees_reduce_pnl(engine):
    ex = PaperExecutor(engine, fee_bps=10.0)  # 0.10% per side, on notional
    ex.open_position(buy(entry=100.0), quantity=10)
    ex.close_position("X", price=110.0)

    fees = (100.0 + 110.0) * 10 * 10.0 / 10_000
    assert engine.equity == pytest.approx(10_000 + 100.0 - fees)


def test_paper_slippage_worsens_fill_price(engine):
    ex = PaperExecutor(engine, slippage_bps=50.0)  # 0.5%
    buy_order = ex.open_position(buy(entry=100.0), quantity=10)
    assert buy_order.price == pytest.approx(100.5)

    sell_order = ex.close_position("X", price=100.0)
    assert sell_order.price == pytest.approx(99.5)


def test_close_nonexistent_position_raises(engine):
    ex = PaperExecutor(engine)
    with pytest.raises(ExecutionError):
        ex.close_position("X", price=100.0)


# ---- PaperExecutor: stop/target monitoring --------------------------------------

def test_check_stops_triggers_stop_loss(engine):
    ex = PaperExecutor(engine)
    ex.open_position(buy(entry=100.0, sl=98.0, tp=104.0), quantity=10)
    closed = ex.check_stops({"X": 97.5})

    assert len(closed) == 1 and closed[0].reason == "stop_loss"
    assert closed[0].price == pytest.approx(98.0)
    assert "X" not in ex.positions


def test_check_stops_triggers_take_profit(engine):
    ex = PaperExecutor(engine)
    ex.open_position(buy(entry=100.0, sl=98.0, tp=104.0), quantity=10)
    closed = ex.check_stops({"X": 105.0})

    assert len(closed) == 1 and closed[0].reason == "take_profit"
    assert closed[0].price == pytest.approx(104.0)


def test_check_stops_for_sell_position(engine):
    ex = PaperExecutor(engine)
    ex.open_position(sell(entry=100.0, sl=102.0, tp=96.0), quantity=10)
    assert ex.check_stops({"X": 103.0})[0].reason == "stop_loss"


def test_check_stops_noop_when_price_between_levels(engine):
    ex = PaperExecutor(engine)
    ex.open_position(buy(entry=100.0, sl=98.0, tp=104.0), quantity=10)
    assert ex.check_stops({"X": 101.0}) == []
    assert "X" in ex.positions


def test_check_stops_ignores_symbols_without_a_price(engine):
    ex = PaperExecutor(engine)
    ex.open_position(buy(), quantity=10)
    assert ex.check_stops({"OTHER": 50.0}) == []


# ---- LiveExecutor (CCXT-compatible order placement, mocked client) -------------

class FakeExchange:
    def __init__(self, fill_price=None, error=None):
        self.fill_price = fill_price
        self.error = error
        self.calls: list[dict] = []

    def create_order(self, symbol, type, side, amount):
        self.calls.append({"symbol": symbol, "type": type, "side": side, "amount": amount})
        if self.error is not None:
            raise self.error
        return {"average": self.fill_price, "price": self.fill_price}


def test_live_open_position_places_market_order(engine):
    fake = FakeExchange(fill_price=101.2)
    ex = LiveExecutor(engine, fake)
    order = ex.open_position(buy(entry=100.0), quantity=3)

    assert order.price == pytest.approx(101.2)
    call = fake.calls[0]
    assert call == {"symbol": "X", "type": "market", "side": "buy", "amount": 3}


def test_live_close_position_places_opposite_side_order(engine):
    fake = FakeExchange(fill_price=100.0)
    ex = LiveExecutor(engine, fake)
    ex.open_position(buy(entry=100.0), quantity=3)
    fake.fill_price = 105.0
    ex.close_position("X", price=105.0)

    assert fake.calls[1]["side"] == "sell"


def test_live_falls_back_to_reference_price_when_exchange_omits_fill_price(engine):
    fake = FakeExchange(fill_price=None)
    ex = LiveExecutor(engine, fake)
    order = ex.open_position(buy(entry=100.0), quantity=3)
    assert order.price == pytest.approx(100.0)


@pytest.mark.parametrize("error", [
    ccxt.InsufficientFunds("not enough balance"),
    ccxt.InvalidOrder("bad order"),
    ccxt.NetworkError("timeout"),
])
def test_live_wraps_ccxt_errors_as_execution_error(engine, error):
    fake = FakeExchange(error=error)
    ex = LiveExecutor(engine, fake)
    with pytest.raises(ExecutionError):
        ex.open_position(buy(), quantity=3)
    assert "X" not in ex.positions
    assert engine.open_positions == 0
