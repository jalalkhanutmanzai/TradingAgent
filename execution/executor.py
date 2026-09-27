"""Execution layer: turns an approved risk decision into a filled order.

Two implementations share one interface:
- PaperExecutor: simulated fills against the RiskEngine's tracked equity, no
  network calls. Safe default for development and forward-testing.
- LiveExecutor: places real orders through a CCXT-compatible exchange client
  (works against any CCXT exchange, sandbox or live per ExchangeConfig).

Neither one second-guesses risk: callers pass a signal and quantity that have
already been approved by RiskEngine.evaluate(); this layer only handles the
mechanics of filling, position bookkeeping and stop/target monitoring.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import ccxt

from risk.engine import RiskEngine
from strategy.signal import Action, TradeSignal

logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """Raised when an order cannot be placed or an invalid state is requested."""


class OrderStatus(str, Enum):
    FILLED = "FILLED"


@dataclass(frozen=True)
class Order:
    id: str
    symbol: str
    side: Action
    quantity: float
    price: float
    status: OrderStatus
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class Position:
    symbol: str
    side: Action
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float


class BaseExecutor:
    """Shared position bookkeeping and stop/target monitoring.

    Subclasses implement `_fill`, the actual order placement mechanics.
    """

    def __init__(self, risk_engine: RiskEngine):
        self.risk_engine = risk_engine
        self.positions: dict[str, Position] = {}
        self.fills: list[Order] = []

    def _fill(self, symbol: str, side: Action, quantity: float, reference_price: float) -> float:
        """Place the order and return the actual fill price."""
        raise NotImplementedError

    def open_position(self, signal: TradeSignal, quantity: float) -> Order:
        if not signal.is_actionable:
            raise ValueError("cannot open a position for a HOLD signal")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if signal.symbol in self.positions:
            raise ExecutionError(f"position already open for {signal.symbol}")

        fill_price = self._place(signal.symbol, signal.action, quantity, signal.entry)

        self.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            side=signal.action,
            quantity=quantity,
            entry_price=fill_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        self.risk_engine.register_open()
        order = Order(
            id=str(uuid.uuid4()), symbol=signal.symbol, side=signal.action,
            quantity=quantity, price=fill_price, status=OrderStatus.FILLED, reason="entry",
        )
        self.fills.append(order)
        return order

    def close_position(self, symbol: str, price: float, reason: str = "manual") -> Order:
        position = self.positions.get(symbol)
        if position is None:
            raise ExecutionError(f"no open position for {symbol}")

        exit_side = Action.SELL if position.side is Action.BUY else Action.BUY
        fill_price = self._place(symbol, exit_side, position.quantity, price)

        del self.positions[symbol]
        self.risk_engine.register_close()
        self._on_close(position, fill_price)

        order = Order(
            id=str(uuid.uuid4()), symbol=symbol, side=exit_side, quantity=position.quantity,
            price=fill_price, status=OrderStatus.FILLED, reason=reason,
        )
        self.fills.append(order)
        return order

    def check_stops(self, prices: dict[str, float]) -> list[Order]:
        """Close any open position whose stop-loss or take-profit has been hit."""
        closed = []
        for symbol, position in list(self.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            if RiskEngine.stop_triggered(position.side, position.stop_loss, price):
                closed.append(self.close_position(symbol, position.stop_loss, "stop_loss"))
            elif self._take_profit_triggered(position, price):
                closed.append(self.close_position(symbol, position.take_profit, "take_profit"))
        return closed

    @staticmethod
    def _take_profit_triggered(position: Position, price: float) -> bool:
        if position.side is Action.BUY:
            return price >= position.take_profit
        if position.side is Action.SELL:
            return price <= position.take_profit
        return False

    def _on_close(self, position: Position, exit_price: float) -> None:
        """Hook for subclasses to realise PnL / refresh equity. No-op by default."""

    def _place(self, symbol: str, side: Action, quantity: float, reference_price: float) -> float:
        try:
            return self._fill(symbol, side, quantity, reference_price)
        except ccxt.BaseError as exc:
            logger.error("Order rejected for %s %s %.8f: %s", side.value, symbol, quantity, exc)
            raise ExecutionError(str(exc)) from exc


class PaperExecutor(BaseExecutor):
    """Simulated fills at the signal's reference price, with optional slippage/fees.

    Equity only moves on realised PnL at close, mirroring spot trading without
    margin: opening a position does not itself change tracked equity.
    """

    def __init__(self, risk_engine: RiskEngine, slippage_bps: float = 0.0, fee_bps: float = 0.0):
        if slippage_bps < 0 or fee_bps < 0:
            raise ValueError("slippage_bps and fee_bps must be >= 0")
        super().__init__(risk_engine)
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps

    def _fill(self, symbol: str, side: Action, quantity: float, reference_price: float) -> float:
        if reference_price is None or reference_price <= 0:
            raise ExecutionError(f"invalid reference price for {symbol}: {reference_price}")
        slip = reference_price * self.slippage_bps / 10_000
        # Slippage always works against the trader: buys fill higher, sells fill lower.
        return reference_price + slip if side is Action.BUY else reference_price - slip

    def _on_close(self, position: Position, exit_price: float) -> None:
        gross = (exit_price - position.entry_price) * position.quantity
        if position.side is Action.SELL:
            gross = -gross
        fees = (position.entry_price + exit_price) * position.quantity * self.fee_bps / 10_000
        pnl = gross - fees
        self.risk_engine.update_equity(self.risk_engine.equity + pnl)


class LiveExecutor(BaseExecutor):
    """Places real market orders through a CCXT-compatible exchange client."""

    def __init__(self, risk_engine: RiskEngine, exchange: ccxt.Exchange):
        super().__init__(risk_engine)
        self.exchange = exchange

    def _fill(self, symbol: str, side: Action, quantity: float, reference_price: float) -> float:
        order = self.exchange.create_order(symbol, "market", side.value.lower(), quantity)
        price = order.get("average") or order.get("price") or reference_price
        if not price or price <= 0:
            raise ExecutionError(f"exchange returned no usable fill price for {symbol}")
        return float(price)
