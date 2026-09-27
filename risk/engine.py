"""Deterministic risk engine: fixed-fractional sizing, trade vetting and drawdown halt.

Nothing here depends on the LLM. A signal is only tradable if every check passes,
and the quantity is always rounded *down* so realised risk never exceeds the cap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from config.settings import RiskConfig
from strategy.signal import Action, TradeSignal


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    quantity: float = 0.0
    risk_amount: float = 0.0
    notional: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def reject(cls, *reasons: str) -> "RiskDecision":
        return cls(approved=False, reasons=tuple(reasons))


class RiskEngine:
    def __init__(self, config: RiskConfig, starting_equity: float, qty_step: float = 0.0):
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        if qty_step < 0:
            raise ValueError("qty_step must be >= 0")
        self.config = config
        self.qty_step = qty_step
        self.peak_equity = starting_equity
        self.equity = starting_equity
        self.open_positions = 0
        self._halted = False

    # ---- drawdown protection -------------------------------------------------
    @property
    def drawdown(self) -> float:
        return 1 - self.equity / self.peak_equity

    @property
    def halted(self) -> bool:
        return self._halted

    def update_equity(self, equity: float) -> None:
        """Record latest equity. Breaching max drawdown latches a halt until reset."""
        if equity < 0:
            raise ValueError("equity cannot be negative")
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if self.drawdown >= self.config.max_drawdown - 1e-9:
            self._halted = True

    def reset_halt(self) -> None:
        """Manual override after review; re-baselines the peak to current equity."""
        self._halted = False
        self.peak_equity = self.equity

    # ---- position bookkeeping ------------------------------------------------
    def register_open(self) -> None:
        self.open_positions += 1

    def register_close(self) -> None:
        self.open_positions = max(0, self.open_positions - 1)

    # ---- sizing ----------------------------------------------------------------
    def _floor_to_step(self, qty: float) -> float:
        if self.qty_step <= 0:
            return qty
        return math.floor(qty / self.qty_step + 1e-9) * self.qty_step

    def position_size(self, equity: float, entry: float, stop_loss: float) -> float:
        """Units such that a stop-out loses at most risk_per_trade * equity,
        further capped so notional exposure <= max_position_pct * equity."""
        risk_per_unit = abs(entry - stop_loss)
        if equity <= 0 or entry <= 0 or risk_per_unit == 0:
            return 0.0
        qty_by_risk = equity * self.config.risk_per_trade / risk_per_unit
        qty_by_notional = equity * self.config.max_position_pct / entry
        return max(0.0, self._floor_to_step(min(qty_by_risk, qty_by_notional)))

    # ---- vetting ---------------------------------------------------------------
    def evaluate(self, signal: TradeSignal, equity: float | None = None) -> RiskDecision:
        equity = self.equity if equity is None else equity
        if self._halted:
            return RiskDecision.reject(
                f"trading halted: drawdown {self.drawdown:.2%} >= {self.config.max_drawdown:.2%}"
            )
        if signal.action is Action.HOLD:
            return RiskDecision.reject("signal is HOLD")

        reasons = []
        if signal.confidence < self.config.min_confidence:
            reasons.append(f"confidence {signal.confidence:.2f} < {self.config.min_confidence:.2f}")
        if signal.reward_risk < self.config.min_reward_risk:
            reasons.append(f"reward/risk {signal.reward_risk:.2f} < {self.config.min_reward_risk:.2f}")
        if self.open_positions >= self.config.max_open_positions:
            reasons.append(f"max open positions ({self.config.max_open_positions}) reached")
        if reasons:
            return RiskDecision.reject(*reasons)

        qty = self.position_size(equity, signal.entry, signal.stop_loss)
        if qty <= 0:
            return RiskDecision.reject("computed position size is zero")

        risk_amount = qty * signal.risk_per_unit
        # Defensive invariant: never exceed the configured per-trade risk.
        max_risk = equity * self.config.risk_per_trade
        if risk_amount > max_risk * (1 + 1e-9):
            return RiskDecision.reject("sized risk exceeds per-trade limit")

        return RiskDecision(
            approved=True,
            quantity=qty,
            risk_amount=risk_amount,
            notional=qty * signal.entry,
            reasons=("approved",),
        )

    @staticmethod
    def stop_triggered(action: Action, stop_loss: float, price: float) -> bool:
        """Hard stop check for an open position at the given market price."""
        if action is Action.BUY:
            return price <= stop_loss
        if action is Action.SELL:
            return price >= stop_loss
        return False
