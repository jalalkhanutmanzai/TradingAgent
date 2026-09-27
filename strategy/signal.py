"""Structured trade signal shared by all evaluators."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class TradeSignal(BaseModel):
    symbol: str
    action: Action
    entry: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    take_profit: float | None = Field(default=None, gt=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    source: str = "unknown"

    @model_validator(mode="after")
    def _check_price_geometry(self) -> "TradeSignal":
        if self.action is Action.HOLD:
            return self
        if None in (self.entry, self.stop_loss, self.take_profit):
            raise ValueError(f"{self.action.value} signal requires entry, stop_loss and take_profit")
        if self.action is Action.BUY and not (self.stop_loss < self.entry < self.take_profit):
            raise ValueError("BUY requires stop_loss < entry < take_profit")
        if self.action is Action.SELL and not (self.take_profit < self.entry < self.stop_loss):
            raise ValueError("SELL requires take_profit < entry < stop_loss")
        return self

    @property
    def is_actionable(self) -> bool:
        return self.action is not Action.HOLD

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop_loss) if self.is_actionable else 0.0

    @property
    def reward_risk(self) -> float:
        if not self.is_actionable:
            return 0.0
        return abs(self.take_profit - self.entry) / self.risk_per_unit

    @classmethod
    def hold(cls, symbol: str, rationale: str, source: str = "system") -> "TradeSignal":
        return cls(symbol=symbol, action=Action.HOLD, rationale=rationale, source=source)
