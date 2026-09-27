"""Central configuration: exchange credentials, LLM settings and risk thresholds.

Values are read from environment variables so that secrets never live in code.
Risk limits are validated on construction and clamped to hard ceilings that
cannot be overridden from the environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Absolute ceilings. Any configured value above these is rejected.
HARD_MAX_RISK_PER_TRADE = 0.02  # 2% of equity per trade
HARD_MAX_DRAWDOWN = 0.25  # 25% peak-to-trough


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ExchangeConfig:
    exchange_id: str = "binance"
    api_key: str | None = None
    api_secret: str | None = None
    sandbox: bool = True  # default to testnet; live trading must be opted into
    default_type: str = "spot"
    timeout_ms: int = 30_000


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = True
    model: str = "claude-opus-5"
    max_tokens: int = 16_000
    # Server-side refusal fallbacks (routes a declined request to another model).
    use_fallbacks: bool = True


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01  # fraction of equity risked per trade
    max_drawdown: float = 0.10  # halt trading beyond this drawdown from peak
    max_position_pct: float = 0.25  # max notional exposure per position vs equity
    min_reward_risk: float = 1.5  # minimum TP distance / SL distance
    min_confidence: float = 0.55  # minimum signal confidence to trade
    max_open_positions: int = 3

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade <= HARD_MAX_RISK_PER_TRADE:
            raise ValueError(
                f"risk_per_trade must be in (0, {HARD_MAX_RISK_PER_TRADE}], got {self.risk_per_trade}"
            )
        if not 0 < self.max_drawdown <= HARD_MAX_DRAWDOWN:
            raise ValueError(
                f"max_drawdown must be in (0, {HARD_MAX_DRAWDOWN}], got {self.max_drawdown}"
            )
        if not 0 < self.max_position_pct <= 1:
            raise ValueError(f"max_position_pct must be in (0, 1], got {self.max_position_pct}")
        if self.min_reward_risk <= 0:
            raise ValueError("min_reward_risk must be positive")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be >= 1")


@dataclass(frozen=True)
class Settings:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            symbol=os.getenv("TRADING_SYMBOL", "BTC/USDT"),
            timeframe=os.getenv("TRADING_TIMEFRAME", "1h"),
            exchange=ExchangeConfig(
                exchange_id=os.getenv("EXCHANGE_ID", "binance"),
                api_key=os.getenv("EXCHANGE_API_KEY") or None,
                api_secret=os.getenv("EXCHANGE_API_SECRET") or None,
                sandbox=_env_bool("EXCHANGE_SANDBOX", True),
                default_type=os.getenv("EXCHANGE_DEFAULT_TYPE", "spot"),
            ),
            llm=LLMConfig(
                enabled=_env_bool("LLM_ENABLED", True),
                model=os.getenv("LLM_MODEL", "claude-opus-5"),
                use_fallbacks=_env_bool("LLM_USE_FALLBACKS", True),
            ),
            risk=RiskConfig(
                risk_per_trade=_env_float("RISK_PER_TRADE", 0.01),
                max_drawdown=_env_float("RISK_MAX_DRAWDOWN", 0.10),
                max_position_pct=_env_float("RISK_MAX_POSITION_PCT", 0.25),
                min_reward_risk=_env_float("RISK_MIN_REWARD_RISK", 1.5),
                min_confidence=_env_float("RISK_MIN_CONFIDENCE", 0.55),
            ),
        )
