"""Broker/exchange bindings: builds an authenticated CCXT client from config."""
from __future__ import annotations

import ccxt

from config.settings import ExchangeConfig


def create_exchange(cfg: ExchangeConfig) -> ccxt.Exchange:
    """Instantiate a CCXT exchange client, in sandbox mode unless explicitly disabled."""
    if not hasattr(ccxt, cfg.exchange_id):
        raise ValueError(f"Unknown CCXT exchange id: {cfg.exchange_id!r}")

    exchange_cls = getattr(ccxt, cfg.exchange_id)
    params: dict = {
        "enableRateLimit": True,
        "timeout": cfg.timeout_ms,
        "options": {"defaultType": cfg.default_type},
    }
    if cfg.api_key and cfg.api_secret:
        params["apiKey"] = cfg.api_key
        params["secret"] = cfg.api_secret

    exchange = exchange_cls(params)
    if cfg.sandbox:
        exchange.set_sandbox_mode(True)
    return exchange
