import ccxt
import pytest

from config.broker import create_exchange
from config.settings import ExchangeConfig, RiskConfig, Settings


def test_defaults_are_conservative():
    s = Settings()
    assert s.exchange.sandbox is True
    assert s.risk.risk_per_trade == 0.01
    assert s.llm.model == "claude-opus-5"


@pytest.mark.parametrize("risk", [0.0, -0.01, 0.021, 0.5])
def test_risk_per_trade_hard_cap(risk):
    with pytest.raises(ValueError):
        RiskConfig(risk_per_trade=risk)


def test_max_drawdown_hard_cap():
    with pytest.raises(ValueError):
        RiskConfig(max_drawdown=0.5)


def test_from_env(monkeypatch):
    monkeypatch.setenv("TRADING_SYMBOL", "ETH/USDT")
    monkeypatch.setenv("RISK_PER_TRADE", "0.015")
    monkeypatch.setenv("EXCHANGE_SANDBOX", "false")
    monkeypatch.setenv("LLM_ENABLED", "0")
    s = Settings.from_env()
    assert s.symbol == "ETH/USDT"
    assert s.risk.risk_per_trade == 0.015
    assert s.exchange.sandbox is False
    assert s.llm.enabled is False


def test_from_env_rejects_excessive_risk(monkeypatch):
    monkeypatch.setenv("RISK_PER_TRADE", "0.05")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_create_exchange_unknown_id():
    with pytest.raises(ValueError):
        create_exchange(ExchangeConfig(exchange_id="not_a_real_exchange"))


def test_create_exchange_no_network():
    ex = create_exchange(ExchangeConfig(exchange_id="kraken", sandbox=False,
                                        api_key="k", api_secret="s"))
    assert isinstance(ex, ccxt.kraken)
    assert ex.apiKey == "k" and ex.secret == "s"
    assert ex.enableRateLimit is True
