# Autonomous Trading Agent Framework

![Tests](https://github.com/jalalkhanutmanzai/TradingAgent/actions/workflows/tests.yml/badge.svg)

A modular, production-ready trading agent framework that fetches real-time market data,
calculates quantitative indicators, evaluates market structure/sentiment, and executes
trades within strict risk parameters.

## Architecture

- **Data Ingestion** (`data/`) — OHLCV market data via CCXT.
- **Indicators** (`indicators/`) — EMA, RSI, MACD, ATR.
- **Signal Generation** (`strategy/`) — deterministic rule-based evaluator and an
  LLM-backed evaluator (Claude), both producing a validated `TradeSignal`.
- **Risk Engine** (`risk/`) — fixed-fractional position sizing, reward/risk and
  confidence gating, max open positions, and a latching drawdown halt.
- **Execution** (`execution/`) — simulated paper trading and CCXT-compatible live
  order placement, with stop-loss/take-profit monitoring.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Test

```bash
pytest
```
