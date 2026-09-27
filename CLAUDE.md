# Autonomous Trading Agent Framework

## Objective
Build a modular, production-ready trading agent framework that fetches real-time market data, calculates quantitative indicators, evaluates market structure/sentiment, and executes trades within strict risk parameters.

## Architecture Guidelines
- Modular OOP architecture in Python (Data Ingestion, Signal Generation, Risk Engine, Execution Layer).
- Deterministic risk guardrails (hard stop-loss, position sizing, max drawdown protection).
- Comprehensive test coverage using `pytest` for all modules.