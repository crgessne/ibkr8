"""
IBKR Trading Platform — Paper & Live

Modules:
    config       – Trading configuration (capital, risk, costs, connection)
    risk         – Position sizing, risk limits, margin cost calculation
    orders       – Order creation, submission, tracking, fill management
    positions    – Position state, P&L tracking, exit management
    strategy     – VWAP mean-reversion strategy (entry/exit signals)
    engine       – Main trading engine (bar loop, state machine)
    runner       – CLI entry point for paper / live execution
"""
