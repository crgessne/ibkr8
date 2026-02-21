@echo off
REM Run concurrent simulation for 2024
cd /d c:\Users\Administrator\ibkr8
.\.venv\Scripts\python.exe sim_trading\simulate_concurrent.py --year 2024 --stop-atr 1.5 --rf-threshold 0.5 --initial-capital 1000000
pause
