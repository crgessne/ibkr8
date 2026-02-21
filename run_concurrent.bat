@echo off
cd /d c:\Users\Administrator\ibkr8
.\.venv\Scripts\python.exe sim_trading\simulate_streaming_clean.py --year 2024 --stop-atr 1.5 --rf-threshold 0.5 --concurrent
pause
