"""Minimal IBKR connection test."""
import sys, traceback
sys.path.insert(0, r"C:\Users\Administrator\ibkr8")

OUTF = r"C:\Users\Administrator\ibkr8\_conn_test.txt"

with open(OUTF, "w") as f:
    try:
        f.write("1. importing ib_insync\n")
        f.flush()
        from ib_insync import IB, Stock
        f.write("2. creating IB\n")
        f.flush()
        ib = IB()
        f.write("3. connecting to 127.0.0.1:7497\n")
        f.flush()
        ib.connect("127.0.0.1", 7497, clientId=99, readonly=True)
        f.write(f"4. connected! accounts={ib.managedAccounts()}\n")
        f.flush()
        
        contract = Stock("TSLA", "SMART", "USD")
        ib.qualifyContracts(contract)
        f.write(f"5. contract qualified: {contract}\n")
        f.flush()
        
        ib.disconnect()
        f.write("6. disconnected OK\n")
        f.flush()
    except Exception as e:
        f.write(f"ERROR: {e}\n")
        f.write(traceback.format_exc())
        f.flush()
