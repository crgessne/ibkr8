"""IBKR client connection management."""

from __future__ import annotations

import time
from typing import Optional

from ib_insync import IB, Contract, Stock

from .utils import load_config


class IBClient:
    """
    Wrapper for IB connection with automatic reconnection.
    
    Usage:
        client = IBClient()
        client.connect()
        
        # Use client.ib for IB API calls
        contract = client.make_stock_contract("TSLA")
        
        client.disconnect()
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: int = 60,
        readonly: bool = True,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.readonly = readonly
        
        self._ib: Optional[IB] = None
        self._connected = False
    
    @classmethod
    def from_config(cls, config_path: str = "config/settings.json") -> "IBClient":
        """Create client from config file."""
        config = load_config(config_path)
        return cls(
            host=config.get("host", "127.0.0.1"),
            port=config.get("port", 7497),
            client_id=config.get("client_id", 1),
            timeout=config.get("timeout", 60),
        )
    
    @property
    def ib(self) -> IB:
        """Get IB instance, creating if needed."""
        if self._ib is None:
            self._ib = IB()
        return self._ib
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self._connected and self._ib is not None and self._ib.isConnected()
    
    def connect(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        """
        Connect to IB TWS/Gateway.
        
        Args:
            max_retries: Number of connection attempts
            retry_delay: Seconds between retries
            
        Returns:
            True if connected successfully
        """
        for attempt in range(max_retries):
            try:
                print(f"Connecting to IB at {self.host}:{self.port} (attempt {attempt + 1}/{max_retries})...")
                
                self.ib.connect(
                    host=self.host,
                    port=self.port,
                    clientId=self.client_id,
                    timeout=self.timeout,
                    readonly=self.readonly,
                )
                
                self._connected = True
                print(f"✓ Connected to IB (client_id={self.client_id})")
                return True
                
            except Exception as e:
                print(f"✗ Connection failed: {e}")
                if attempt < max_retries - 1:
                    print(f"  Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
        
        return False
    
    def disconnect(self) -> None:
        """Disconnect from IB."""
        if self._ib is not None:
            try:
                self._ib.disconnect()
                print("✓ Disconnected from IB")
            except Exception as e:
                print(f"✗ Disconnect error: {e}")
            finally:
                self._connected = False
    
    def make_stock_contract(
        self,
        symbol: str,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Stock:
        """Create a stock contract."""
        return Stock(symbol, exchange, currency)
    
    def make_contract(
        self,
        symbol: str,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Contract:
        """Create a generic contract."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
    def qualify_contract(self, contract: Contract) -> Contract:
        """Qualify contract with IB to get full details."""
        if not self.is_connected:
            raise RuntimeError("Not connected to IB")
        
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError(f"Could not qualify contract: {contract}")
        
        return qualified[0]
    
    def __enter__(self) -> "IBClient":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()
