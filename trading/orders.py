"""
Order Management — create, submit, and track IBKR orders.

Thin wrapper around ib_insync order objects.  Handles:
  • Market / limit order creation
  • Bracket orders (entry + stop + target)
  • OCA (one-cancels-all) grouping for stop/target
  • Order status callbacks
  • Fill tracking
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from ib_insync import (
    IB,
    Contract,
    LimitOrder,
    MarketOrder,
    Order,
    StopLimitOrder,
    StopOrder,
    Trade,
)

# How far past the stop trigger the limit price is set (caps slippage).
# E.g. for a LONG position with stop trigger $410.00, the sell-limit would be
# $410.00 - $0.25 = $409.75.  Worst-case fill is $409.75 instead of market.
STOP_LIMIT_BUFFER = 0.25

_log = logging.getLogger("trading.orders")
_bar_log = logging.getLogger("trading.barlog")


class Side(Enum):
    BUY  = auto()
    SELL = auto()


@dataclass
class OrderTicket:
    """Internal record of a submitted order group (entry + exit bracket)."""

    ticket_id: str
    symbol: str
    side: Side
    shares: int
    entry_order_id: Optional[int] = None
    stop_order_id: Optional[int] = None
    target_order_id: Optional[int] = None
    entry_trade: Optional[Trade] = None
    stop_trade: Optional[Trade] = None
    target_trade: Optional[Trade] = None
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    status: str = "pending"      # pending / filled / stopped / target / cancelled / error
    oca_group: str = ""

    @property
    def is_open(self) -> bool:
        return self.status in ("pending", "filled")


class OrderManager:
    """Submit and track orders through an ib_insync.IB connection.

    Parameters
    ----------
    ib : IB
        Connected ``ib_insync.IB`` instance.
    on_fill : callable, optional
        ``fn(ticket: OrderTicket, trade: Trade)`` called when the entry fills.
    on_exit : callable, optional
        ``fn(ticket: OrderTicket, reason: str, fill_price: float)`` called when
        stop or target fills, or order is cancelled.
    """

    def __init__(
        self,
        ib: IB,
        on_fill: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
    ):
        self.ib = ib
        self._on_fill = on_fill
        self._on_exit = on_exit
        self._tickets: Dict[str, OrderTicket] = {}

        # Wire up ib_insync event handlers
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details

    # ── public API ──────────────────────────────────────────────────────

    def submit_bracket(
        self,
        contract: Contract,
        side: Side,
        shares: int,
        stop_price: float,
        target_price: float,
        entry_type: str = "MKT",
        entry_limit_price: Optional[float] = None,
    ) -> OrderTicket:
        """Submit an entry order with an attached stop + target bracket.

        The stop and target are linked via an OCA group so that when one
        fills the other is automatically cancelled by IB.

        Parameters
        ----------
        contract : Contract
            Qualified IB contract.
        side : Side
            BUY for long entry, SELL for short entry.
        shares : int
            Number of shares.
        stop_price : float
            Stop-loss trigger price.
        target_price : float
            Profit-target (limit) price.
        entry_type : str
            ``"MKT"`` (default) or ``"LMT"``.
        entry_limit_price : float, optional
            Required when ``entry_type == "LMT"``.

        Returns
        -------
        OrderTicket
        """
        ticket_id = uuid.uuid4().hex[:12]
        oca = f"oca_{ticket_id}"

        action = "BUY" if side == Side.BUY else "SELL"
        exit_action = "SELL" if side == Side.BUY else "BUY"

        # ── Entry order ─────────────────────────────────────────────
        if entry_type == "LMT" and entry_limit_price is not None:
            entry_order = LimitOrder(action, shares, entry_limit_price)
        else:
            entry_order = MarketOrder(action, shares)
        entry_order.transmit = False  # hold until bracket children attached

        entry_trade = self.ib.placeOrder(contract, entry_order)

        # ── Stop-limit order (caps slippage vs plain stop-market) ───
        # Limit price sits STOP_LIMIT_BUFFER worse than the trigger so
        # the order still fills in a fast move, but won't chase to infinity.
        if exit_action == "SELL":
            # Long position stop: trigger=stop_price, limit slightly below
            stop_limit_price = round(stop_price - STOP_LIMIT_BUFFER, 2)
        else:
            # Short position stop: trigger=stop_price, limit slightly above
            stop_limit_price = round(stop_price + STOP_LIMIT_BUFFER, 2)

        stop_order = StopLimitOrder(
            exit_action, shares, stop_limit_price, stop_price,
        )
        stop_order.parentId = entry_order.orderId
        stop_order.ocaGroup = oca
        stop_order.ocaType = 1      # cancel remaining on fill
        stop_order.transmit = False

        stop_trade = self.ib.placeOrder(contract, stop_order)

        # ── Target (limit) order ────────────────────────────────────
        target_order = LimitOrder(exit_action, shares, target_price)
        target_order.parentId = entry_order.orderId
        target_order.ocaGroup = oca
        target_order.ocaType = 1
        target_order.transmit = True  # transmit the whole bracket

        target_trade = self.ib.placeOrder(contract, target_order)

        ticket = OrderTicket(
            ticket_id=ticket_id,
            symbol=contract.symbol,
            side=side,
            shares=shares,
            entry_order_id=entry_order.orderId,
            stop_order_id=stop_order.orderId,
            target_order_id=target_order.orderId,
            entry_trade=entry_trade,
            stop_trade=stop_trade,
            target_trade=target_trade,
            oca_group=oca,
        )
        self._tickets[ticket_id] = ticket
        return ticket

    def submit_market(
        self,
        contract: Contract,
        side: Side,
        shares: int,
    ) -> OrderTicket:
        """Submit a simple market order (no bracket).

        Useful for EOD flatten or emergency close.
        """
        ticket_id = uuid.uuid4().hex[:12]
        action = "BUY" if side == Side.BUY else "SELL"
        order = MarketOrder(action, shares)
        order.transmit = True
        trade = self.ib.placeOrder(contract, order)

        ticket = OrderTicket(
            ticket_id=ticket_id,
            symbol=contract.symbol,
            side=side,
            shares=shares,
            entry_order_id=order.orderId,
            entry_trade=trade,
        )
        self._tickets[ticket_id] = ticket
        return ticket

    def cancel_ticket(self, ticket_id: str) -> None:
        """Cancel all open orders for a ticket."""
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return
        for trade in (ticket.entry_trade, ticket.stop_trade, ticket.target_trade):
            if trade is not None and trade.isActive():
                self.ib.cancelOrder(trade.order)
        ticket.status = "cancelled"

    def cancel_exit_orders(self, ticket_id: str) -> None:
        """Cancel only the stop and target orders (keep position open for manual exit)."""
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return
        for trade in (ticket.stop_trade, ticket.target_trade):
            if trade is not None and trade.isActive():
                self.ib.cancelOrder(trade.order)

    def get_ticket(self, ticket_id: str) -> Optional[OrderTicket]:
        return self._tickets.get(ticket_id)

    @property
    def open_tickets(self) -> List[OrderTicket]:
        return [t for t in self._tickets.values() if t.is_open]

    @property
    def all_tickets(self) -> List[OrderTicket]:
        return list(self._tickets.values())

    # ── event handlers ──────────────────────────────────────────────────

    def _on_order_status(self, trade: Trade) -> None:
        """Called by ib_insync when any order status changes."""
        oid = trade.order.orderId
        status = trade.orderStatus.status
        filled = trade.orderStatus.filled
        remaining = trade.orderStatus.remaining

        _bar_log.info("ORDER STATUS | orderId=%d status=%s filled=%.0f remaining=%.0f avg=%.2f",
                      oid, status, filled, remaining, trade.orderStatus.avgFillPrice)

        for ticket in self._tickets.values():
            # --- Entry order ---
            if oid == ticket.entry_order_id:
                if status == "Filled" or (status in ("Submitted", "PreSubmitted") and filled > 0):
                    # IB sometimes sends Cancelled then Submitted+fills.
                    # Always override to "filled" when shares actually execute.
                    ticket.fill_price = trade.orderStatus.avgFillPrice
                    ticket.fill_time = datetime.now()
                    ticket.status = "filled"
                    _bar_log.info("TICKET %s | entry -> filled | avg=$%.2f filled=%.0f",
                                  ticket.ticket_id, ticket.fill_price, filled)
                    if self._on_fill and status == "Filled":
                        self._on_fill(ticket, trade)
                elif status == "Cancelled" and filled == 0 and ticket.status == "pending":
                    # Only treat as cancelled if nothing filled AND still pending
                    ticket.status = "cancelled"
                    _bar_log.info("TICKET %s | entry -> cancelled (0 filled)", ticket.ticket_id)

            # --- Stop fill ---
            elif oid == ticket.stop_order_id and status == "Filled":
                ticket.status = "stopped"
                fill_px = trade.orderStatus.avgFillPrice
                _bar_log.info("TICKET %s | STOPPED @ $%.2f", ticket.ticket_id, fill_px)
                if self._on_exit:
                    self._on_exit(ticket, "stop", fill_px)

            # --- Target fill ---
            elif oid == ticket.target_order_id and status == "Filled":
                ticket.status = "target"
                fill_px = trade.orderStatus.avgFillPrice
                _bar_log.info("TICKET %s | TARGET HIT @ $%.2f", ticket.ticket_id, fill_px)
                if self._on_exit:
                    self._on_exit(ticket, "vwap", fill_px)

    def _on_exec_details(self, trade: Trade, fill) -> None:
        """Called on execution details — currently a no-op, status handler does the work."""
        pass
