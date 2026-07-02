"""Thin wrapper around the MetaTrader5 terminal API.

The `MetaTrader5` package only works on Windows, next to a running MT5
terminal, so the import is deferred until a client is actually constructed.
Backtesting and optimization never touch this module — they read historical
bars from CSV via `csv_loader.py` instead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[int]
    message: str
    raw: object = None


class MT5Client:
    """Connects to a running MT5 terminal and exposes the calls the bot needs."""

    def __init__(self, login: int, password: str, server: str, path: str | None = None):
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:  # pragma: no cover - exercised only on Windows
            raise RuntimeError(
                "The MetaTrader5 package is only available on Windows with a "
                "MT5 terminal installed. Live trading / direct history pulls "
                "must run there; backtesting works anywhere."
            ) from exc

        self._mt5 = mt5
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._connected = False

    def connect(self) -> None:
        mt5 = self._mt5
        ok = mt5.initialize(path=self._path) if self._path else mt5.initialize()
        if not ok:
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
        if not mt5.login(self._login, password=self._password, server=self._server):
            error = mt5.last_error()
            mt5.shutdown()
            raise RuntimeError(f"MT5 login failed: {error}")
        self._connected = True
        logger.info("Connected to MT5 as %s on %s", self._login, self._server)

    def shutdown(self) -> None:
        if self._connected:
            self._mt5.shutdown()
            self._connected = False

    def __enter__(self) -> "MT5Client":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    def account_info(self) -> dict:
        info = self._mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info() failed: {self._mt5.last_error()}")
        return info._asdict()

    def fetch_rates(
        self, symbol: str, timeframe: str, date_from: datetime, date_to: datetime
    ) -> pd.DataFrame:
        """Pull historical OHLCV bars directly from the broker's history, i.e.
        real market data as recorded by the terminal you're connected to."""
        mt5 = self._mt5
        tf = getattr(mt5, TIMEFRAME_MAP[timeframe])
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"No rates returned for {symbol} {timeframe}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(
            columns={"tick_volume": "volume", "open": "open", "high": "high",
                     "low": "low", "close": "close"}
        )
        return df.set_index("time")[["open", "high", "low", "close", "volume"]]

    def open_positions(self, symbol: str | None = None) -> list[dict]:
        positions = self._mt5.positions_get(symbol=symbol) if symbol else self._mt5.positions_get()
        return [p._asdict() for p in (positions or [])]

    def send_market_order(
        self,
        symbol: str,
        volume: float,
        side: str,  # "buy" or "sell"
        sl_price: float | None,
        tp_price: float | None,
        magic: int,
        deviation: int = 20,
        comment: str = "modiri_bot",
    ) -> OrderResult:
        mt5 = self._mt5
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(False, None, f"No tick for {symbol}")

        price = tick.ask if side == "buy" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl_price is not None:
            request["sl"] = sl_price
        if tp_price is not None:
            request["tp"] = tp_price

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, None, f"order_send failed: {result}", raw=result)
        return OrderResult(True, result.order, "ok", raw=result)

    def close_position(self, position: dict, deviation: int = 20) -> OrderResult:
        mt5 = self._mt5
        symbol = position["symbol"]
        volume = position["volume"]
        is_buy = position["type"] == mt5.POSITION_TYPE_BUY
        tick = mt5.symbol_info_tick(symbol)
        price = tick.bid if is_buy else tick.ask
        order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": position["ticket"],
            "price": price,
            "deviation": deviation,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(False, None, f"close failed: {result}", raw=result)
        return OrderResult(True, result.order, "ok", raw=result)
