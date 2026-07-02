"""Live execution loop: poll MT5 for the latest closed bar, ask the chosen
strategy for a signal, and manage a single net position per symbol subject
to the same risk limits enforced in backtesting.

Only runs where the MetaTrader5 package is available (Windows, MT5 terminal
running). Always start on a demo account. This module places real orders
once pointed at a live account — there is no simulation mode here.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from modiri_bot.data.mt5_client import MT5Client
from modiri_bot.risk.position_sizing import lots_for_fixed_risk
from modiri_bot.risk.risk_manager import RiskLimits, RiskManager
from modiri_bot.strategies.base import Strategy
from modiri_bot.strategies.indicators import atr as atr_indicator
from modiri_bot.utils.config import SymbolConfig
from modiri_bot.utils.timeframes import timeframe_to_timedelta

logger = logging.getLogger(__name__)


class LiveTrader:
    def __init__(
        self,
        client: MT5Client,
        symbol_cfg: SymbolConfig,
        strategy: Strategy,
        risk_limits: RiskLimits,
        stop_loss_pips: float,
        take_profit_pips: float,
        magic: int,
        deviation: int = 20,
        lookback_bars: int = 500,
        max_hold_bars: int | None = None,
        use_volatility_filter: bool = False,
        volatility_percentile_threshold: float = 95.0,
        volatility_size_mult: float = 0.5,
        use_trailing_stop: bool = False,
        trailing_start_r_multiple: float = 0.4,
        trailing_distance_r_multiple: float = 0.4,
    ):
        self.client = client
        self.symbol_cfg = symbol_cfg
        self.strategy = strategy
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.magic = magic
        self.deviation = deviation
        self.lookback_bars = lookback_bars
        self.max_hold_bars = max_hold_bars
        self.use_volatility_filter = use_volatility_filter
        self.volatility_percentile_threshold = volatility_percentile_threshold
        self.volatility_size_mult = volatility_size_mult
        self.use_trailing_stop = use_trailing_stop
        self.trailing_start_r_multiple = trailing_start_r_multiple
        self.trailing_distance_r_multiple = trailing_distance_r_multiple
        self._entry_sl_distance: dict[int, float] = {}

        account = client.account_info()
        self.risk_manager = RiskManager(risk_limits, starting_equity=account["equity"])

    def _our_positions(self) -> list[dict]:
        positions = self.client.open_positions(self.symbol_cfg.name)
        return [p for p in positions if p.get("magic") == self.magic]

    def _current_position_side(self) -> int:
        magic_positions = self._our_positions()
        if not magic_positions:
            return 0
        # POSITION_TYPE_BUY == 0, POSITION_TYPE_SELL == 1 in the MT5 API.
        return 1 if magic_positions[0]["type"] == 0 else -1

    def _bars_held(self, position: dict) -> float:
        entry_time = datetime.fromtimestamp(position["time"], tz=timezone.utc)
        elapsed = datetime.now(timezone.utc) - entry_time
        return elapsed / timeframe_to_timedelta(self.symbol_cfg.timeframe)

    def _update_trailing_stops(self, df) -> None:
        if not self.use_trailing_stop:
            return
        for p in self._our_positions():
            ticket = p["ticket"]
            sl_distance = self._entry_sl_distance.get(ticket, self.stop_loss_pips * self.symbol_cfg.pip_size)
            # df's index is naive broker-server time (from MT5Client.fetch_rates);
            # compare against a naive UTC timestamp built from the position's epoch time.
            entry_time_naive = datetime.utcfromtimestamp(p["time"])
            recent = df[df.index >= entry_time_naive]
            if recent.empty:
                continue

            is_buy = p["type"] == 0
            if is_buy:
                best_price = recent["high"].max()
                profit_r = (best_price - p["price_open"]) / sl_distance
                if profit_r >= self.trailing_start_r_multiple:
                    new_sl = best_price - self.trailing_distance_r_multiple * sl_distance
                    if new_sl > p.get("sl", 0.0):
                        result = self.client.modify_position_sl(p, new_sl)
                        logger.info("Trailing stop moved to %.5f for %s: %s", new_sl, ticket, result.message)
            else:
                best_price = recent["low"].min()
                profit_r = (p["price_open"] - best_price) / sl_distance
                if profit_r >= self.trailing_start_r_multiple:
                    new_sl = best_price + self.trailing_distance_r_multiple * sl_distance
                    if p.get("sl", 0.0) == 0.0 or new_sl < p["sl"]:
                        result = self.client.modify_position_sl(p, new_sl)
                        logger.info("Trailing stop moved to %.5f for %s: %s", new_sl, ticket, result.message)

    def _fetch_recent_bars(self):
        span = timeframe_to_timedelta(self.symbol_cfg.timeframe) * self.lookback_bars
        now = datetime.now(timezone.utc)
        return self.client.fetch_rates(
            self.symbol_cfg.name, self.symbol_cfg.timeframe, now - span, now
        )

    def poll_once(self) -> None:
        account = self.client.account_info()
        equity = account["equity"]
        today = datetime.now(timezone.utc).date()
        self.risk_manager.update_equity(equity, today)

        if self.max_hold_bars is not None:
            for p in self._our_positions():
                if self._bars_held(p) >= self.max_hold_bars:
                    result = self.client.close_position(p, deviation=self.deviation)
                    logger.info("Time stop: closed position %s after %.1f bars: %s",
                                p["ticket"], self._bars_held(p), result.message)
                    return  # let the next poll cycle decide whether to re-enter

        df = self._fetch_recent_bars()
        if len(df) < 2:
            logger.warning("Not enough bars returned, skipping this cycle")
            return

        self._update_trailing_stops(df)

        # Use the last *closed* bar, not the still-forming current one.
        signal = int(self.strategy.generate_signals(df).iloc[-2])
        current_side = self._current_position_side()

        if signal == current_side:
            return

        if current_side != 0:
            for p in self.client.open_positions(self.symbol_cfg.name):
                if p.get("magic") == self.magic:
                    result = self.client.close_position(p, deviation=self.deviation)
                    logger.info("Closed position %s: %s", p["ticket"], result.message)

        if signal == 0:
            return

        allowed, reason = self.risk_manager.can_open_new_trade(equity, open_trade_count=0)
        if not allowed:
            logger.warning("Not opening new trade: %s", reason)
            return

        risk_pct = self.risk_manager.limits.risk_per_trade_pct
        if self.use_volatility_filter:
            atr_series = atr_indicator(df["high"], df["low"], df["close"], 14)
            if len(atr_series) >= 20:
                percentile_rank = (atr_series.iloc[-1] > atr_series).mean() * 100
                if percentile_rank > self.volatility_percentile_threshold:
                    risk_pct *= self.volatility_size_mult

        lots = lots_for_fixed_risk(
            equity=equity,
            risk_per_trade_pct=risk_pct,
            stop_loss_pips=self.stop_loss_pips,
            pip_value_per_lot=self.symbol_cfg.pip_value_per_lot,
            min_lot=self.symbol_cfg.min_lot,
            lot_step=self.symbol_cfg.lot_step,
        )
        side = "buy" if signal == 1 else "sell"
        last_price = float(df["close"].iloc[-1])
        pip = self.symbol_cfg.pip_size
        if signal == 1:
            sl = last_price - self.stop_loss_pips * pip
            tp = last_price + self.take_profit_pips * pip
        else:
            sl = last_price + self.stop_loss_pips * pip
            tp = last_price - self.take_profit_pips * pip

        result = self.client.send_market_order(
            symbol=self.symbol_cfg.name,
            volume=lots,
            side=side,
            sl_price=sl,
            tp_price=tp,
            magic=self.magic,
            deviation=self.deviation,
        )
        if result.success and result.order_id is not None:
            self._entry_sl_distance[result.order_id] = self.stop_loss_pips * pip
        logger.info("Opened %s %.2f lots %s: %s", side, lots, self.symbol_cfg.name, result.message)

    def run_forever(self, poll_seconds: int = 30) -> None:
        logger.info("Starting live trading loop for %s (strategy=%s)",
                     self.symbol_cfg.name, self.strategy)
        while True:
            try:
                self.poll_once()
            except Exception:
                logger.exception("Error during poll cycle, will retry")
            time.sleep(poll_seconds)
