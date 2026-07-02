from datetime import datetime, timedelta, timezone

import pandas as pd

from modiri_bot.live.live_trader import LiveTrader
from modiri_bot.risk.risk_manager import RiskLimits
from modiri_bot.strategies.ma_crossover import MACrossoverStrategy
from modiri_bot.utils.config import SymbolConfig


class FakeClient:
    def __init__(self):
        self.modify_calls = []

    def account_info(self):
        return {"equity": 1000.0}

    def modify_position_sl(self, position, new_sl):
        self.modify_calls.append((position["ticket"], new_sl))

        class _Result:
            message = "ok"

        return _Result()


def make_trader(max_hold_bars=None, use_trailing_stop=False,
                 trailing_start_r_multiple=0.4, trailing_distance_r_multiple=0.4, client=None):
    symbol_cfg = SymbolConfig(
        name="EURUSD", timeframe="H4", pip_size=0.0001, pip_value_per_lot=10.0,
        contract_size=100000, min_lot=0.01, lot_step=0.01, spread_pips=1.2, commission_per_lot=7.0,
    )
    risk_limits = RiskLimits(risk_per_trade_pct=1.0, max_concurrent_trades=1,
                              max_daily_loss_pct=3.0, max_drawdown_pct=15.0)
    return LiveTrader(
        client=client or FakeClient(), symbol_cfg=symbol_cfg, strategy=MACrossoverStrategy(),
        risk_limits=risk_limits, stop_loss_pips=40.0, take_profit_pips=80.0,
        magic=1, max_hold_bars=max_hold_bars, use_trailing_stop=use_trailing_stop,
        trailing_start_r_multiple=trailing_start_r_multiple,
        trailing_distance_r_multiple=trailing_distance_r_multiple,
    )


def test_bars_held_computes_elapsed_time_in_bar_units():
    trader = make_trader(max_hold_bars=15)
    entry_time = datetime.now(timezone.utc) - timedelta(hours=4 * 3)  # 3 H4 bars ago
    position = {"time": int(entry_time.timestamp())}
    held = trader._bars_held(position)
    assert 2.9 < held < 3.1


def test_bars_held_zero_for_a_just_opened_position():
    trader = make_trader(max_hold_bars=15)
    position = {"time": int(datetime.now(timezone.utc).timestamp())}
    assert trader._bars_held(position) < 0.1


class FakeClientWithPositions(FakeClient):
    def __init__(self, positions):
        super().__init__()
        self._positions = positions

    def open_positions(self, symbol=None):
        return self._positions


def test_trailing_stop_moves_sl_once_profit_threshold_is_reached():
    client = FakeClientWithPositions([])
    trader = make_trader(use_trailing_stop=True, trailing_start_r_multiple=0.4,
                          trailing_distance_r_multiple=0.4, client=client)
    trader._entry_sl_distance[1] = 40 * 0.0001  # matches stop_loss_pips=40

    entry_time = datetime.utcnow() - timedelta(hours=8)
    position = {
        "ticket": 1, "type": 0, "price_open": 1.1000, "sl": 1.0960, "magic": 1,
        "time": int(entry_time.replace(tzinfo=timezone.utc).timestamp()),
    }
    client._positions = [position]

    index = pd.date_range(entry_time, periods=5, freq="4h")
    df = pd.DataFrame({
        "open": [1.1000] * 5, "close": [1.1000] * 5,
        "high": [1.1005, 1.1010, 1.1030, 1.1020, 1.1015],  # clears 0.4*40=16 pip profit (1.1016)
        "low": [1.0995, 1.1000, 1.1010, 1.1005, 1.1005],
    }, index=index)

    trader._update_trailing_stops(df)

    assert len(client.modify_calls) == 1
    ticket, new_sl = client.modify_calls[0]
    assert ticket == 1
    assert new_sl > position["sl"]  # tightened, not loosened
    assert new_sl < 1.1030  # trails behind the best price, not sitting on top of it


def test_trailing_stop_does_nothing_before_the_profit_threshold():
    client = FakeClientWithPositions([])
    trader = make_trader(use_trailing_stop=True, trailing_start_r_multiple=1.0,
                          trailing_distance_r_multiple=0.4, client=client)
    trader._entry_sl_distance[1] = 40 * 0.0001

    entry_time = datetime.utcnow() - timedelta(hours=4)
    position = {
        "ticket": 1, "type": 0, "price_open": 1.1000, "sl": 1.0960, "magic": 1,
        "time": int(entry_time.replace(tzinfo=timezone.utc).timestamp()),
    }
    client._positions = [position]

    index = pd.date_range(entry_time, periods=2, freq="4h")
    df = pd.DataFrame({
        "open": [1.1000, 1.1000], "close": [1.1000, 1.1000],
        "high": [1.1005, 1.1008],  # only ~8 pips in favor, below the 1.0R (40 pip) threshold
        "low": [1.0995, 1.0998],
    }, index=index)

    trader._update_trailing_stops(df)
    assert len(client.modify_calls) == 0
