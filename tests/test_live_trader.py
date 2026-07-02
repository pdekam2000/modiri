from datetime import datetime, timedelta, timezone

from modiri_bot.live.live_trader import LiveTrader
from modiri_bot.risk.risk_manager import RiskLimits
from modiri_bot.strategies.ma_crossover import MACrossoverStrategy
from modiri_bot.utils.config import SymbolConfig


class FakeClient:
    def account_info(self):
        return {"equity": 1000.0}


def make_trader(max_hold_bars=None):
    symbol_cfg = SymbolConfig(
        name="EURUSD", timeframe="H4", pip_size=0.0001, pip_value_per_lot=10.0,
        contract_size=100000, min_lot=0.01, lot_step=0.01, spread_pips=1.2, commission_per_lot=7.0,
    )
    risk_limits = RiskLimits(risk_per_trade_pct=1.0, max_concurrent_trades=1,
                              max_daily_loss_pct=3.0, max_drawdown_pct=15.0)
    return LiveTrader(
        client=FakeClient(), symbol_cfg=symbol_cfg, strategy=MACrossoverStrategy(),
        risk_limits=risk_limits, stop_loss_pips=40.0, take_profit_pips=80.0,
        magic=1, max_hold_bars=max_hold_bars,
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
