from datetime import date

from modiri_bot.risk.risk_manager import RiskLimits, RiskManager


def make_manager(**overrides):
    limits = RiskLimits(
        risk_per_trade_pct=1.0,
        max_concurrent_trades=1,
        max_daily_loss_pct=3.0,
        max_drawdown_pct=15.0,
    )
    for k, v in overrides.items():
        setattr(limits, k, v)
    return RiskManager(limits, starting_equity=1000.0)


def test_allows_trading_initially():
    rm = make_manager()
    allowed, _ = rm.can_open_new_trade(equity=1000.0, open_trade_count=0)
    assert allowed


def test_blocks_when_concurrent_trade_limit_hit():
    rm = make_manager()
    allowed, reason = rm.can_open_new_trade(equity=1000.0, open_trade_count=1)
    assert not allowed
    assert "concurrent" in reason


def test_blocks_after_daily_loss_limit():
    rm = make_manager()
    rm.update_equity(1000.0, date(2024, 1, 1))
    allowed, reason = rm.can_open_new_trade(equity=965.0, open_trade_count=0)  # -3.5%
    assert not allowed
    assert "daily" in reason


def test_daily_loss_resets_on_new_day():
    rm = make_manager()
    rm.update_equity(1000.0, date(2024, 1, 1))
    rm.update_equity(965.0, date(2024, 1, 1))
    allowed, _ = rm.can_open_new_trade(equity=965.0, open_trade_count=0)
    assert not allowed

    rm.update_equity(965.0, date(2024, 1, 2))
    allowed, _ = rm.can_open_new_trade(equity=965.0, open_trade_count=0)
    assert allowed


def test_drawdown_kill_switch_is_permanent_until_manual_reset():
    rm = make_manager()
    rm.update_equity(1000.0, date(2024, 1, 1))
    rm.update_equity(840.0, date(2024, 1, 1))  # -16% drawdown from peak
    allowed, reason = rm.can_open_new_trade(equity=840.0, open_trade_count=0)
    assert not allowed
    assert "kill-switch" in reason

    # Recovering equity alone doesn't clear it -- only reset_kill_switch() does.
    rm.update_equity(1000.0, date(2024, 1, 2))
    allowed, _ = rm.can_open_new_trade(equity=1000.0, open_trade_count=0)
    assert not allowed

    rm.reset_kill_switch()
    allowed, _ = rm.can_open_new_trade(equity=1000.0, open_trade_count=0)
    assert allowed
