import pytest

from modiri_bot.risk.position_sizing import lots_for_fixed_risk


def test_basic_sizing_matches_hand_calculation():
    # Risking 1% of 1000 = 10, over a 20 pip stop at $10/pip/lot => 0.05 lots.
    lots = lots_for_fixed_risk(
        equity=1000.0, risk_per_trade_pct=1.0, stop_loss_pips=20.0,
        pip_value_per_lot=10.0, min_lot=0.01, lot_step=0.01,
    )
    assert lots == pytest.approx(0.05)


def test_rounds_down_to_lot_step():
    lots = lots_for_fixed_risk(
        equity=1000.0, risk_per_trade_pct=1.0, stop_loss_pips=33.0,
        pip_value_per_lot=10.0, min_lot=0.01, lot_step=0.01,
    )
    # raw = 10 / (33*10) = 0.0303... -> floors to 0.03
    assert lots == pytest.approx(0.03)


def test_never_goes_below_min_lot():
    lots = lots_for_fixed_risk(
        equity=10.0, risk_per_trade_pct=0.1, stop_loss_pips=100.0,
        pip_value_per_lot=10.0, min_lot=0.01, lot_step=0.01,
    )
    assert lots == pytest.approx(0.01)


def test_respects_max_lot_cap():
    lots = lots_for_fixed_risk(
        equity=1_000_000.0, risk_per_trade_pct=5.0, stop_loss_pips=5.0,
        pip_value_per_lot=10.0, min_lot=0.01, lot_step=0.01, max_lot=10.0,
    )
    assert lots == 10.0
