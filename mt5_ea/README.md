# Modiri Bot -- MT5 Expert Advisor

`ModiriBot_EA.mq5` is a self-contained, native MQL5 port of the validated
champion strategy from this repo's Python research
(`config/current_best_strategy.json`): a weighted ensemble of 2x RSI-reversion
+ MFI-reversion + CCI-reversion, plus the 3 risk overlays validated on top of
it (15-bar time stop, ATR volatility-percentile position sizing, 0.4R/0.4R
trailing stop). It needs nothing else installed -- no Python, no external
files -- just MetaEditor.

Holdout backtest (real EURUSD H4 data, ~9 months never used to pick the
strategy): **+8.20% return, Sharpe 2.71, max drawdown 2.69%, 76.3% win rate,
59 trades.** Past performance, not a live guarantee -- see the main
[README](../README.md) for the full research history and caveats.

## Install and compile

1. Open **MetaEditor** (from MT5: `Tools -> MetaQuotes Language Editor`, or press F4).
2. `File -> Open Data Folder` in MT5 first if you need to find the right folder,
   then in MetaEditor go to `File -> Open` and navigate to
   `MQL5/Experts/`, or simply copy `ModiriBot_EA.mq5` into that folder yourself
   (typically `C:\Users\<you>\AppData\Roaming\MetaQuotes\Terminal\<hash>\MQL5\Experts\`).
3. Open `ModiriBot_EA.mq5` in MetaEditor and press **F7** (Compile). It should
   compile with 0 errors (a couple of harmless "unused variable" style warnings
   are fine, if any).
4. Back in MT5, open the **Navigator** panel (`Ctrl+N`), find `ModiriBot_EA`
   under `Expert Advisors`, and drag it onto an **EURUSD, H4** chart.
   This ensemble was validated specifically on that symbol/timeframe --
   the EA will still run elsewhere but prints a warning, since nothing
   backs those numbers there.
5. In the "Common" tab of the EA properties dialog, make sure
   **"Allow Algo Trading"** is checked. In MT5's toolbar, the "Algo Trading"
   button must also be enabled (green).
6. Start on a **demo account**. Watch it for a meaningful stretch of real
   time and compare live results to the backtest before ever considering
   a live account, and only risk capital you can afford to lose.

## What it does automatically

- Recomputes the ensemble's signal from scratch on every newly closed H4 bar
  (RSI x2 + MFI + CCI mean-reversion sub-signals, weighted vote against a
  0.2685 threshold) and opens/closes/flips positions accordingly.
- Sizes every trade to risk exactly `InpRiskPerTradePct` (default 1%) of
  current equity on the stop-loss distance, halved automatically when ATR
  is in the extreme upper tail of its recent range (volatility filter).
- Force-closes any position still open after `InpMaxHoldBars` (default 15)
  H4 bars, regardless of the signal (time stop).
- Trails the stop once a trade is `InpTrailingStartR` (default 0.4R) in
  profit, `InpTrailingDistR` (default 0.4R) behind the best price seen.
- Enforces a daily loss limit (`InpMaxDailyLossPct`, default 3%) and a
  permanent drawdown kill-switch (`InpMaxDrawdownPct`, default 15% --
  stops opening new trades until you remove and re-attach the EA).
- Draws a live on-chart dashboard: balance, equity, floating P/L, total
  closed trades, win rate, profit factor, net P/L, current position, and
  kill-switch status, refreshed once per second.

All inputs are editable in the EA's properties dialog without recompiling,
grouped as: ensemble weights, risk management, volatility filter, trailing
stop, execution, and dashboard.

## Note on the dashboard "logo"

The on-chart panel uses a small colored "MB" badge instead of an embedded
image, so the `.mq5` file stays a single self-contained text file with no
external `.bmp`/resource dependencies to keep track of -- it just compiles
and runs anywhere you copy it.
