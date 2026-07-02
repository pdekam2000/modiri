# modiri — ربات بک‌تست و معاملات الگوریتمی فارکس برای MetaTrader 5

## هشدار مهم درباره هدف سود (لطفاً قبل از هر چیز بخوانید)

هدف «۱۰۰۰ یورو به ۲۵۰۰ یورو در یک ماه» یعنی ۱۵۰٪ بازده ماهانه. این عدد **غیرواقعی و به‌شدت پرریسک** است:

- حتی صندوق‌های حرفه‌ای و معامله‌گران الگوریتمی موفق معمولاً به چند درصد تا حداکثر ۱۰-۱۵٪ سود ماهانه پایدار می‌رسند؛ آن هم با سال‌ها تجربه و مدیریت ریسک دقیق.
- طبق آمار بروکرها (که طبق قانون باید افشا شود)، حدود ۷۰ تا ۸۵٪ حساب‌های معامله‌گران خرد فارکس ضرر می‌دهند.
- هر استراتژی یا ربات که ادعا کند می‌تواند به‌طور پایدار ۱۵۰٪ در ماه سود بدهد، یا دروغ می‌گوید یا ریسک نابودی کامل حساب (Risk of Ruin) را در آن پنهان کرده است. برای رسیدن به چنین بازدهی باید حجم معاملات (لوریج) آن‌قدر بالا برود که یک حرکت نامطلوب بازار کل سرمایه را از بین می‌برد.

**این پروژه چه کاری واقعاً انجام می‌دهد:** یک فریم‌ورک حرفه‌ای برای بک‌تست با داده‌های واقعی، ترکیب و بهینه‌سازی استراتژی‌ها (walk-forward optimization) و اجرای معاملات روی MT5 با مدیریت ریسک است. خروجی آن یک گزارش صادقانه از عملکرد ریسک-بازده است — نه یک استراتژی «تضمینی» برای رسیدن به هدف ۱۵۰٪. همیشه با حساب دمو شروع کنید و هرگز بیشتر از چیزی که توان از دست دادنش را دارید ریسک نکنید.

---

## What this actually is

A Python framework that:

1. Pulls **real historical bars** — either directly from a running MT5 terminal, or from CSV exported from MT5's History Center.
2. Runs a realistic bar-by-bar **backtest engine**: spread + commission costs, stop-loss/take-profit checked against each bar's high/low, fixed-fractional position sizing, a daily loss limit, and a drawdown kill-switch.
3. Ships 5 baseline strategies (trend + mean-reversion) and an **ensemble combiner** that weights them together.
4. Includes a **walk-forward optimizer**: it grid-searches each strategy's parameters and the ensemble's weights, scores everything out-of-sample across multiple time folds, and then reports performance on a *final holdout slice the search never touched* — so the number you see isn't just curve-fit to the same data it was tuned on.
5. Provides a **live trading loop** for MT5 that reuses the exact same risk manager and position sizing as the backtest.

It does **not** guess a strategy that hits an arbitrary target return. It optimizes a risk-adjusted objective (Sharpe/Sortino/Calmar) and tells you honestly when nothing found in the data holds up out-of-sample — which, on genuinely efficient markets or short/noisy datasets, is common. See `scripts/optimize_strategies.py --demo` for a worked example of that happening on purpose.

## Project layout

```
modiri_bot/
  data/          MT5 connector (Windows-only), CSV loader, synthetic data (tests only)
  strategies/    Strategy base class, 5 strategies, ensemble combiner, registry
  backtest/      Bar-by-bar engine, performance metrics, walk-forward optimizer
  risk/          Position sizing, live risk manager (daily loss limit + drawdown kill-switch)
  live/          MT5 live trading loop
  utils/         config.yaml / .env loading, timeframe helpers
scripts/
  fetch_mt5_history.py    export real history from MT5 to CSV (Windows only)
  run_backtest.py         backtest one strategy against a CSV
  optimize_strategies.py  walk-forward search across strategies + ensemble
  run_live.py             run the winning config live against MT5 (Windows only)
config/
  config.yaml    symbols, costs, risk limits, backtest settings
tests/           pytest unit tests for every module above
mt5_ea/
  ModiriBot_EA.mq5   native MQL5 Expert Advisor -- same champion strategy, no Python needed
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # fill in your MT5 login/password/server (only needed for live trading)
```

Edit `config/config.yaml`: set your symbol's real spread, commission, pip value, and the account's starting balance and risk limits. The numbers shipped there are reasonable examples, not your broker's actual costs.

`MetaTrader5` (the Python package) only installs and works on **Windows**, next to a running MT5 terminal — that's a limitation of the MT5 API itself, not this project. Backtesting and optimization don't need it at all; they only need a CSV of historical bars.

## 1. Get real historical data

Either export it from MT5 (File → Open Data Folder → history, or Tools → History Center → Export), or, on a Windows machine with MT5 running:

```bash
python scripts/fetch_mt5_history.py --symbol EURUSD --timeframe H1 \
    --from 2022-01-01 --to 2026-07-01 --out data/EURUSD_H1.csv
```

## 2. Backtest a single strategy

```bash
python scripts/run_backtest.py --data data/EURUSD_H1.csv --symbol EURUSD \
    --strategy donchian_breakout --params '{"period": 20}' --plot results/equity.png
```

## 3. Search for the best strategy / ensemble (walk-forward)

```bash
python scripts/optimize_strategies.py --data data/EURUSD_H1.csv --symbol EURUSD
```

This writes the winning single-strategy and ensemble configs to `config/best_strategy.json`, along with their **holdout** (never-optimized-on) performance. Read that holdout number, not the in-sample search score — the search score will almost always look better than what actually held up.

Try it against synthetic data first to see the search correctly report failure when there's no real edge to find:

```bash
python scripts/optimize_strategies.py --demo
```

## 4. Go live (start on a demo account)

By default this trades the validated champion in `config/current_best_strategy.json` (an ensemble found across 15M+ tested combinations, holdout Sharpe 2.71 / return +8.20% over ~9 months — see that file's `production_note` for the full history). Run on a **Windows machine or VPS with your MT5 terminal installed and logged in** — this cannot run inside a Linux sandbox or without a live terminal, since the `MetaTrader5` package talks to the local terminal process, not the broker's server directly.

```bash
cp .env.example .env        # fill in MT5_LOGIN / MT5_PASSWORD / MT5_SERVER
pip install -r requirements.txt
python scripts/run_live.py --symbol EURUSD --use ensemble --poll-seconds 30
```

This polls MT5 for new closed bars, asks the chosen strategy for a signal, and manages a single position per symbol subject to `config/config.yaml`'s risk limits (per-trade risk %, daily loss limit, drawdown kill-switch) plus the three production overlays baked into that config (15-bar time stop, volatility-percentile filter, trailing stop). Run it against a **demo account** for a meaningful stretch of time before ever pointing it at real money, and only risk capital you can afford to lose.

To trade a different strategy config, pass `--best-strategy-file path/to/file.json` (e.g. one produced by `scripts/optimize_strategies.py`).

### Alternative: native MT5 Expert Advisor (no Python needed)

`mt5_ea/ModiriBot_EA.mq5` is a self-contained MQL5 port of the same champion
strategy and risk overlays -- compile it in MetaEditor and attach it directly
to an EURUSD H4 chart, no Python process required. It also draws a live
on-chart dashboard (balance, equity, trade count, win rate, P&L). See
`mt5_ea/README.md` for install/compile steps.

## Tests

```bash
pytest
```

34 unit tests cover the strategies, backtest engine (including stop-loss/take-profit/cost mechanics and the drawdown kill-switch), metrics, position sizing, the risk manager, the CSV loader, and the optimizer's data splitting.
