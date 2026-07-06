//+------------------------------------------------------------------+
//|                                                  ModiriBot_EA.mq5 |
//|                                    Copyright 2026, Pedram Kamangar |
//+------------------------------------------------------------------+
//
// MODIRI BOT -- fully automatic Expert Advisor.
//
// This is a native MQL5 port of the validated "champion" ensemble found
// in the modiri_bot Python research project (config/current_best_strategy.json):
// a weighted vote of 2x RSI-reversion + MFI-reversion + CCI-reversion,
// plus the 3 risk overlays that were tested and validated on top of it
// (15-bar time stop, ATR volatility-percentile position-size filter,
// and a 0.4R/0.4R trailing stop). All default input values below are
// exactly the numbers that produced, on ~2.5 years of real EURUSD H4
// data (holdout / never-optimized-on segment): +8.20% return, Sharpe
// 2.71, max drawdown 2.69%, 76.3% win rate, 59 trades over ~9 months.
//
// IMPORTANT: this was validated specifically on EURUSD, H4 timeframe.
// Attach it to an EURUSD H4 chart. Using it on another symbol/timeframe
// without re-validating there first is not covered by the numbers above.
//
// Past backtest performance is not a guarantee of future results. Start
// on a DEMO account and watch it for a meaningful stretch of time before
// ever pointing it at real money.
//
#property copyright "Pedram Kamangar"
#property link      "https://github.com/pdekam2000/modiri"
#property version   "1.00"
#property description "Modiri Bot -- automated EURUSD H4 mean-reversion ensemble"
#property strict

#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| Inputs                                                            |
//+------------------------------------------------------------------+
input group "===== Ensemble members (validated champion weights) ====="
input int    InpRSI1_Period       = 7;
input double InpRSI1_Oversold     = 20.0;
input double InpRSI1_Overbought    = 90.0;
input double InpRSI1_Weight        = 1.6662639379501343;

input int    InpRSI2_Period       = 25;
input double InpRSI2_Oversold      = 30.0;
input double InpRSI2_Overbought    = 85.0;
input double InpRSI2_Weight        = 1.6935628652572632;

input int    InpMFI_Period        = 21;
input double InpMFI_Oversold       = 25.0;
input double InpMFI_Overbought     = 85.0;
input double InpMFI_Weight         = 1.5513395071029663;

input int    InpCCI_Period        = 14;
input double InpCCI_Threshold      = 200.0;
input double InpCCI_Weight         = 1.4190824031829834;

input double InpEnsembleThreshold  = 0.2684963643550873;
input int    InpStateLookbackBars  = 300;   // bars used to rebuild each sub-strategy's on/off state

input group "===== Risk management ====="
input double InpRiskPerTradePct    = 1.0;   // % of equity risked on the stop-loss distance of one trade
input double InpStopLossPips       = 40.0;
input double InpTakeProfitPips     = 80.0;
input double InpMaxDailyLossPct    = 3.0;   // stop opening new trades for the day after this loss
input double InpMaxDrawdownPct     = 15.0;  // kill-switch: stop trading entirely until EA is restarted
input int    InpMaxHoldBars        = 15;    // time stop: force-exit after N H4 bars regardless of signal

input group "===== Volatility filter ====="
input bool   InpUseVolatilityFilter       = true;
input double InpVolPercentileThreshold    = 95.0;
input double InpVolSizeMult               = 0.5;
input int    InpVolLookbackBars           = 500;
input int    InpATR_Period                = 14;

input group "===== Trailing stop ====="
input bool   InpUseTrailingStop    = true;
input double InpTrailingStartR     = 0.4;
input double InpTrailingDistR      = 0.4;

input group "===== Execution ====="
input ulong  InpMagicNumber        = 20260702;
input int    InpSlippagePoints     = 20;

input group "===== Dashboard ====="
input bool   InpShowDashboard      = true;
input string InpOwnerName          = "Pedram Kamangar";

//+------------------------------------------------------------------+
//| Globals                                                            |
//+------------------------------------------------------------------+
CTrade trade;

int hRSI1 = INVALID_HANDLE;
int hRSI2 = INVALID_HANDLE;
int hMFI  = INVALID_HANDLE;
int hCCI  = INVALID_HANDLE;
int hATR  = INVALID_HANDLE;

double   g_bestPrice        = 0.0;     // best favorable price seen since entry (trailing stop)
double   g_equityPeak       = 0.0;
double   g_dayStartEquity   = 0.0;
datetime g_currentDay       = 0;
bool     g_haltedForGood    = false;
datetime g_lastBarTime      = 0;
uint     g_lastDashboardTick = 0;

string   g_prefix = "ModiriBot_";

//+------------------------------------------------------------------+
//| Small helpers                                                     |
//+------------------------------------------------------------------+
double Pip()
  {
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(digits == 3 || digits == 5)
      return point * 10.0;
   return point;
  }

int CountOurPositions()
  {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;
      count++;
     }
   return count;
  }

int CurrentPositionSide()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;
      return (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
     }
   return 0;
  }

void CloseOurPositions()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;
      trade.PositionClose(ticket, InpSlippagePoints);
     }
   g_bestPrice = 0.0;
  }

//+------------------------------------------------------------------+
//| Replays a mean-reversion state machine over the last `lookback`   |
//| closed bars of an oscillator, exactly mirroring the Python        |
//| strategies: long below `oversold`, short above `overbought`,      |
//| flat once the oscillator crosses back through `exitMid`.          |
//+------------------------------------------------------------------+
int ReversionState(int handle, double oversold, double overbought, double exitMid, int lookback)
  {
   double buf[];
   ArraySetAsSeries(buf, true);
   int copied = CopyBuffer(handle, 0, 1, lookback, buf);
   if(copied <= 0)
      return 0;

   int position = 0;
   for(int i = copied - 1; i >= 0; i--)   // oldest -> newest, ending at the last closed bar
     {
      double v = buf[i];
      if(position == 0)
        {
         if(v < oversold)
            position = 1;
         else if(v > overbought)
            position = -1;
        }
      else if(position == 1 && v >= exitMid)
         position = 0;
      else if(position == -1 && v <= exitMid)
         position = 0;
     }
   return position;
  }

int ComputeEnsembleSignal(int &s1, int &s2, int &s3, int &s4, double &combinedOut)
  {
   s1 = ReversionState(hRSI1, InpRSI1_Oversold, InpRSI1_Overbought, 50.0, InpStateLookbackBars);
   s2 = ReversionState(hRSI2, InpRSI2_Oversold, InpRSI2_Overbought, 50.0, InpStateLookbackBars);
   s3 = ReversionState(hMFI,  InpMFI_Oversold,  InpMFI_Overbought,  50.0, InpStateLookbackBars);
   s4 = ReversionState(hCCI, -InpCCI_Threshold, InpCCI_Threshold,    0.0, InpStateLookbackBars);

   double totalWeight = MathAbs(InpRSI1_Weight) + MathAbs(InpRSI2_Weight)
                       + MathAbs(InpMFI_Weight) + MathAbs(InpCCI_Weight);
   if(totalWeight <= 0) totalWeight = 1.0;

   double combined = (s1 * InpRSI1_Weight + s2 * InpRSI2_Weight
                     + s3 * InpMFI_Weight  + s4 * InpCCI_Weight) / totalWeight;
   combinedOut = combined;

   if(combined > InpEnsembleThreshold)  return 1;
   if(combined < -InpEnsembleThreshold) return -1;
   return 0;
  }

//+------------------------------------------------------------------+
//| Position sizing: risk a fixed % of equity on the stop distance.   |
//+------------------------------------------------------------------+
double LotsForFixedRisk(double equity, double riskPct, double slPips)
  {
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double pip       = Pip();
   double minLot    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lotStep   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double maxLot    = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   double pipValuePerLot = (tickSize > 0) ? tickValue * (pip / tickSize) : 10.0;
   if(slPips <= 0 || pipValuePerLot <= 0)
      return minLot;

   double riskAmount = equity * (riskPct / 100.0);
   double rawLots = riskAmount / (slPips * pipValuePerLot);

   double steps = MathFloor(rawLots / lotStep + 1e-8);
   double lots = MathMax(steps, 0) * lotStep;
   lots = MathMax(lots, minLot);
   lots = MathMin(lots, maxLot);
   return NormalizeDouble(lots, 2);
  }

double VolatilityPercentileRank()
  {
   double buf[];
   ArraySetAsSeries(buf, true);
   int copied = CopyBuffer(hATR, 0, 1, InpVolLookbackBars, buf);
   if(copied < 20)
      return 0.0;

   double current = buf[0];
   int countBelow = 0;
   for(int i = 0; i < copied; i++)
      if(current > buf[i])
         countBelow++;
   return (double)countBelow / copied * 100.0;
  }

//+------------------------------------------------------------------+
//| Risk state: daily loss limit + permanent drawdown kill-switch     |
//| (same semantics as modiri_bot/risk/risk_manager.py).               |
//+------------------------------------------------------------------+
void UpdateRiskState()
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime today = StructToTime(dt);
   if(today != g_currentDay)
     {
      g_currentDay = today;
      g_dayStartEquity = equity;
     }

   if(equity > g_equityPeak)
      g_equityPeak = equity;

   if(InpMaxDrawdownPct > 0 && g_equityPeak > 0 && !g_haltedForGood)
     {
      double ddPct = (g_equityPeak - equity) / g_equityPeak * 100.0;
      if(ddPct >= InpMaxDrawdownPct)
        {
         g_haltedForGood = true;
         Print("MODIRI BOT: drawdown kill-switch triggered at ", DoubleToString(ddPct, 2),
               "% -- no new trades will be opened until the EA is removed and re-attached.");
        }
     }
  }

bool CanOpenNewTrade(double equity)
  {
   if(g_haltedForGood)
      return false;
   if(CountOurPositions() >= 1)
      return false;
   if(InpMaxDailyLossPct > 0 && g_dayStartEquity > 0)
     {
      double dayLossPct = (g_dayStartEquity - equity) / g_dayStartEquity * 100.0;
      if(dayLossPct >= InpMaxDailyLossPct)
         return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Trailing stop -- runs every tick for live intrabar precision.     |
//+------------------------------------------------------------------+
void UpdateTrailingStop()
  {
   if(!InpUseTrailingStop)
      return;

   double slDist = InpStopLossPips * Pip();
   if(slDist <= 0)
      return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

      long   type       = PositionGetInteger(POSITION_TYPE);
      double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      double currentSL  = PositionGetDouble(POSITION_SL);
      double currentTP  = PositionGetDouble(POSITION_TP);

      if(type == POSITION_TYPE_BUY)
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(g_bestPrice == 0.0 || bid > g_bestPrice)
            g_bestPrice = bid;
         double profitR = (g_bestPrice - entryPrice) / slDist;
         if(profitR >= InpTrailingStartR)
           {
            double newSL = g_bestPrice - InpTrailingDistR * slDist;
            if(newSL > currentSL)
               trade.PositionModify(ticket, newSL, currentTP);
           }
        }
      else
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         if(g_bestPrice == 0.0 || ask < g_bestPrice)
            g_bestPrice = ask;
         double profitR = (entryPrice - g_bestPrice) / slDist;
         if(profitR >= InpTrailingStartR)
           {
            double newSL = g_bestPrice + InpTrailingDistR * slDist;
            if(currentSL == 0.0 || newSL < currentSL)
               trade.PositionModify(ticket, newSL, currentTP);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Time stop: force-exit after InpMaxHoldBars closed bars.           |
//| Returns true if a position was closed this call.                  |
//+------------------------------------------------------------------+
bool ApplyTimeStop()
  {
   if(InpMaxHoldBars <= 0)
      return false;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

      datetime entryTime = (datetime)PositionGetInteger(POSITION_TIME);
      double barsHeld = (double)(TimeCurrent() - entryTime) / PeriodSeconds(_Period);
      if(barsHeld >= InpMaxHoldBars)
        {
         trade.PositionClose(ticket, InpSlippagePoints);
         Print("MODIRI BOT: time stop closed position ", ticket, " after ",
               DoubleToString(barsHeld, 1), " bars");
         g_bestPrice = 0.0;
         return true;
        }
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Open a new position sized by fixed-risk + volatility filter.      |
//+------------------------------------------------------------------+
void OpenPosition(int signal, double equity)
  {
   double riskPct = InpRiskPerTradePct;
   if(InpUseVolatilityFilter)
     {
      double percentileRank = VolatilityPercentileRank();
      if(percentileRank > InpVolPercentileThreshold)
         riskPct *= InpVolSizeMult;
     }

   double pip = Pip();
   double lots = LotsForFixedRisk(equity, riskPct, InpStopLossPips);
   if(lots <= 0)
      return;

   bool ok;
   if(signal == 1)
     {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = price - InpStopLossPips * pip;
      double tp = price + InpTakeProfitPips * pip;
      ok = trade.Buy(lots, _Symbol, price, sl, tp, "modiri_bot");
      if(ok) g_bestPrice = price;
     }
   else
     {
      double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = price + InpStopLossPips * pip;
      double tp = price - InpTakeProfitPips * pip;
      ok = trade.Sell(lots, _Symbol, price, sl, tp, "modiri_bot");
      if(ok) g_bestPrice = price;
     }

   if(!ok)
      Print("MODIRI BOT: order failed, retcode=", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
   else
      Print("MODIRI BOT: opened ", (signal == 1 ? "BUY" : "SELL"), " ", DoubleToString(lots, 2),
            " lots on ", _Symbol, " (risk ", DoubleToString(riskPct, 2), "%)");
  }

//+------------------------------------------------------------------+
//| Once-per-closed-bar decision loop (mirrors LiveTrader.poll_once). |
//+------------------------------------------------------------------+
void ManageBar()
  {
   if(ApplyTimeStop())
      return; // let the next bar decide whether to re-enter

   int s1, s2, s3, s4;
   double combined;
   int signal = ComputeEnsembleSignal(s1, s2, s3, s4, combined);
   int currentSide = CurrentPositionSide();

   Print("MODIRI BOT bar check: signal=", signal, " currentSide=", currentSide,
         " sub-states(rsi1,rsi2,mfi,cci)=", s1, ",", s2, ",", s3, ",", s4,
         " combined=", DoubleToString(combined, 3));

   if(signal == currentSide)
      return;

   if(currentSide != 0)
      CloseOurPositions();

   if(signal == 0)
      return;

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(!CanOpenNewTrade(equity))
      return;

   OpenPosition(signal, equity);
  }

bool IsNewBar()
  {
   datetime t = iTime(_Symbol, _Period, 0);
   if(t != g_lastBarTime)
     {
      g_lastBarTime = t;
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Dashboard: colorful on-chart panel with owner name + live trade   |
//| report (balance, equity, trades, win rate, P&L, kill-switch).     |
//+------------------------------------------------------------------+
void CreateRect(string name, int x, int y, int w, int h, color bg, color border)
  {
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetInteger(0, name, OBJPROP_XSIZE, w);
   ObjectSetInteger(0, name, OBJPROP_YSIZE, h);
   ObjectSetInteger(0, name, OBJPROP_BGCOLOR, bg);
   ObjectSetInteger(0, name, OBJPROP_COLOR, border);
   ObjectSetInteger(0, name, OBJPROP_BORDER_TYPE, BORDER_FLAT);
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
  }

void CreateLabel(string name, int x, int y, string text, color clr, int size, bool bold = false)
  {
   if(ObjectFind(0, name) < 0)
      ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(0, name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   ObjectSetInteger(0, name, OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, name, OBJPROP_YDISTANCE, y);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
   ObjectSetString(0, name, OBJPROP_FONT, bold ? "Arial Bold" : "Arial");
   ObjectSetInteger(0, name, OBJPROP_FONTSIZE, size);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
  }

#define PANEL_X 12
#define PANEL_W 300

void CreateDashboard()
  {
   color navy   = (color)C'16,22,38';
   color slate  = (color)C'24,30,46';
   color gold   = (color)C'255,196,0';
   color teal   = (color)C'0,180,180';
   color silver = (color)C'190,196,206';

   // Header: a small colored "MB" badge stands in for a logo (kept ASCII-only
   // so the file compiles cleanly regardless of MetaEditor's text encoding).
   CreateRect(g_prefix + "hdr_bg", PANEL_X, 12, PANEL_W, 46, navy, gold);
   CreateRect(g_prefix + "hdr_accent", PANEL_X, 12, PANEL_W, 4, gold, gold);
   CreateRect(g_prefix + "logo_bg", PANEL_X + 10, 20, 28, 28, teal, gold);
   CreateLabel(g_prefix + "logo", PANEL_X + 15, 26, "MB", navy, 12, true);
   CreateLabel(g_prefix + "title", PANEL_X + 48, 18, "MODIRI BOT", gold, 15, true);
   CreateLabel(g_prefix + "subtitle", PANEL_X + 48, 39, "Auto FX Trading | " + InpOwnerName, silver, 8);

   // Body
   int bodyY = 12 + 46 + 6;
   int bodyH = 245;
   CreateRect(g_prefix + "body_bg", PANEL_X, bodyY, PANEL_W, bodyH, slate, teal);

   // Note: MT5 ignores ObjectSetString(...OBJPROP_TEXT, "") on a freshly
   // created OBJ_LABEL and leaves its built-in default text "Label" visible,
   // so spacer rows must not be created as label objects at all -- they
   // only ever need to reserve vertical space between real rows below.
   string rows[] =
     {
      "sym", "status", "sep1",
      "balance", "equity", "floatpl", "sep2",
      "trades", "winrate", "pf", "netpl", "sep3",
      "position", "signal"
     };
   int y = bodyY + 10;
   for(int i = 0; i < ArraySize(rows); i++)
     {
      bool isSep = (StringFind(rows[i], "sep") == 0);
      if(!isSep)
         CreateLabel(g_prefix + "row_" + rows[i], PANEL_X + 12, y, " ", silver, 9);
      y += isSep ? 8 : 17;
     }
  }

void RemoveDashboard()
  {
   ObjectsDeleteAll(0, g_prefix);
  }

void ComputeTradeStats(int &totalTrades, int &wins, double &grossProfit, double &grossLoss, double &netPnl)
  {
   totalTrades = 0; wins = 0; grossProfit = 0.0; grossLoss = 0.0; netPnl = 0.0;
   HistorySelect(0, TimeCurrent());
   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong dealTicket = HistoryDealGetTicket(i);
      if(dealTicket == 0) continue;
      if(HistoryDealGetInteger(dealTicket, DEAL_MAGIC) != (long)InpMagicNumber) continue;
      if(HistoryDealGetString(dealTicket, DEAL_SYMBOL) != _Symbol) continue;
      if(HistoryDealGetInteger(dealTicket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;

      double profit = HistoryDealGetDouble(dealTicket, DEAL_PROFIT)
                    + HistoryDealGetDouble(dealTicket, DEAL_SWAP)
                    + HistoryDealGetDouble(dealTicket, DEAL_COMMISSION);
      totalTrades++;
      netPnl += profit;
      if(profit > 0)
        {
         wins++;
         grossProfit += profit;
        }
      else
         grossLoss += -profit;
     }
  }

void UpdateDashboard()
  {
   color green = (color)C'80,220,120';
   color red   = (color)C'240,90,90';
   color silver = (color)C'190,196,206';
   color gold  = (color)C'255,196,0';

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double floatPl = equity - balance;

   int totalTrades, wins;
   double grossProfit, grossLoss, netPnl;
   ComputeTradeStats(totalTrades, wins, grossProfit, grossLoss, netPnl);
   double winRate = (totalTrades > 0) ? (double)wins / totalTrades * 100.0 : 0.0;
   double pf = (grossLoss > 0) ? grossProfit / grossLoss : (grossProfit > 0 ? 999.0 : 0.0);

   int side = CurrentPositionSide();
   string sideText = (side == 1) ? "LONG open" : (side == -1) ? "SHORT open" : "flat";

   ObjectSetString(0, g_prefix + "row_sym", OBJPROP_TEXT, _Symbol + "  " + EnumToString(_Period));
   ObjectSetInteger(0, g_prefix + "row_sym", OBJPROP_COLOR, silver);

   ObjectSetString(0, g_prefix + "row_status", OBJPROP_TEXT,
                    g_haltedForGood ? "STATUS: HALTED (drawdown)" : "STATUS: active");
   ObjectSetInteger(0, g_prefix + "row_status", OBJPROP_COLOR, g_haltedForGood ? red : green);

   ObjectSetString(0, g_prefix + "row_balance", OBJPROP_TEXT,
                    "Balance: " + DoubleToString(balance, 2) + "   Equity: " + DoubleToString(equity, 2));
   ObjectSetInteger(0, g_prefix + "row_balance", OBJPROP_COLOR, silver);

   ObjectSetString(0, g_prefix + "row_equity", OBJPROP_TEXT,
                    "Peak equity: " + DoubleToString(g_equityPeak, 2));
   ObjectSetInteger(0, g_prefix + "row_equity", OBJPROP_COLOR, silver);

   ObjectSetString(0, g_prefix + "row_floatpl", OBJPROP_TEXT,
                    "Floating P/L: " + DoubleToString(floatPl, 2));
   ObjectSetInteger(0, g_prefix + "row_floatpl", OBJPROP_COLOR, (floatPl >= 0) ? green : red);

   ObjectSetString(0, g_prefix + "row_trades", OBJPROP_TEXT,
                    "Total trades: " + IntegerToString(totalTrades));
   ObjectSetInteger(0, g_prefix + "row_trades", OBJPROP_COLOR, silver);

   ObjectSetString(0, g_prefix + "row_winrate", OBJPROP_TEXT,
                    "Win rate: " + DoubleToString(winRate, 1) + "%");
   ObjectSetInteger(0, g_prefix + "row_winrate", OBJPROP_COLOR, silver);

   ObjectSetString(0, g_prefix + "row_pf", OBJPROP_TEXT,
                    "Profit factor: " + DoubleToString(pf, 2));
   ObjectSetInteger(0, g_prefix + "row_pf", OBJPROP_COLOR, silver);

   ObjectSetString(0, g_prefix + "row_netpl", OBJPROP_TEXT,
                    "Net P/L (closed): " + DoubleToString(netPnl, 2));
   ObjectSetInteger(0, g_prefix + "row_netpl", OBJPROP_COLOR, (netPnl >= 0) ? green : red);

   ObjectSetString(0, g_prefix + "row_position", OBJPROP_TEXT, "Position: " + sideText);
   ObjectSetInteger(0, g_prefix + "row_position", OBJPROP_COLOR, gold);

   ObjectSetString(0, g_prefix + "row_signal", OBJPROP_TEXT,
                    "Magic: " + IntegerToString((long)InpMagicNumber));
   ObjectSetInteger(0, g_prefix + "row_signal", OBJPROP_COLOR, silver);

   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
//| Expert lifecycle                                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpSlippagePoints);
   trade.SetTypeFillingBySymbol(_Symbol);

   hRSI1 = iRSI(_Symbol, _Period, InpRSI1_Period, PRICE_CLOSE);
   hRSI2 = iRSI(_Symbol, _Period, InpRSI2_Period, PRICE_CLOSE);
   hMFI  = iMFI(_Symbol, _Period, InpMFI_Period, VOLUME_TICK);
   hCCI  = iCCI(_Symbol, _Period, InpCCI_Period, PRICE_TYPICAL);
   hATR  = iATR(_Symbol, _Period, InpATR_Period);

   if(hRSI1 == INVALID_HANDLE || hRSI2 == INVALID_HANDLE || hMFI == INVALID_HANDLE
      || hCCI == INVALID_HANDLE || hATR == INVALID_HANDLE)
     {
      Print("MODIRI BOT: failed to create one or more indicator handles");
      return INIT_FAILED;
     }

   if(_Period != PERIOD_H4)
      Print("MODIRI BOT: WARNING - this ensemble was validated on the H4 timeframe. "
            "Running on ", EnumToString(_Period), " is untested.");
   if(StringFind(_Symbol, "EURUSD") < 0)
      Print("MODIRI BOT: WARNING - this ensemble was validated on EURUSD. "
            "Running on ", _Symbol, " is untested.");

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_equityPeak = equity;
   g_dayStartEquity = equity;
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   g_currentDay = StructToTime(dt);
   g_haltedForGood = false;
   g_lastBarTime = 0;

   // If a position already exists (EA re-attached mid-trade), seed the
   // trailing-stop tracker with the current market price as a safe start.
   if(CountOurPositions() > 0)
     {
      int side = CurrentPositionSide();
      g_bestPrice = (side == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                                 : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
     }

   if(InpShowDashboard)
     {
      CreateDashboard();
      UpdateDashboard();
     }

   Print("MODIRI BOT initialized on ", _Symbol, " ", EnumToString(_Period), " -- owner: ", InpOwnerName);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   if(hRSI1 != INVALID_HANDLE) IndicatorRelease(hRSI1);
   if(hRSI2 != INVALID_HANDLE) IndicatorRelease(hRSI2);
   if(hMFI  != INVALID_HANDLE) IndicatorRelease(hMFI);
   if(hCCI  != INVALID_HANDLE) IndicatorRelease(hCCI);
   if(hATR  != INVALID_HANDLE) IndicatorRelease(hATR);
   RemoveDashboard();
  }

void OnTick()
  {
   UpdateRiskState();
   UpdateTrailingStop();

   if(IsNewBar())
      ManageBar();

   if(InpShowDashboard)
     {
      uint now = GetTickCount();
      if(now - g_lastDashboardTick > 1000)
        {
         UpdateDashboard();
         g_lastDashboardTick = now;
        }
     }
  }
//+------------------------------------------------------------------+
