import pandas as pd
import numpy as np
from math import sqrt
from typing import Dict
from alpha_research.includes import PositionManager, Side, Trade
from collections import defaultdict
from typing import List, Dict, Any, Tuple

class ResultReport:
    def __init__(self):
        self.initial_capital = 100

        # dict of ticker -> list (for flexibility if you want multiple snapshots)
        self.equity_curve = defaultdict(list)
        self.original_curve = defaultdict(list)
        self.timestamps = defaultdict(list)
        self.returns = defaultdict(list)
        self.pos_trades = defaultdict(list)
        self.neg_trades = defaultdict(list)
        self.total_trades = defaultdict(list)

    def calculate_round_trip_stats(self, trades: List[Trade]) -> Tuple[int, int, int]:
   
        pos_trades = 0
        neg_trades = 0
        total_trades = 0

        open_trades = []   # store (price, qty, side)
        cum_pnl = 0.0
        net_qty = 0

        for tr in trades:
            # Maintain running open trades
            open_trades.append(tr)
            net_qty += tr.quantity

            # ✅ When position closes fully
            if net_qty == 0:
                # Compute PnL for this round trip
                entry_price = 0.0
                exit_price = 0.0
                buy_value = 0.0
                sell_value = 0.0
                buy_qty = 0
                sell_qty = 0

                for t in open_trades:
                    if t.side == Side.BUY:
                        buy_value += t.price * t.quantity
                        buy_qty += t.quantity
                    else:
                        sell_value += t.price * -t.quantity
                        sell_qty += -t.quantity

                if buy_qty == sell_qty and buy_qty > 0:
                    cum_pnl = sell_value - buy_value
                    if cum_pnl > 0:
                        pos_trades += 1
                    elif cum_pnl < 0:
                        neg_trades += 1
                    total_trades += 1

                # reset for next cycle
                open_trades.clear()
                cum_pnl = 0.0

        return pos_trades, neg_trades, total_trades

    def update(self, pm: PositionManager, ts) -> None:
        """Call this every 100ms with PositionManager snapshot"""
        for ticker , pos in pm.position_map.items():
            self.equity_curve[ticker].append(pos.net_realised_pnl)
            self.original_curve[ticker].append(pos.net_pnl)
            self.timestamps[ticker].append(ts)
            pos_trades, neg_trades, total_trades = self.calculate_round_trip_stats(pos.trades)
            self.pos_trades[ticker].append(pos_trades)
            self.neg_trades[ticker].append(neg_trades)
            self.total_trades[ticker].append(total_trades)

    
       

    def generate_report(self) -> Dict[str, Dict[str, float]]:
    
        report = {}

        for ticker, increments in self.equity_curve.items():
            equity_increments = np.array(increments, dtype=float)

            if equity_increments.size == 0:
                continue

            # --- Total realised PnL (sum of all increments) ---
            pnl_rs = float(equity_increments.sum())

            # --- Final equity (initial capital + pnl) ---
            final_equity = self.initial_capital + pnl_rs

            # --- PnL % ---
            pnl_pct = (final_equity - self.initial_capital) / self.initial_capital * 100.0

            # --- Build cumulative equity curve in absolute terms ---
            equity = self.initial_capital + np.cumsum(equity_increments)

            # --- Returns (step-by-step percentage changes) ---
            rets = np.diff(equity) / (np.abs(equity[:-1]) + 1e-9)
            self.returns[ticker] = rets.tolist()

            # --- Sharpe ratio ---
            sharpe = 0.0
            if rets.size > 1 and np.std(rets) > 0:
                sharpe = (np.mean(rets) / np.std(rets)) * sqrt(len(rets))

            # --- Sortino ratio ---
            sortino = 0.0
            downside = rets[rets < 0]
            if downside.size > 0 and np.std(downside) > 0:
                sortino = (np.mean(rets) / np.std(downside)) * sqrt(len(rets))

            # --- Max drawdown (relative to starting capital) ---
            pnl_cumsum = np.cumsum(equity_increments)
                
            # 1. Calculate the running maximum (the high water mark)
            running_max = np.maximum.accumulate(pnl_cumsum)
                
            # 2. Calculate the drawdowns from the peak (all values will be >= 0)
            drawdowns = running_max - pnl_cumsum
                
            # 3. Find the largest drawdown (in currency terms)
            max_drawdown_value = np.max(drawdowns)
                
            # 4. Calculate max_dd as a percentage of initial capital
            if self.initial_capital > 0:
                max_dd = (max_drawdown_value / self.initial_capital) * 100.0
            else:
                max_dd = 0.0 # Or np.nan

            # --- Calmar ratio ---
            calmar = 0.0

            # max_dd is now a positive percentage (e.g., 5.5 for 5.5%)
            if max_dd > 0:
                # Assumes pnl_pct is also a percentage (e.g., 14.08)
                calmar = pnl_pct / max_dd

            # --- Trade stats ---
            pos_trades = sum(self.pos_trades.get(ticker, []))
            neg_trades = sum(self.neg_trades.get(ticker, []))
            total_trades = sum(self.total_trades.get(ticker, []))
            win_rate = (pos_trades / total_trades) * 100 if total_trades > 0 else 0.0

            report[ticker] = {
                "Final_Equity": final_equity,
                "PnL_Rs": pnl_rs,
                "PnL_%": pnl_pct,
                "Sharpe": float(sharpe),
                "Sortino": float(sortino),
                "Calmar": float(calmar),
                "Max_Drawdown_%": float(max_dd),
                "Positive_Trades": pos_trades,
                "Negative_Trades": neg_trades,
                "Total_Trades": total_trades,
                "WinRate_%": win_rate,
            }

        return report
    
    def export_equity_curve(self) -> Dict[str, pd.DataFrame]:
        reports: Dict[str, pd.DataFrame] = {}

        for ticker, realised_list in self.equity_curve.items():
            pnl_after_tc = np.asarray(realised_list, dtype=float)
            pnl_before_tc = np.asarray(self.original_curve.get(ticker, []), dtype=float)
            ts_list       = list(self.timestamps.get(ticker, []))
            pos_list      = np.asarray(self.pos_trades.get(ticker, []), dtype=float)
            neg_list      = np.asarray(self.neg_trades.get(ticker, []), dtype=float)
            total_list    = np.asarray(self.total_trades.get(ticker, []), dtype=float)

            n = min(len(pnl_after_tc), len(pnl_before_tc), len(ts_list),
                    len(pos_list), len(neg_list), len(total_list))
            if n == 0:
                continue

            pnl_after_tc  = pnl_after_tc[:n]
            pnl_before_tc = pnl_before_tc[:n]
            ts_list       = ts_list[:n]
            pos_list      = pos_list[:n]
            neg_list      = neg_list[:n]
            total_list    = total_list[:n]

            # cumulative pnl (Rs)
            pnl_cumsum = np.cumsum(pnl_after_tc)

            # pnl % relative to initial capital
            pnl_pct = (pnl_after_tc / self.initial_capital) * 100.0

            # drawdown vs fixed baseline (100)
            drawdown_rs_inst   = np.minimum(pnl_cumsum, 0.0)
            drawdown_pct_inst  = (drawdown_rs_inst / self.initial_capital) * 100.0
            drawdown_pct_run   = np.minimum.accumulate(drawdown_pct_inst)

            # hit rate per day
            with np.errstate(divide="ignore", invalid="ignore"):
                hit_rate = np.where(total_list > 0, (pos_list / total_list) * 100.0, np.nan)

            df = pd.DataFrame({
                "day":    ts_list,
                "pnl":          pnl_before_tc,
                "pnl_with_tc": pnl_after_tc,
                "pnl_pct":      pnl_pct,
                "pnl_cumsum":      pnl_cumsum,
                "max_drawdown_pct": drawdown_pct_run,
                "pos_trades":   pos_list,
                "neg_trades":   neg_list,
                "total_trades": total_list,
                "hit_rate_%":   hit_rate
            })

            reports[ticker] = df

        return reports
