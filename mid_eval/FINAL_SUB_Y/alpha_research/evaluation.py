"""evaluation.py – ResearchInfra Metrics Helper
=============================================
A compact, production‑ready utility for accumulating prediction events and
reporting error metrics in a high‑frequency‑trading research framework.

Implemented capabilities
------------------------
* **Price regression**  – RMSE, MAD, *weighted* RMSE/MAD.
* **Direction classification** – Accuracy, macro Precision/Recall/F1,
  macro ROC‑AUC.
* **Range prediction** – Separate RMSE & MAD for *high* and *low* bands.

Public API (CamelCase)
----------------------
* ``onPredictedPrice(target_price, predicted_price, curr_price, curr_timestamp)``
* ``onPredictedDirection(target_price, predicted_direction, curr_price, curr_timestamp)``
* ``onPredictedRange(target_price, predicted_high, predicted_low, curr_timestamp, curr_price)``
* ``reportResults() → dict[str, float]``

All metrics are lazily computed on demand so ingesting events stays O(1).

Usage example
-------------
```python
from evaluation import Evaluation

ev = Evaluation(token=1337, target_time=1_000_000)
# feed events ...
print(ev.reportResults())
```
"""
from __future__ import annotations

import math
import statistics
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

__all__ = ["Evaluation"]


class Evaluation:
    """Collects prediction events and emits evaluation metrics."""

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        token: int,
        target_time: int,
        data_type: str = "TIMER",
        window: int = 50,
    ) -> None:
        self.token: int = token
        self.target_time: int = target_time
        self.data_type: str = data_type
        self._window: int = window

        # rolling buffers -------------------------------------------------
        self._rolling_prices: List[float] = []  # for price‑weight calc
        self._rolling_prices_dir: List[float] = []  # for direction bands
        self._rolling_prices_range: List[float] = []  # for range bands

        # event stores ----------------------------------------------------
        self._price_events: List[Dict] = []
        self._dir_events: List[Dict] = []
        self._range_events: List[Dict] = []

        # cumulative PnL trackers
        self._pnl_price: float = 0.0
        self._pnl_dir: float = 0.0

    # ------------------------------------------------------- static helpers
    @staticmethod
    def _truncate(buffer: List[float], window: int) -> None:
        while len(buffer) > window:
            buffer.pop(0)

    # ---------------------- weighting / target‑generation helpers ---------
    @staticmethod
    def defaultWeight(curr_price: float, target_price: float) -> float:
        """Default absolute‑error weight."""
        return abs(curr_price - target_price)

    @staticmethod
    def computeTargetDirection(
        prev_prices: List[float],
        curr_price: float,
        target_price: float,
        window: int,
    ) -> int:
        """Derive direction label {-1, 0, +1} from rolling σ bands."""
        if len(prev_prices) < 2:
            return 0  # insufficient history
        std = statistics.pstdev(prev_prices[-window:]) if len(prev_prices) >= window else statistics.pstdev(prev_prices)
        upper_band = curr_price + std/2
        lower_band = curr_price - std/2
        if target_price > upper_band:
            return 1
        if target_price < lower_band:
            return -1
        return 0

    @staticmethod
    def computeTargetRange(
        prev_prices: List[float],
        target_price: float,
        window: int,
    ) -> Tuple[float, float]:
        """Return (target_high, target_low) using rolling σ."""
        if len(prev_prices) < 2:
            return target_price, target_price  # degenerate range
        std = statistics.pstdev(prev_prices[-window:]) if len(prev_prices) >= window else statistics.pstdev(prev_prices)
        return target_price + std, target_price - std

    # ----------------------------------------------------------- callbacks
    def onPredictedPrice(
        self,
        target_price: float,
        predicted_price: float,
        curr_timestamp: int,
        curr_price: float,
    ) -> None:
        """Ingest a price‑prediction event."""
        weight = self.defaultWeight(curr_price, target_price)
        self._price_events.append(
            {
                "ts": curr_timestamp,
                "curr_price": curr_price,
                "target_price": target_price,
                "predicted_price": predicted_price,
                "weight": weight,
            }
        )
        # update rolling buffer
        self._rolling_prices.append(curr_price)
        self._truncate(self._rolling_prices, self._window)

        # update price-based PnL
        if predicted_price > curr_price:           # long → sell at target
            self._pnl_price += (target_price - curr_price)
        elif predicted_price < curr_price:         # short → buy back at target
            self._pnl_price += (curr_price - target_price)

    def onPredictedDirection(
        self,
        target_price: float,
        predicted_direction: int,
        curr_timestamp: int,
        curr_price: float,
    ) -> None:
        """Ingest a direction‑classification event."""
        target_direction = self.computeTargetDirection(
            self._rolling_prices_dir, curr_price, target_price, self._window
        )
        self._dir_events.append(
            {
                "ts": curr_timestamp,
                "curr_price": curr_price,
                "target_price": target_price,
                "target_dir": target_direction,
                "predicted_dir": predicted_direction,
            }
        )
        # update buffer
        self._rolling_prices_dir.append(curr_price)
        self._truncate(self._rolling_prices_dir, self._window)

        # NEW: update direction-based PnL
        if predicted_direction == 1:               # long
            self._pnl_dir += (target_price - curr_price)
        elif predicted_direction == -1:            # short
            self._pnl_dir += (curr_price - target_price)
        # (predicted_direction == 0 → no position)


    def onPredictedRange(
        self,
        target_price: float,
        predicted_high: float,
        predicted_low: float,
        curr_timestamp: int,
        curr_price: float,
    ) -> None:
        """Ingest a range‑prediction event."""
        target_high, target_low = self.computeTargetRange(
            self._rolling_prices_range, target_price, self._window
        )
        self._range_events.append(
            {
                "ts": curr_timestamp,
                "curr_price": curr_price,
                "target_price": target_price,
                "target_high": target_high,
                "predicted_high": predicted_high,
                "target_low": target_low,
                "predicted_low": predicted_low,
            }
        )
        # update buffer
        self._rolling_prices_range.append(curr_price)
        self._truncate(self._rolling_prices_range, self._window)

    # ------------------------------------------------------------- metrics
    def reportResults(self) -> Dict[str, float]:
        """Compute and return all metrics in a dict."""
        results: Dict[str, float] = {}

        # ---------- price regression ------------------------------------
        if self._price_events:
            df_p = pd.DataFrame(self._price_events)
            mse = mean_squared_error(df_p["target_price"], df_p["predicted_price"])
            results["price_rmse"] = math.sqrt(mse)
            results["price_mad"] = mean_absolute_error(df_p["target_price"], df_p["predicted_price"])
            # weighted
            mse_w = mean_squared_error(
                df_p["target_price"], df_p["predicted_price"], sample_weight=df_p["weight"]
            )
            results["price_wrmse"] = math.sqrt(mse_w)
            results["price_wmad"] = mean_absolute_error(
                df_p["target_price"], df_p["predicted_price"], sample_weight=df_p["weight"]
            )
            results["pnl_price"] = self._pnl_price

        # ---------- direction classification ----------------------------
        if self._dir_events:
            df_d = pd.DataFrame(self._dir_events)
            y_true = df_d["target_dir"]
            y_pred = df_d["predicted_dir"]
            results["dir_accuracy"] = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[-1, 0, 1], average="macro", zero_division=0
            )
            results["dir_precision_macro"] = prec
            results["dir_recall_macro"] = rec
            results["dir_f1_macro"] = f1
            # ROC‑AUC – binarize labels; predictions are crisp, so AUC is limited
            y_true_bin = label_binarize(y_true, classes=[-1, 0, 1])
            y_pred_bin = label_binarize(y_pred, classes=[-1, 0, 1])
            try:
                results["dir_roc_auc_macro"] = roc_auc_score(
                    y_true_bin, y_pred_bin, average="macro", multi_class="ovo"
                )
            except ValueError:
                results["dir_roc_auc_macro"] = float("nan")

            results["pnl_dir"]   = self._pnl_dir

        # ---------- range regression ------------------------------------
        if self._range_events:
            df_r = pd.DataFrame(self._range_events)
            # high band
            mse_high = mean_squared_error(df_r["target_high"], df_r["predicted_high"])
            results["range_high_rmse"] = math.sqrt(mse_high)
            results["range_high_mad"] = mean_absolute_error(
                df_r["target_high"], df_r["predicted_high"]
            )
            # low band
            mse_low = mean_squared_error(df_r["target_low"], df_r["predicted_low"])
            results["range_low_rmse"] = math.sqrt(mse_low)
            results["range_low_mad"] = mean_absolute_error(
                df_r["target_low"], df_r["predicted_low"]
            )




        return results

    # ------------------------------------------------------- introspection
    @property
    def priceEventCount(self) -> int:
        return len(self._price_events)

    @property
    def directionEventCount(self) -> int:
        return len(self._dir_events)

    @property
    def rangeEventCount(self) -> int:
        return len(self._range_events)
