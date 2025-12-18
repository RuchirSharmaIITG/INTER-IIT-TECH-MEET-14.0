import pandas as pd
from alpha_research.includes import PositionManager as PositionManager, Side as Side
from _typeshed import Incomplete

class ResultReport:
    initial_capital: int
    equity_curve: Incomplete
    original_curve: Incomplete
    timestamps: Incomplete
    returns: Incomplete
    pos_trades: Incomplete
    neg_trades: Incomplete
    total_trades: Incomplete
    def __init__(self) -> None: ...
    def update(self, pm: PositionManager, ts) -> None: ...
    def generate_report(self) -> dict[str, dict[str, float]]: ...
    def export_equity_curve(self) -> dict[str, pd.DataFrame]: ...
