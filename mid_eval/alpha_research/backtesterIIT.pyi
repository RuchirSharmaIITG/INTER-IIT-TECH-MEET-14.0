from alpha_research.dataLoaderIIT import DataLoaderIIT as DataLoaderIIT
from alpha_research.includes import BroadcastData as BroadcastData, Data as Data, PositionManager as PositionManager, Side as Side, Ticker as Ticker, Trade as Trade
from alpha_research.resultReport import ResultReport as ResultReport
from _typeshed import Incomplete
from datetime import datetime as datetime, timedelta as timedelta
from typing import Any, Callable

class BacktesterIIT:
    config: Incomplete
    loaders: list[DataLoaderIIT]
    start_date: Incomplete
    end_date: Incomplete
    timerInterval: Incomplete
    prevTs: int
    state_map: dict[str, Data]
    broadcast_state_map: dict[str, dict[str, Any]]
    timestamp: int
    tcost: Incomplete
    position_manager: PositionManager
    result_report: ResultReport
    position_map: dict[str, int]
    limitPosition: bool
    def __init__(self, config_file: str) -> None: ...
    def place_order(self, ticker: str, qty: int, side: Side): ...
    def default_broadcast_callback(self, state: dict[str, dict[str, Any]], ts: int) -> None: ...
    def auto_square_off(self) -> None: ...
    def on_timer(self) -> None: ...
    def generate_eod_report(self) -> None: ...
    def run(self, broadcast_callback: Callable[[dict[str, dict[str, Any]], int], None] = None, timer_callback: Callable[[dict[str, Data], int], None] = None) -> None: ...
