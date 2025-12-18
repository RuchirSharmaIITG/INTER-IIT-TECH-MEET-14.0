import pandas as pd
from alpha_research.contractParser import ContractParser as ContractParser
from alpha_research.includes import BroadcastData as BroadcastData, Data as Data, Ticker as Ticker
from _typeshed import Incomplete
from typing import Any, Callable

class DataLoaderIIT:
    data: list[dict]
    data_path: str
    eod_file_path: str
    ticker_list: list[Ticker]
    ticker_dynamic_list: dict[str, Ticker]
    contractParser: ContractParser
    token_list: list[str]
    option_chain_list: list[str]
    option_chain_dynamic_list: list[str]
    date: str
    trading_days_file: str
    ticker_file_path_map: dict[str, str]
    broadcast_ticker_file_path_map: dict[str, str]
    dynamic_ticker_file_path_map: dict[str, str]
    file_iters: Incomplete
    broadcast_file_iters: Incomplete
    dynamic_file_iters: Incomplete
    heap: Incomplete
    broadcast_heap: Incomplete
    dynamic_heap: Incomplete
    option_chain_dynamic_index_to_ticker: dict[str, str]
    broadcast: list[str]
    broadcast_data_path: str
    prev_ts: int
    dynamic_file_iter_index: int
    global_dynamic_option_tickers: list[str]
    trading_days_df: pd.DataFrame
    state_map_broadcast: dict[str, dict[str, Any]]
    def __init__(self, data_path: str, date: str, broadcast: list[str]) -> None: ...
    def load_data(self) -> None: ...
    def parse_timestamp(self, time_str): ...
    def parse_csv_line(self, line: str) -> list: ...
    def convert_row(self, row: dict) -> dict: ...
    def load_files(self) -> None: ...
    curr_ts: int | None
    def run(self, broadcast_callback: Callable[[dict[str, dict[str, Any]], int], None]) -> None: ...
    def close_files(self) -> None: ...
