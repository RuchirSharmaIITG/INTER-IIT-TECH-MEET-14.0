import pandas as pd
import os
import csv
import typing
from typing import List, Dict, Any , Callable
import heapq
from alpha_research.includes import Ticker , Data, BroadcastData
from alpha_research.contractParser import ContractParser
from math import inf
'''
Input required by the data loader class:
    
    1. Folder containing the data - it should be in the following format:
       DATA_PATH/dateDDMMYYY/EXCHANGE/TICKER/(EQ or FUT_DURATIONTYPE_DURATIONLENGTH or SPOT or OPT_DURATIONTYPE_DURATIONLENGTH_STRIKE_CEPE).csv
       e.g. /home/100ms_data/04052024/NSE/RELIANCE/eq.csv, /home/100ms_data/04052024/NSE/RELIANCE/fut_m_0.csv / /home/100ms_data/04052024/NSE/RELIANCE/opt_m_0_2000_ce.csv
    
    2. /DATA_PATH/dateDDMMYYYY/EXCHANGE/eod_ohlcv.csv -> date
        a. Ticker, Open, High, Low, Close, Volume, StrikeSize

'''


class DataLoaderIIT:
    data: List[dict] = []
    data_path: str
    eod_file_path: str
    ticker_list: List[Ticker] = []
    ticker_dynamic_list: Dict[str, Ticker] = {}
    contractParser: ContractParser
    token_list: List[str]
    option_chain_list: List[str]
    option_chain_dynamic_list: List[str]
    date: str
    trading_days_file: str
    ticker_file_path_map: Dict[str , str] = {}
    broadcast_ticker_file_path_map: Dict[str , str] = {}
    dynamic_ticker_file_path_map: Dict[str , str] = {}
    # state_map: Dict[str , Data] = {}
    # state_map_broadcast: Dict[str , BroadcastData] = {}
    file_iters = []
    broadcast_file_iters = []
    dynamic_file_iters = []
    heap = []
    broadcast_heap = []
    dynamic_heap = []
    option_chain_dynamic_index_to_ticker: Dict[str , str] = {}
    broadcast: List[str] = []
    broadcast_data_path : str
    prev_ts: int
    dynamic_file_iter_index: int = 0
    global_dynamic_option_tickers: List[str] = []
    trading_days_df: pd.DataFrame

    def __init__(self, data_path: str, date: str, broadcast: List[str]) -> None:
        self.data_path = data_path
        self.date = date
        self.broadcast = broadcast
        self.state_map_broadcast: Dict[str, Dict[str, Any]] = {}
        self.broadcast_ticker_file_path_map: Dict[str , str] = {}
        if not os.path.exists(self.data_path):
            print(f"The path {self.data_path} does not exist - Exiting!")
            exit(0)

    def load_data(self) -> None:
        for spot_ticker in self.broadcast:
            ticker_path = f"{self.data_path}/{spot_ticker}/day{self.date}_signals.csv"
            print(ticker_path)
            if os.path.exists(ticker_path):
                self.broadcast_ticker_file_path_map[spot_ticker] = ticker_path
                print(f"Replaying {ticker_path} ...")
            else:
                print(f"The path {ticker_path} does not exist")
        self.load_files()

    def parse_timestamp(self,time_str):
        try:
            h, m, s = map(int, time_str.strip().split(":"))
            return h * 3600 + m * 60 + s
        except Exception:
            return 0

    def parse_csv_line(self,line: str) -> list:
        parts = [x.strip() for x in line.split(",")]
        data = []

        for v in parts:
            if v == "" or v is None:
                data.append(0.0)
            else:
                try:
                    data.append(float(v))
                except ValueError:
                    data.append(v)

        return data
    
    def convert_row(self,row: dict) -> dict:
        converted = {}
        for k, v in row.items():
            if v == "" or v is None:
                converted[k] = v
            else:
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
        return converted

    def load_files(self):
        for idx, (ticker, path) in enumerate(self.broadcast_ticker_file_path_map.items()):
            print(ticker , path)
            f = open(path, "r")
            reader = csv.DictReader(f)
            self.broadcast_file_iters.append((f, ticker, reader))
            try:
                line = self.convert_row(next(reader)) 
                if not line:
                    continue
                self.state_map_broadcast[ticker] = line
                heapq.heappush(self.broadcast_heap, (self.parse_timestamp(line['Time']), idx))  
            except (StopIteration, KeyError, ValueError) as e:
                print(f"[WARN] Skipping {ticker} in {path}: {e}")
                continue
        
    
    def run(self,broadcast_callback: Callable[[Dict[str, Dict[str, Any]], int], None]) -> None:
        self.curr_ts: int | None = None
        while self.broadcast_heap:
            ts_bcast  = self.broadcast_heap[0][0] if self.broadcast_heap else None
            # pick earliest timestamp
            next_ts = ts_bcast if ts_bcast >= 0 else inf
           
            if next_ts is inf:
                break

            # 🔑 now this is the "current" timestamp being processed
            self.curr_ts = next_ts
            # --- broadcast side ---
            if ts_bcast is not None and ts_bcast == self.curr_ts:
                broadcast_callback(dict(self.state_map_broadcast), self.curr_ts)
                while self.broadcast_heap and self.broadcast_heap[0][0] == self.curr_ts:
                    _, idx = heapq.heappop(self.broadcast_heap)
                    f, ticker, reader = self.broadcast_file_iters[idx]
                    try:
                        line = self.convert_row(next(reader)) 
                        if line:
                            self.state_map_broadcast[ticker] = line
                            heapq.heappush(self.broadcast_heap, (self.parse_timestamp(line['Time']), idx))
                    except (StopIteration, KeyError, ValueError) as e:
                        continue
                        

    def close_files(self):
        for f, ticker, reader in self.broadcast_file_iters:
            try: f.close()
            except: pass
        self.broadcast_file_iters.clear()
        


        










    


