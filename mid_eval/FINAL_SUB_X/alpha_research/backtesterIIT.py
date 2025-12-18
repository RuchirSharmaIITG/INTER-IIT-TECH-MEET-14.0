import json
from typing import Dict, Callable, List, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from alpha_research.includes import Ticker, Data, BroadcastData, Side, Trade, PositionManager
from alpha_research.dataLoaderIIT import DataLoaderIIT
from alpha_research.resultReport import ResultReport

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

class BacktesterIIT:
    def __init__(self, config_file: str) -> None:
        self.config = self._load_config(config_file)
        self.loaders: List[DataLoaderIIT] = []
        self.start_date = int(self.config["start_date"])
        self.end_date = int(self.config["end_date"])
        for d in range(self.start_date, self.end_date + 1):
            self.loaders.append(
                DataLoaderIIT(
                    data_path=self.config["data_path"],
                    date=d,
                    broadcast=self.config.get("broadcast", []),
                )
            )
        
        self.timerInterval=int(self.config["timer"])
        self.prevTs = 0
        self.state_map: Dict[str, Data] = {}
        self.broadcast_state_map: Dict[str, Dict[str, Any]] = {}
        self.timestamp : int  = 0
        self.csvtimestamp : str
        self.tcost = float(self.config["tcost"]) * 0.0001
        self.position_manager: PositionManager = PositionManager(tcost = self.tcost)
        self.result_report: ResultReport = ResultReport()
        self.position_map: Dict[str , int] = {}
        self.limitPosition: bool = True

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r") as f:
            return json.load(f)
        
    def place_order(self,ticker:str , qty: int , side:Side):

        if self.limitPosition:
            imp_qty = self.position_map.get(ticker, 0) + (qty if side == Side.BUY else -qty)
            if abs(imp_qty) > 1:
                print(f"[ORDER REJECTED] Ticker:{ticker},CurrentPosition:{self.position_map.get(ticker, 0)} [ORDER DETAILS] Qty:{qty},Side:{side.value},timestamp:{self.csvtimestamp}")
                return None
        
        data: List[Any] = self.broadcast_state_map.get(ticker)
        exec_price = None        
        if data:
            exec_price = (
                data['Price'] if data['Price'] != 0.0 else
                None
            )

        if exec_price is None:
            return None
        

        trade = Trade(ticker=ticker,side=side,price=exec_price,quantity=qty,timestamp=self.csvtimestamp)

        if side is Side.BUY:
            self.position_map[ticker] = self.position_map.get(ticker, 0) + qty
        else:
            self.position_map[ticker] = self.position_map.get(ticker, 0) - qty

        self.position_manager.on_trade(
        ticker=ticker,
        price=exec_price,
        quantity=qty if side is Side.BUY else -qty,  # convention: BUY = +qty, SELL = -qty
        timestamp=self.csvtimestamp,
        )
        return trade
   

    # --- default broadcast callback ---
    def default_broadcast_callback(self, state: Dict[str, Dict[str, Any]], ts: str) -> None:
        self.broadcast_state_map = state
        self.csvtimestamp = ts
        for ticker, data in state.items():
            if data['Price'] != 0 and not np.isnan(data['Price']):
                self.position_manager.update_ltp(ticker, data['Price'])
        # print(f"\n[BROADCAST] Timestamp: {ts}")
        # for ticker, data in state.items():
        #     print(f"{ticker}: spot={data.ticker}, open={data.open}, high={data.high}, "
        #           f"low={data.low}, close={data.close}")

    
    def auto_square_off(self):
        sqof: bool = False
        for ticker, pos in self.position_map.items():
            if pos != 0:
                sqof = True
                self.position_manager.add_penalty(ticker)
                side: Side = Side.BUY if pos < 0 else Side.SELL
                self.place_order(ticker, abs(pos), side)      
        self.position_manager.print_details()

    def on_timer(self) -> None:
        """Call this every 1s to print positions"""
        print(f"\n[TIMER] Timestamp={self.csvtimestamp}")
        self.position_manager.print_details()

    def generate_eod_report(self) -> None:
        report = self.result_report.generate_report()
        print(f"EOD_CUMMULATIVE_SUMMARY : from {self.start_date} to {self.end_date}")
        dReport = pd.DataFrame(report).T
        dReport.index.name = "Ticker"
        print(dReport.to_string())
        print(" ")
        print(f"EOD_DAYWISE_SUMMARY : from {self.start_date} to {self.end_date}")
        df = self.result_report.export_equity_curve()
        for k , d in df.items():
            print(k)
            print(d)
        
    def run(
    self,
    broadcast_callback: Callable[[Dict[str, Dict[str, Any]], str], None] = None,
    timer_callback: Callable[[Dict[str, Data], str], None] = None,
    ) -> None:
        for loader, d in zip(self.loaders,  range(self.start_date, self.end_date + 1)):
            self.prevTs = 0
            self.state_map = {}
            self.broadcast_state_map = {}
            self.timestamp = 0
            self.csvtimestamp = ""
            self.position_map = {}
            self.position_manager = PositionManager(tcost = self.tcost)
            loader.load_data()
                

            def combined_broadcast_callback(state: Dict[str, Dict[str, Any]], ts: int):
                latest_ts = (max(datetime.strptime(v['Time'], "%H:%M:%S") for v in state.values() if 'Time' in v)).strftime("%H:%M:%S")
                self.default_broadcast_callback(state, latest_ts)
                if broadcast_callback is not None:
                    broadcast_callback(state, latest_ts)
                if self.prevTs == 0:
                    self.prevTs = ts
                elif ts - self.prevTs >= self.timerInterval:
                    self.prevTs = ts
                    self.on_timer()
                    if timer_callback is not None:
                        timer_callback(latest_ts)
            # call loader with both
            loader.run(combined_broadcast_callback)
            loader.close_files()
            self.auto_square_off()
            self.result_report.update(self.position_manager, d)
        self.generate_eod_report()

