import pandas as pd
import numpy as np
import typing
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

class Exchange(Enum):
    NSE = "NSE"
    BSE = "BSE"
    OPRA = "OPRA"

class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"

class OptionType(Enum):
    CE = "C"
    PE = "P"
    STRADDLE = "SE"
    INVALID = "INVALID"

class DurationType(Enum):
    D = "W"
    W = "W"
    M = "M"

class StrikeType(Enum):
    STATIC = "Static"
    DYNAMIC = "Dynamic"

class ProductType(Enum):
    EQ = "EQ"
    OPT = "OPT"
    FUT = "FUT"
    SPOT = "SPOT"

class Index(Enum):
    SPXW = "SPX"

@dataclass
class Ticker:
    exchange: str
    parent_ticker: str
    product_type: ProductType
    duration_type: DurationType
    duration_length: int
    strike_type: StrikeType
    option_type: OptionType
    strike_price: float
    current_strike_price: float
    ticker_name: str
    def __str__(self) -> str:
        return (f"{self.exchange}:{self.parent_ticker}:{self.ticker_name}:"
                f"{self.product_type}:{self.duration_type.value}:"
                f"{self.duration_length}:{self.strike_type}:"
                f"{self.option_type}:{self.strike_price}")

    def print_details(self) -> None:
        print(str(self))

    def get_current_strike_price(self) -> float:
        return self.current_strike_price

    def get_strike_price(self) -> float:
        return self.current_strike_price
    
    def get_ticker_name(self) -> str:
        return self.ticker_name
    
    def get_expiry(self, date: str, trading_day_df: pd.DataFrame = None) -> str:

        if self.duration_length == 0:
            return date
        """Return expiry date based on duration type and length"""
        if trading_day_df is None:
            raise ValueError("trading_day_df must be provided")

        # Ensure DataFrame dates are datetime
        trading_day_df = trading_day_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(trading_day_df["Date"]):
            trading_day_df["Date"] = pd.to_datetime(trading_day_df["Date"], format="%m/%d/%Y")
        trading_day_df = trading_day_df.sort_values("Date").reset_index(drop=True)

        # Parse input date string (YYYY-MM-DD)
        date = pd.to_datetime(date, format="%Y-%m-%d")
        if self.duration_type == DurationType.D:
            # Filter all trading days strictly after input date
            mask = trading_day_df["Date"] > date
            future_days = trading_day_df.loc[mask, "Date"].tolist()

            if len(future_days) < self.duration_length:
                raise ValueError(f"Not enough trading days after {date.date()} for duration_length={self.duration_length}")

            expiry = future_days[self.duration_length - 1]  # nth valid day ahead
            return expiry.strftime("%Y-%m-%d")  # return string in YYYY-MM-DD

        elif self.duration_type == DurationType.W:
            # weekly expiry logic
            pass
        elif self.duration_type == DurationType.M:
            # monthly expiry logic
            pass
        else:
            raise ValueError(f"Unknown DurationType: {self.duration_type}")
    
    def create_and_get_ticker_name(self, date: str) -> str:
        self.ticker_name = f"{self.parent_ticker}{date.replace('-', '')[2:]}{self.option_type.value}{str(self.strike_price).rjust(8, '0')}"
        return self.ticker_name

    def get_file_path(self, data_path: str, date: str, trading_day_df : pd.DataFrame = None, strike_price: float = None) -> str:
        baseFolder = self.parent_ticker
        if self.parent_ticker in Index.__members__:  # check if symbol matches an enum member name
            baseFolder = (Index[self.parent_ticker].value)
        
        base_path: str = f"{data_path}/{date}/{self.exchange}/{baseFolder}"
        
        if self.product_type == ProductType.EQ:
            return f"{base_path}/eq.csv"
        elif self.product_type == ProductType.SPOT:
            return f"{base_path}/spot.csv"
        elif self.product_type == ProductType.FUT:
            return f"{base_path}/fut_{self.duration_type}_{self.duration_length}.csv"
        elif self.product_type == ProductType.OPT:
            base_path: str = f"{data_path}/{baseFolder}/{date.replace('-', '/')}"
            expiry: str = self.get_expiry(date, trading_day_df)
            self.ticker_name = f"{self.parent_ticker}{expiry.replace('-', '')[2:]}{self.option_type.value}{str(self.strike_price).rjust(8, '0')}"
            return f"{base_path}/{self.ticker_name}.csv"
        else:
            return None
        

@dataclass
class Trade:
    ticker: Ticker
    side: Side
    price: float
    quantity: int
    timestamp: str


@dataclass
class Position:
    ticker: str
    tcost: float
    net_quantity: int = 0
    ltp: float = 0.0
    net_value: float = 0.0
    net_pnl: float = 0.0
    ltt: str = 0
    net_transaction_cost:float = 0.0
    net_realised_pnl: float = 0.0
    penalty: float = 0.0

    trades: List[Trade] = field(default_factory=list)
    pnl_list: List[float] = field(default_factory=list)
    quantity_list: List[int] = field(default_factory=list)

    def print_details(self) -> None: 
        if self.net_pnl != 0:
            print(f"[POSITION DETAILS] Ticker:{self.ticker},CurrentPosition={self.net_quantity},NetPnl_without_tc={self.net_pnl:.2f},NetPnl_with_tc={self.net_realised_pnl:.2f},Ltp={self.ltp:.4f},Ltt={self.ltt},Penalty={self.penalty}")

    def update_ltp(self, ltp: float) -> None:
        self.ltp = ltp
        self.net_pnl = self.net_value + (self.net_quantity * ltp) - self.penalty
        self.net_realised_pnl = self.net_pnl - self.net_transaction_cost
        self.pnl_list.append(self.net_pnl)
        self.quantity_list.append(self.net_quantity)

    def on_trade(self, price: float, quantity: int, timestamp: str) -> None:
        self.net_quantity += quantity
        self.net_transaction_cost += (abs(quantity) * price * self.tcost) 
        self.net_value += -(quantity * price)
        self.net_pnl = self.net_value + (self.net_quantity * self.ltp) - self.penalty
        self.net_realised_pnl = self.net_pnl - self.net_transaction_cost
        self.ltt = timestamp
        side: Side = Side.BUY if quantity > 0  else Side.SELL
        self.trades.append(Trade(self.ticker, side, price, quantity, timestamp))

    def add_penalty(self):
        self.penalty = (abs(self.net_quantity) * self.ltp * 0.01)

@dataclass
class PositionManager:
    tcost: float
    position_map: Dict[str, Position] = field(default_factory=dict)
    def add_ticker(self, ticker: str) -> None:
        if ticker not in self.position_map:
            self.position_map[ticker] = Position(ticker, self.tcost)
        else:
            return
    
    def update_ltp(self, ticker: str, ltp: float) -> None:
        if ticker not in self.position_map:
            self.add_ticker(ticker)
        self.position_map[ticker].update_ltp(ltp)
    
    def on_trade(self, ticker: str, price: float, quantity: int, timestamp: int) -> None:
        if ticker not in self.position_map:
            self.add_ticker(ticker)
        self.position_map[ticker].on_trade(price, quantity, timestamp)

    def add_penalty(self, ticker:str):
        self.position_map[ticker].add_penalty()

    def print_details(self) -> None:
        # netpnl : float = 0
        # netrealpnl : float = 0
        # netTC : float = 0
        for ticker in self.position_map.keys():
            self.position_map[ticker].print_details()
        #     netpnl = netpnl + self.position_map[ticker].net_pnl
        #     netrealpnl = netrealpnl + self.position_map[ticker].net_realised_pnl
        #     netTC = netTC + self.position_map[ticker].net_transaction_cost
        # print("MTM:",netpnl,"|Realised_MTM:",netrealpnl,"|Total_TC:",f"{netTC:.2f}")
        

@dataclass
class Data:
    ticker: str
    timestamp: int = 0
    last_quote_timestamp : int = 0
    last_trade_timestamp : int = 0
    bid_price : float = 0.0
    bid_qty : float = 0.0
    ask_price : float = 0.0
    ask_qty : float = 0.0
    ltp : float = 0.0
    ltq : float = 0.0
    total_trade_value : float = 0.0
    total_trade_qty : float = 0.0
    total_trade_valuewholeday : float = 0.0
    total_trade_qtywholeday : float = 0.0
    vvap : float = 0.0
    open : float = 0.0
    high : float = 0.0
    low : float = 0.0
    close : float = 0.0
              
    def update_from_csv_line(self, line: str):
        """Update fields in place, keeping old values when new ones are 0."""
        parts = line.strip().split(",")
        self.timestamp=int(parts[0])
        self.last_quote_timestamp=int(parts[1])
        self.last_trade_timestamp=int(parts[2])
        self.bid_price=float(parts[3])
        self.bid_qty=float(parts[4])
        self.ask_price=float(parts[5])
        self.ask_qty=float(parts[6])
        self.ltp=float(parts[7])
        self.ltq=float(parts[8])
        self.total_trade_value=float(parts[9])
        self.total_trade_qty=float(parts[10])
        self.total_trade_valuewholeday=float(parts[11])
        self.total_trade_qtywholeday=float(parts[12])
        self.vvap=float(parts[13])
        self.open=float(parts[14])
        self.high=float(parts[15])
        self.low=float(parts[16])
        self.close=float(parts[17])

    def clearInterval(self):
        self.vvap=0.0
        self.open=0.0
        self.high=0.0
        self.low=0.0
        self.close=0.0
        self.total_trade_value=0.0
        self.total_trade_qty=0.0
    
    def __str__(self) -> str:
        """Pretty-print all fields in key=value format"""
        items = asdict(self)
        return ", ".join(f"{k}={v}" for k, v in items.items())

    
@dataclass
class BroadcastData:
    ticker: str
    timestamp: int = 0
    open : float = 0.0
    high : float = 0.0
    low : float = 0.0
    close : float = 0.0
    volume : float = 0.0
    volume_count : float = 0.0
              
    def update_from_csv_line(self, line: str):
        """Update fields in place, keeping old values when new ones are 0."""
        parts = line.strip().split(",")
        if float(parts[1]) == 0.0 and float(parts[2]) == 0.0 and float(parts[3]) == 0.0 and float(parts[4]) == 0.0:
            return
        self.timestamp=int(parts[0])
        self.open=float(parts[1])
        self.high=float(parts[2])
        self.low=float(parts[3])
        self.close=float(parts[4])
        self.volume=float(parts[5])
        self.volume_count=float(parts[6])
    
    def __str__(self) -> str:
        """Pretty-print all fields in key=value format"""
        items = asdict(self)
        return ", ".join(f"{k}={v}" for k, v in items.items())
    
    def get_json(self) -> str:
        """Return compact JSON representation of the Data object."""
        data = asdict(self)
        data["event_type"] = "A"
        return json.dumps(data)
        

@dataclass
class BookQuote:
    ticker: str
    timestamp: int = 0
    event_timestamp: int = 0
    bid_price : float = 0.0
    bid_qty : float = 0.0
    ask_price : float = 0.0
    ask_qty : float = 0.0

    def get_json(self) -> str:
        """Return compact JSON representation of the Data object."""
        data = asdict(self)
        data["event_type"] = "Q"
        return json.dumps(data)

@dataclass
class TradeQuote:
    ticker: str
    timestamp: int = 0
    event_timestamp: int = 0
    ltp : float = 0.0
    ltq : float = 0.0

    def get_json(self) -> str:
        """Return compact JSON representation of the Data object."""
        data = asdict(self)
        data["event_type"] = "T"
        return json.dumps(data)
   


@dataclass
class StraddleData:
    call_data: Data
    put_data: Data





