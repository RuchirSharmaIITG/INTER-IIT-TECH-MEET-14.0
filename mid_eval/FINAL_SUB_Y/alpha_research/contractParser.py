from alpha_research.includes import Ticker , ProductType , DurationType, StrikeType, OptionType, Index, BroadcastData
from typing import List, Dict, Any, Union
import pandas as pd
import os

'''

Example Usage of a ticker parser in Python:
    tokens = ["NSE:RELIANCE:EQ", "NSE:RELIANCE:FUT:M:0", "NSE:RELIANCE:OPT:W:0:S:24000:CE", "NSE:RELIANCE:OPT:W:0:D:0:PE", "NSE:RELIANCE:OPT:W:0:S:24000:SE", "NSE:RELIANCE:OPT:W:0:D:0:SE"]
    option_chain = ["NSE:NIFTY:W:0:D:-2:2", "NSE:NIFTY:W:0:S:24000:25000"]

Inputs: 
    1. eod_ohlcv.csv -> date
        a. Ticker, Open, High, Low, Close, Volume, StrikeSize

    2. Folder 

'''

class ContractParser:
    ohlcv_df: pd.DataFrame
    date: str
    broadcast_data_path: str

    def __init__(self, date: str,indices_file_path: str = None, broadcast_data_path: str = None):
        self.date = date.replace("-","")
        if indices_file_path and os.path.exists(indices_file_path):
            self.ohlcv_df = pd.read_csv(indices_file_path)
            self.broadcast_data_path = broadcast_data_path
        else:
            self.ohlcv_df = None

    def parse_single_token(self, token: str) -> Union[Ticker, List[Ticker], None]:
        components: List[str] = token.split(':')

        assert len(components) >= 3

        exchange: str = components[0]
        symbol: str = components[1]
        product_type: str = components[2]

        # if product_type == "EQ":
        #     return Ticker(exchange, symbol, ProductType.EQ, None, None, None, None, None, None, None)
        
        # if product_type == "SPOT":
        #     return Ticker(exchange, symbol, ProductType.SPOT, None, None, None, None, None, None, None)
        
        duration_type: DurationType = DurationType.D if components[3] == "D"  else DurationType.W if components[3] == "W"  else DurationType.M
        duration_length = int(components[4])
        # if product_type == "FUT":
        #     assert len(components) == 5
        #     return Ticker(exchange, symbol, ProductType.FUT, duration_type, duration_length None, None, None, None, None)

        strike_type: StrikeType = StrikeType.STATIC if components[5] == "S"  else StrikeType.DYNAMIC
        option_type: OptionType = OptionType.CE if components[7] == "CE"  else OptionType.PE if components[7] == "PE" else OptionType.STRADDLE if components[7] == "SE" else OptionType.INVALID
        
        if product_type == "OPT" and option_type != OptionType.INVALID:
            assert len(components) == 8
            strike = int(components[6])
            if option_type == OptionType.STRADDLE:
                return [
                    Ticker(exchange, symbol, ProductType.OPT, duration_type, duration_length,
                        strike_type, OptionType.CE, strike, 0, None),
                    Ticker(exchange, symbol, ProductType.OPT, duration_type, duration_length,
                        strike_type, OptionType.PE, strike, 0, None)
                ]
            else:
                return Ticker(exchange, symbol, ProductType.OPT, duration_type, duration_length,
                            strike_type, option_type, strike, 0, None)
        
        print("Product type is not supported")
        return None

    def parse_single_option_chain(self, option_chain: str) -> List[Ticker]:
        components = option_chain.split(':')
        assert len(components) == 8 
        
        exchange: str = components[0]
        symbol: str = components[1]
        duration_type: str = components[2]
        duration_length: int = int(components[3])
        start_strike: int = int(components[5])
        end_strike: int = int(components[6])
        strike_size: int =  int(components[7])
        ticker_list: List[Ticker] = []
        
        for strike in range(start_strike, end_strike + strike_size, strike_size):
            for opt_type in ["CE", "PE"]:
                parsed = self.parse_single_token(f"{exchange}:{symbol}:OPT:{duration_type}:{duration_length}:S:{strike}:{opt_type}")
                if parsed is not None:
                    if isinstance(parsed, list):
                        ticker_list.extend(parsed)
                    else:
                        ticker_list.append(parsed)

        return ticker_list
    
    def parse_single_option_dynamic_chain(self, state_map_broadcast: Dict[str , BroadcastData], option_chain: str) -> List[Ticker]:
        components = option_chain.split(':')
        assert len(components) == 8 
        
        exchange: str = components[0]
        symbol: str = components[1]
        duration_type: str = components[2]
        duration_length: int = int(components[3])
        start_strike: float = float(components[5])
        start_strike_index = int(start_strike)
        end_strike: float = float(components[6])
        end_strike_index = int(end_strike)
        strike_size: float =  float(components[7])
        step = int(strike_size)
        ticker_map: Dict[str ,Ticker] = {}

        #if fixed_type == "D": #asap
        indexToken: str = symbol
        if indexToken in Index.__members__:  # check if symbol matches an enum member name
            indexToken = Index[symbol].value  # map to its value

        if indexToken not in state_map_broadcast:
            return ticker_map

        if state_map_broadcast[indexToken].low == 0.0 or state_map_broadcast[indexToken].high == 0.0:
            return ticker_map

        low_price: int = int(state_map_broadcast[indexToken].low/10) * 10000
        high_price: int = int(state_map_broadcast[indexToken].high/10) * 10000
        start_strike = low_price + (start_strike * step)
        end_strike = high_price + (end_strike * step)
        # print(option_chain, indexToken, low_price , high_price, start_strike , end_strike)
        
        idx = start_strike_index
        for strike in range(int(start_strike), int(end_strike) + step, step):
            for opt_type in ["CE", "PE"]:
                parsed = self.parse_single_token(f"{exchange}:{symbol}:OPT:{duration_type}:{duration_length}:D:{strike}:{opt_type}")
                if parsed is not None: 
                    key = f"{symbol}_{idx}{opt_type}"                  
                    ticker_map[key] = parsed
            idx += 1
        return ticker_map
    
    def parse_tokens(self, token_list: List[str]) -> List[Ticker]:
        ticker_list: List[Ticker] = []
        for token in token_list:
            parsed = self.parse_single_token(token)
            if parsed is None:
                continue
            if isinstance(parsed, list):
                ticker_list.extend(parsed)
            else:
                ticker_list.append(parsed)
        return ticker_list

    def parse_option_chain(self, option_chain_list: List[str]) -> List[Ticker]:
        ticker_list: List[Ticker] = []
        for option_chain in option_chain_list:
            tickers = self.parse_single_option_chain(option_chain)
            if tickers:
                ticker_list.extend(tickers)
        return ticker_list

    def parse_ticker_options(self, token_list: List[str], option_chain_list: List[str]):
        ticker_list: List[Ticker] = []
        ticker_list.extend(self.parse_tokens(token_list))
        ticker_list.extend(self.parse_option_chain(option_chain_list))
        return ticker_list
    
    def parse_ticker_dynamic_options(self, state_map_broadcast: Dict[str , BroadcastData],option_chain_dynamic_list: List[str]) -> Dict[str, Ticker]:
        ticker_dynamic_list: Dict[str, Ticker] = {}
        index_ticker : List[str] = []

        for option_dynamic_chain in option_chain_dynamic_list:
            parsed = self.parse_single_option_dynamic_chain(state_map_broadcast,option_dynamic_chain)
            if parsed is None:
                continue
            if isinstance(parsed, dict):
                ticker_dynamic_list.update(parsed)  # merge dicts
        
        return ticker_dynamic_list

    def parse_spot_tickers(self, option_chain_dynamic_list: List[str]) -> List[str]:
        index_ticker : List[str] = []            
        for option_dynamic_chain in option_chain_dynamic_list:
            components = option_dynamic_chain.split(':')
            assert len(components) == 8 
            symbol: str = components[1]
            if symbol in Index.__members__:  # check if symbol matches an enum member name
                index_ticker.append(Index[symbol].value)  # map to its value
            else:
                index_ticker.append(symbol)
        return index_ticker
        
        

        





    

