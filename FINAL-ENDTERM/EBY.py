import os
import shutil
import sys
import glob
import re
import torch
import warnings
import numpy as np
import pandas as pd
import gymnasium as gym
from pathlib import Path
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
from datetime import datetime
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
import json
try:
    from alpha_research.backtesterIIT import BacktesterIIT
    from alpha_research.includes import Side
except ImportError:
    print("Error: alpha_research module not found")
    print("Please ensure alpha_research.py is in your Python path")
matplotlib.use('Agg')

warnings.filterwarnings("ignore")

PARAMS = {
    'WARMUP_MINUTES': 30,
    'TRANSACTION_COST': 0.0002,
    'WINDOW_SIZE': 10,
    'STOP_LOSS_TR': -0.0005,
    'TRAIL_PCT_TR': 0.0005,
    'STOP_LOSS_TE': -0.0004,
    'TRAIL_PCT_TE': 0.0004,
    'NUM_PROCS' : 8,
    'INDICATOR_LOOKBACKS': [3, 5, 10, 20],
    'CHOP_PERIOD': 14,
    'KAMA_PERIOD': 20,
    'KAMA_FAST': 2,
    'KAMA_SLOW': 20,
    'AROON_PERIOD': 20,
    'HA_ITERATIONS': 10,
    'RIBBON_PERIODS': [2, 3, 5, 8, 12, 15, 18],
    'RIBBON_REF_PERIOD': 100,
    'OPPORTUNITY_WINDOW': 5,
    'OPPORTUNITY_THRESHOLD': 0.0005,
    'TRAIN_RATIO': 0.5,
    'SEED': 33,
    'EPISODES': 1_000_000,
    'CANDLE_FREQUENCY': '2min',
    'SOURCE_FOLDER' : 'EBY'
}

REWARD_PARAMS = {
    'STOP_LOSS_PENALTY': -100,
    'TRAILING_STOP_PENALTY': -10,
    'EOD_CLOSE_PENALTY': -10,
    'TRADE_ENTRY_PENALTY': -5,
    'LOSS_MULTIPLIER': 4.0,
    'MISSED_OPPORTUNITY_PENALTY': -2.0,
    'WAIT_BONUS': 0.1,
    'SCALE': 0.01,
}

def resample_to_candles(source_folder: str, target_folder: str, frequency: str = '2min'):
    os.makedirs(target_folder, exist_ok=True)
    file_paths = sorted(Path(source_folder).glob("*.csv"))
    
    agg_dict = {'Price': ['first', 'max', 'min', 'last']}
    final_cols = ['Open', 'High', 'Low', 'Close']
    
    existing_files = {f.name for f in Path(target_folder).glob("*.csv")}
    files_to_process = [f for f in file_paths if f.name not in existing_files]
    
    if not files_to_process:
        print(f"All files already processed in {target_folder}")
        return
    
    desc = f"Resampling {source_folder} to {frequency}"
    for file_path in tqdm(files_to_process, desc=desc):
        df = pd.read_csv(file_path)
        df = df[['Time', 'Price']]
        
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.set_index('Time').sort_index()
        
        candles = df.resample(frequency).agg(agg_dict)
        candles.columns = final_cols
        candles = candles.dropna(subset=['Open'])
        candles = candles.reset_index()
        
        out_path = Path(target_folder) / file_path.name
        candles.to_csv(out_path, index=False)

def load_ohlc_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=['Time'])
    df.rename(columns={'Time': 'Timestamp'}, inplace=True)
    df.set_index('Timestamp', inplace=True)
    return df[['Open', 'High', 'Low', 'Close']].copy()

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = df.copy()
    ha['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    ha_opens = [df['Open'].iloc[0]]
    ha_closes = ha['Close'].values
    for i in range(len(df) - 1):
        ha_opens.append((ha_opens[i] + ha_closes[i]) / 2)
    ha['Open'] = ha_opens
    ha['High'] = df[['High']].join(ha[['Open', 'Close']]).max(axis=1)
    ha['Low'] = df[['Low']].join(ha[['Open', 'Close']]).min(axis=1)
    return ha

def apply_ha_iterations(df: pd.DataFrame, iterations: int) -> pd.DataFrame:
    result = df.copy()
    for _ in range(iterations):
        result = calculate_heikin_ashi(result)
    return result

def ema(price: pd.Series, window: int) -> pd.Series:
    return price.ewm(span=window, adjust=False).mean()

def rsi(price: pd.Series, window: int) -> pd.Series:
    delta = price.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def cci(price: pd.Series, window: int) -> pd.Series:
    tp = price
    sma_tp = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad + 1e-9)

def cmo(price: pd.Series, window: int) -> pd.Series:
    delta = price.diff()
    gain = delta.where(delta > 0, 0).rolling(window).sum()
    loss = -delta.where(delta < 0, 0).rolling(window).sum()
    return 100 * (gain - loss) / (gain + loss + 1e-9)

def atr_simple(price: pd.Series, window: int) -> pd.Series:
    high_low = price.rolling(2).max() - price.rolling(2).min()
    return high_low.rolling(window=window).mean()

def standard_deviation(price: pd.Series, window: int) -> pd.Series:
    return price.rolling(window=window).std()

def calculate_chop(df: pd.DataFrame, period: int = 14) -> pd.Series:
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['TR'] = df[['High', 'Low']].apply(
        lambda x: max(x['High'] - x['Low'], 
                     abs(x['High'] - df.loc[x.name, 'prev_close']) if pd.notna(df.loc[x.name, 'prev_close']) else 0,
                     abs(x['Low'] - df.loc[x.name, 'prev_close']) if pd.notna(df.loc[x.name, 'prev_close']) else 0),
        axis=1
    )
    sum_tr = df['TR'].rolling(window=period).sum()
    max_high = df['High'].rolling(window=period).max()
    min_low = df['Low'].rolling(window=period).min()
    range_hl = max_high - min_low
    chop = 100 * np.log10(sum_tr / (range_hl + 1e-9)) / np.log10(period)
    return chop

def calculate_kama(series: pd.Series, period: int = 20, fast: int = 2, slow: int = 20) -> Tuple[pd.Series, pd.Series]:
    change = abs(series - series.shift(period))
    volatility = abs(series - series.shift(1)).rolling(window=period).sum()
    er = change / (volatility + 1e-9)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(series)
    kama[:] = np.nan
    start_idx = period
    if start_idx > 0 and start_idx <= len(series):
        kama[start_idx-1] = series.iloc[start_idx-1]
    price_values = series.values
    sc_values = sc.values
    for i in range(start_idx, len(series)):
        if np.isnan(sc_values[i]):
            kama[i] = price_values[i]
        else:
            kama[i] = kama[i-1] + sc_values[i] * (price_values[i] - kama[i-1])
    return pd.Series(kama, index=series.index), er

def calculate_aroon(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    arg_max = df['High'].rolling(window=window).apply(lambda x: x.argmax(), raw=True)
    days_since_high = (window - 1) - arg_max
    arg_min = df['Low'].rolling(window=window).apply(lambda x: x.argmin(), raw=True)
    days_since_low = (window - 1) - arg_min
    aroon_up = ((window - days_since_high) / window) * 100
    aroon_down = ((window - days_since_low) / window) * 100
    return aroon_up, aroon_down

def calculate_johnny_ribbon(price: pd.Series, ma_periods: List[int], ref_period: int = 100) -> Tuple[dict, pd.Series]:
    ref_ma = ema(price, ref_period)
    ribbon_data = {}
    for period in ma_periods:
        ma = ema(price, period)
        diff = ma.diff()
        cond_lime = (diff >= 0) & (ma > ref_ma)
        cond_maroon = (diff < 0) & (ma > ref_ma)
        cond_rubi = (diff <= 0) & (ma < ref_ma)
        cond_green = (diff >= 0) & (ma < ref_ma)
        regime = np.select([cond_lime, cond_maroon, cond_rubi, cond_green], [1, 2, 3, 4], default=0)
        ribbon_data[period] = {'ma': ma, 'regime': regime}
    return ribbon_data, ref_ma

def add_time_features(index: pd.DatetimeIndex) -> dict:
    t = pd.to_datetime(index)
    return {
        'H_sin': np.sin(2 * np.pi * t.hour / 24),
        'H_cos': np.cos(2 * np.pi * t.hour / 24),
        'M_sin': np.sin(2 * np.pi * t.minute / 60),
        'M_cos': np.cos(2 * np.pi * t.minute / 60)
    }

def calculate_all_indicators(df_120s: pd.DataFrame) -> pd.DataFrame:
    indicators = pd.DataFrame(index=df_120s.index)
    
    prev_open = df_120s['Open'].shift(1)
    prev_high = df_120s['High'].shift(1)
    prev_low = df_120s['Low'].shift(1)
    prev_close = df_120s['Close'].shift(1)
    
    indicators['candle_open'] = np.log(prev_open / prev_open.shift(1))
    indicators['candle_high'] = np.log(prev_high / prev_high.shift(1))
    indicators['candle_low'] = np.log(prev_low / prev_low.shift(1))
    indicators['candle_close'] = np.log(prev_close / prev_close.shift(1))
    
    for period in PARAMS['INDICATOR_LOOKBACKS']:
        rsi_vals = rsi(prev_close, period)
        cci_vals = cci(prev_close, period)
        cmo_vals = cmo(prev_close, period)
        atr_vals = atr_simple(prev_close, period)
        std_vals = standard_deviation(prev_close, period)
        
        indicators[f'rsi_{period}'] = rsi_vals
        indicators[f'cci_{period}'] = cci_vals
        indicators[f'cmo_{period}'] = cmo_vals
        indicators[f'atr_{period}'] = atr_vals / (prev_close + 1e-9)
        indicators[f'std_{period}'] = std_vals / (prev_close + 1e-9)
    
    df_prev = pd.DataFrame({
        'Open': prev_open,
        'High': prev_high,
        'Low': prev_low,
        'Close': prev_close
    })
    
    chop_vals = calculate_chop(df_prev, PARAMS['CHOP_PERIOD'])
    indicators[f'chop_{PARAMS["CHOP_PERIOD"]}'] = chop_vals
    indicators['chop_binary'] = (chop_vals < 43.2).astype(int)
    
    kama_val, er_val = calculate_kama(prev_close, PARAMS['KAMA_PERIOD'], 
                                      PARAMS['KAMA_FAST'], PARAMS['KAMA_SLOW'])
    indicators[f'kama_{PARAMS["KAMA_PERIOD"]}'] = kama_val
    indicators[f'er_{PARAMS["KAMA_PERIOD"]}'] = er_val
    indicators['er_binary'] = (indicators[f'er_{PARAMS["KAMA_PERIOD"]}'] > 0.3).astype(int)
    
    aroon_up, aroon_down = calculate_aroon(df_prev, PARAMS['AROON_PERIOD'])
    indicators[f'aroon_up_{PARAMS["AROON_PERIOD"]}'] = aroon_up
    indicators[f'aroon_down_{PARAMS["AROON_PERIOD"]}'] = aroon_down
    
    df_ha = apply_ha_iterations(df_prev, PARAMS['HA_ITERATIONS'])
    indicators['ha_trend'] = (df_ha['Close'] >= df_ha['Open']).astype(int)
    
    indicators['ha_candle_width'] = ((df_ha['Close'] - df_ha['Open']) / 
                                     (df_ha['Open'] + 1e-9))
    indicators['ha_body_size'] = (abs(df_ha['Close'] - df_ha['Open']) / 
                                  (df_ha['Open'] + 1e-9))
    indicators['ha_upper_wick'] = ((df_ha['High'] - df_ha[['Open', 'Close']].max(axis=1)) / 
                                   (df_ha['Open'] + 1e-9))
    indicators['ha_lower_wick'] = ((df_ha[['Open', 'Close']].min(axis=1) - df_ha['Low']) / 
                                   (df_ha['Open'] + 1e-9))
    
    indicators['ha_open'] = np.log(df_ha['Open'] / df_ha['Open'].shift(1))
    indicators['ha_high'] = np.log(df_ha['High'] / df_ha['High'].shift(1))
    indicators['ha_low'] = np.log(df_ha['Low'] / df_ha['Low'].shift(1))
    indicators['ha_close'] = np.log(df_ha['Close'] / df_ha['Close'].shift(1))
    
    ribbon_data, ref_ma = calculate_johnny_ribbon(df_ha['Close'], 
                                                   PARAMS['RIBBON_PERIODS'], 
                                                   PARAMS['RIBBON_REF_PERIOD'])
    for period, data in ribbon_data.items():
        indicators[f'ribbon_ma_{period}'] = np.log(data['ma'] / data['ma'].shift(1))
        indicators[f'ribbon_regime_{period}'] = pd.Series(data['regime'], 
                                                           index=data['ma'].index)
    indicators['ribbon_ref_ma_100'] = np.log(ref_ma / ref_ma.shift(1))
    
    time_features = add_time_features(df_120s.index)
    for key, val in time_features.items():
        indicators[key] = val
    
    indicators['_close'] = df_120s['Close']
    
    return indicators.fillna(method='ffill').fillna(0)

def is_lookback_feature(feature_name: str) -> bool:
    lookback_prefixes = ['kama_', 'er_', 'aroon_', 'ribbon_ma_', 'ribbon_regime_', 'ha_', 'candle_']
    return any(feature_name.startswith(prefix) for prefix in lookback_prefixes)

def process_single_file(file_path: str) -> pd.DataFrame:
    try:
        df_120s = load_ohlc_data(file_path)
        indicators = calculate_all_indicators(df_120s)
        warmup_cutoff = indicators.index[0] + pd.Timedelta(minutes=PARAMS['WARMUP_MINUTES'])
        return indicators[indicators.index > warmup_cutoff].reset_index(drop=True)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def precompute_all_data(file_pairs: List[str]) -> List[pd.DataFrame]:
    print(f"\n{'='*80}")
    print(f"PRE-COMPUTING INDICATORS FOR {len(file_pairs)} DAYS")
    print(f"{'='*80}")
    
    results = []
    for fp in tqdm(file_pairs, desc="Processing files"):
        r = process_single_file(fp)
        if r is not None:
            results.append(r)
    
    print(f"\nSuccessfully processed {len(results)}/{len(file_pairs)} days")
    print(f"{'='*80}\n")
    return results

def save_feature_info(feature_names: List[str], lookback_features: List[str], 
                     scalar_features: List[str], state_size: int, window_size: int, 
                     ticker: str, output_file: str = "feature_info.txt"):
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"STATE SPACE FEATURE INFORMATION - {ticker}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total State Space Size: {state_size}\n")
        f.write(f"Window Size (Lookback): {window_size}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"LOOKBACK FEATURES (with {window_size}-step history)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Count: {len(lookback_features)}\n")
        f.write(f"Total dimensions: {len(lookback_features) * window_size}\n\n")
        
        for i, feat in enumerate(lookback_features, 1):
            f.write(f"  {i:2d}. {feat:30s} --> {window_size} timesteps\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("SCALAR FEATURES (current timestep only)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Count: {len(scalar_features)}\n")
        f.write(f"Total dimensions: {len(scalar_features)}\n\n")
        
        for i, feat in enumerate(scalar_features, 1):
            f.write(f"  {i:2d}. {feat}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("AGENT STATE FEATURES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Count: 2\n")
        f.write(f"Total dimensions: 2\n\n")
        f.write(f"  1. position (0=flat, 1=long, -1=short)\n")
        f.write(f"  2. mark-to-market PnL (unrealized return)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nFeature information saved to {output_file}")

def linear_schedule(initial_value: float, min_lr: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        current_lr = initial_value * progress_remaining
        return max(current_lr, min_lr)
    return func

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.n_envs = PARAMS['NUM_PROCS']
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.explained_variances = []
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", unit="steps")
        
    def _on_step(self):
        if self.pbar:
            self.pbar.update(self.n_envs)
        return True
    
    def _on_rollout_end(self):
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            policy_loss = self.model.logger.name_to_value.get('train/policy_loss', None)
            value_loss = self.model.logger.name_to_value.get('train/value_loss', None)
            entropy_loss = self.model.logger.name_to_value.get('train/entropy_loss', None)
            explained_var = self.model.logger.name_to_value.get('train/explained_variance', None)
            
            if policy_loss is not None:
                self.policy_losses.append(policy_loss)
            if value_loss is not None:
                self.value_losses.append(value_loss)
            if entropy_loss is not None:
                self.entropy_losses.append(entropy_loss)
            if explained_var is not None:
                self.explained_variances.append(explained_var)
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

def get_exit_params(mode: str = 'train') -> dict:
    if mode.lower() == 'train':
        return {
            'STOP_LOSS': PARAMS['STOP_LOSS_TR'],
            'TRAIL_PCT': PARAMS['TRAIL_PCT_TR'],
        }
    elif mode.lower() == 'test':
        return {
            'STOP_LOSS': PARAMS['STOP_LOSS_TE'],
            'TRAIL_PCT': PARAMS['TRAIL_PCT_TE'],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'test'")

class IntradayTradingEnv(gym.Env):
    metadata = {"render_modes": []}
    
    def __init__(self, processed_data_list: List[pd.DataFrame], mode: str = 'train'):
        super().__init__()
        
        self.all_days = processed_data_list
        self.mode = mode.lower()
        self.tc = PARAMS['TRANSACTION_COST']
        self.window = PARAMS['WINDOW_SIZE']
        self.exit_params = get_exit_params(self.mode)
        
        sample_df = self.all_days[0]
        self.feature_names = [f for f in sample_df.columns if f != '_close']
        self.lookback_features = [f for f in self.feature_names if is_lookback_feature(f)]
        self.scalar_features = [f for f in self.feature_names if f not in self.lookback_features]
        
        num_lookback_features = len(self.lookback_features) * self.window
        num_scalar_features = len(self.scalar_features)
        num_agent_features = 2
        
        self.state_size = num_lookback_features + num_scalar_features + num_agent_features
        
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.state_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.n_steps = 0
        self.indicators_df = None
        self.trade_history = []
        self.trade_count = 0
        self.daily_pnl_abs = 0.0
        self.highest_price = None
        self.lowest_price = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.trade_history = []
        self.trade_count = 0
        self.daily_pnl_abs = 0.0
        
        if hasattr(self, "np_random") and self.np_random is not None:
            random_day_idx = self.np_random.integers(len(self.all_days))
        else:
            random_day_idx = np.random.randint(len(self.all_days))
        
        self.indicators_df = self.all_days[random_day_idx]
        
        self.n_steps = len(self.indicators_df)
        self.position = 0
        self.entry_price = 0
        self.highest_price = None
        self.lowest_price = None
        self.current_step = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        start = max(0, self.current_step - self.window + 1)
        frames = self.indicators_df.iloc[start:self.current_step + 1]
        
        obs_list = []
        
        for feat in self.lookback_features:
            seq = frames[feat].values[-self.window:]
            if len(seq) < self.window:
                seq = np.pad(seq, (self.window - len(seq), 0), constant_values=0)
            obs_list.append(seq)
        
        for feat in self.scalar_features:
            obs_list.append([self.indicators_df.loc[self.current_step, feat]])
        
        obs_list.append([self.position])
        
        mtm = 0
        if self.position != 0 and self.entry_price > 0:
            current_price = self.indicators_df.loc[self.current_step, '_close']
            mtm = (current_price - self.entry_price) / (self.entry_price + 1e-9)
            if self.position == -1:
                mtm = -mtm
        obs_list.append([mtm])
        
        obs = np.concatenate([np.array(x).flatten() for x in obs_list])
        return obs.astype(np.float32)
    
    def _calculate_reward(self, action: int, prev_pos: int, prev_price: float, new_price: float) -> float:
        SCALE = REWARD_PARAMS['SCALE']
        reward = 0.0
        
        if prev_pos != 0 and self.position == 0:
            exit_price = prev_price
            
            if prev_pos == 1:
                ret = (exit_price - self.entry_price) / (self.entry_price + 1e-9)
            else:
                ret = (self.entry_price - exit_price) / (self.entry_price + 1e-9)
            
            pnl_score = ret * 10000
            
            if pnl_score > 0:
                reward += pnl_score * SCALE
            else:
                reward += pnl_score * REWARD_PARAMS['LOSS_MULTIPLIER'] * SCALE
        
        trailing_stop_hit = False
        
        if prev_pos != 0:
            if prev_pos == 1:
                self.highest_price = new_price if self.highest_price is None else max(self.highest_price, new_price)
                trailing_level = self.highest_price * (1 - self.exit_params['TRAIL_PCT'])
                if new_price <= trailing_level:
                    trailing_stop_hit = True
            elif prev_pos == -1:
                self.lowest_price = new_price if self.lowest_price is None else min(self.lowest_price, new_price)
                trailing_level = self.lowest_price * (1 + self.exit_params['TRAIL_PCT'])
                if new_price >= trailing_level:
                    trailing_stop_hit = True
            
            if trailing_stop_hit:
                if prev_pos == 1:
                    ret = (new_price - self.entry_price) / (self.entry_price + 1e-9)
                else:
                    ret = (self.entry_price - new_price) / (self.entry_price + 1e-9)
                
                pnl_score = ret * 10000
                reward += pnl_score * SCALE
                reward += REWARD_PARAMS["TRAILING_STOP_PENALTY"] * SCALE
                self.position = 0
                self.highest_price = None
                self.lowest_price = None
        
        if self.position != 0:
            curr_ret = (new_price - self.entry_price) / (self.entry_price + 1e-9)
            if self.position == -1:
                curr_ret = -curr_ret
            
            if curr_ret <= self.exit_params['STOP_LOSS']:
                pnl_score = curr_ret * 10000 * REWARD_PARAMS['LOSS_MULTIPLIER']
                reward += pnl_score * SCALE
                reward += REWARD_PARAMS['STOP_LOSS_PENALTY'] * SCALE
                self.position = 0
        
        if self.position == 0:
            lookback = PARAMS.get('OPPORTUNITY_WINDOW', 5)
            threshold = PARAMS.get('OPPORTUNITY_THRESHOLD', 0.0005)
            
            if self.current_step >= lookback:
                past_price = self.indicators_df.loc[self.current_step - lookback, '_close']
                move_pct = abs(new_price - past_price) / (past_price + 1e-9)
                
                if move_pct > threshold:
                    penalty = REWARD_PARAMS.get('MISSED_OPPORTUNITY_PENALTY', -2.0)
                    reward += penalty * SCALE
                else:
                    bonus = REWARD_PARAMS.get('WAIT_BONUS', 0.1)
                    reward += bonus * SCALE
        
        return reward
    
    def step(self, action: int):
        prev_price = self.indicators_df.loc[self.current_step, '_close']
        prev_pos = self.position
        
        if action == 1:
            if self.position == -1:
                pnl_abs = (prev_price - self.entry_price)
                self.daily_pnl_abs += pnl_abs - self.tc * prev_price
                self.position = 0
            elif self.position == 0:
                self.position = 1
                self.entry_price = prev_price
                self.daily_pnl_abs -= self.tc * prev_price
        
        elif action == 2:
            if self.position == 1:
                pnl_abs = (prev_price - self.entry_price)
                self.daily_pnl_abs += pnl_abs - self.tc * prev_price
                self.position = 0
            elif self.position == 0:
                self.position = -1
                self.entry_price = prev_price
                self.daily_pnl_abs -= self.tc * prev_price
        
        step_penalty = 0.0
        if prev_pos == 0 and self.position != 0:
            step_penalty = REWARD_PARAMS['TRADE_ENTRY_PENALTY'] * REWARD_PARAMS['SCALE']
        
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        new_price = self.indicators_df.loc[self.current_step, '_close']
        
        if self.position == 1:
            self.daily_pnl_abs += (new_price - prev_price)
        elif self.position == -1:
            self.daily_pnl_abs += (prev_price - new_price)
        
        reward = self._calculate_reward(action, prev_pos, prev_price, new_price)
        reward += step_penalty
        
        if terminated and self.position != 0:
            reward += REWARD_PARAMS['EOD_CLOSE_PENALTY'] * REWARD_PARAMS['SCALE']
            if self.position == 1:
                self.daily_pnl_abs += (new_price - self.entry_price)
            elif self.position == -1:
                self.daily_pnl_abs += (self.entry_price - new_price)
            self.daily_pnl_abs -= self.tc * new_price
            self.position = 0
        
        opened = (prev_pos == 0 and self.position != 0)
        closed = (prev_pos != 0 and self.position == 0)
        
        if opened:
            if self.position == 1:
                self.highest_price = prev_price
                self.lowest_price = None
            elif self.position == -1:
                self.lowest_price = prev_price
                self.highest_price = None
        
        if closed:
            self.highest_price = None
            self.lowest_price = None
        
        if opened or closed:
            self.trade_count += 1
            trade_type = (
                "OPEN_LONG" if prev_pos == 0 and self.position == 1 else
                "OPEN_SHORT" if prev_pos == 0 and self.position == -1 else
                "CLOSE_LONG" if prev_pos == 1 and self.position == 0 else
                "CLOSE_SHORT"
            )
            
            self.trade_history.append({
                "step": self.current_step - 1,
                "trade_type": trade_type,
                "prev_position": prev_pos,
                "new_position": self.position,
                "entry_price": float(self.entry_price),
                "exit_price": float(new_price if closed else self.entry_price),
                "realized_pnl": float(self.daily_pnl_abs),
                "reward": float(reward),
            })
        
        return self._get_obs(), float(reward), terminated, False, {}
    

def load_data(folder_path: str, ticker: str, train_ratio: float = 0.5) -> Tuple[List[str], List[str]]:
    files = sorted(glob.glob(f"{folder_path}/*.csv"))
    file_pairs = [str(Path(f)) for f in files]
    
    n_train = int(len(file_pairs) * train_ratio)
    np.random.seed(PARAMS['SEED'])
    np.random.shuffle(file_pairs)
    
    train_files = file_pairs[:n_train]
    test_files = file_pairs[n_train:]

    train_ids = [Path(f).stem for f in train_files]
    test_ids = [Path(f).stem for f in test_files]

    train_ids.sort(key=lambda x: int(x.split('day')[-1]))
    test_ids.sort(key=lambda x: int(x.split('day')[-1]))

    with open(f"train_days_{ticker}.txt", "w") as f:
        for item in train_ids:
            f.write(item + "\n")

    with open(f"test_days_{ticker}.txt", "w") as f:
        for item in test_ids:
            f.write(item + "\n")
    
    print(f"Saved train_days_{ticker}.txt and test_days_{ticker}.txt with day identifiers only.")
    
    return train_files, test_files

def load_saved_days(ticker: str) -> Tuple[List[str], List[str]]:
    target_folder = f"{ticker}_{PARAMS['CANDLE_FREQUENCY']}"
    with open(f"train_days_{ticker}.txt", "r") as f:
        train_files = [Path(f'{Path(target_folder)}/{Path(line.strip())}.csv') for line in f if line.strip()]
    
    with open(f"test_days_{ticker}.txt", "r") as f:
        test_files = [Path(f'{Path(target_folder)}/{Path(line.strip())}.csv') for line in f if line.strip()]
    
    return train_files, test_files

def make_env(data_list: List[pd.DataFrame], mode: str = 'train') -> Callable:
    def _init():
        return Monitor(IntradayTradingEnv(data_list, mode=mode))
    return _init

def plot_training_metrics(callback: TqdmCallback, ticker: str, save_folder: str = "training_plots"):
    os.makedirs(save_folder, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    if len(callback.entropy_losses) > 0:
        axes[0].plot(callback.entropy_losses, label='Entropy Loss', color='blue', linewidth=2)
        axes[0].set_xlabel('Training Updates')
        axes[0].set_ylabel('Entropy Loss')
        axes[0].set_title(f'{ticker} - Entropy Loss Over Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    if len(callback.explained_variances) > 0:
        axes[1].plot(callback.explained_variances, label='Explained Variance', color='green', linewidth=2)
        axes[1].set_xlabel('Training Updates')
        axes[1].set_ylabel('Explained Variance')
        axes[1].set_title(f'{ticker} - Explained Variance Over Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{save_folder}/{ticker}_training_metrics.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Training metrics plot saved to {save_path}")

def train_model_parallel(train_file_pairs: List[str], ticker: str, total_timesteps: int = PARAMS['EPISODES'], 
                         use_gpu: bool = True, seed: int = PARAMS['SEED']):
    print("\n" + "=" * 80)
    print(f"INITIALIZING PARALLEL TRAINING FOR {ticker}")
    print("=" * 80)
    
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    train_data_memory = precompute_all_data(train_file_pairs)
    
    temp_env = IntradayTradingEnv(train_data_memory, mode='train')
    save_feature_info(
        temp_env.feature_names,
        temp_env.lookback_features,
        temp_env.scalar_features,
        temp_env.state_size,
        temp_env.window,
        ticker,
        f"feature_info_{ticker}.txt"
    )
    
    print(f"\nState Space Dimension: {temp_env.state_size}")
    print(f"Action Space: {temp_env.action_space.n} actions (0=Hold, 1=Buy, 2=Sell)")
    print(f"Lookback Features: {len(temp_env.lookback_features)} x {temp_env.window} timesteps = {len(temp_env.lookback_features) * temp_env.window} dims")
    print(f"Scalar Features: {len(temp_env.scalar_features)} dims")
    print("Agent Features: 2 dims (position, mtm)")
    
    exit_params = get_exit_params('train')
    print(f"\nTraining Exit Parameters:")
    print(f"  Stop Loss: {exit_params['STOP_LOSS']}")
    print(f"  Trailing Stop: {exit_params['TRAIL_PCT']}")
    
    n_procs = PARAMS['NUM_PROCS']+2
    n_envs = max(1, n_procs - 2)
    print(f"Launching {n_envs} parallel environments...")
    
    env_cmds = [make_env(train_data_memory, mode='train') for _ in range(n_envs)]
    env = SubprocVecEnv(env_cmds)
    env.seed(seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    
    print("\nApplied VecNormalize (obs + reward normalization)")
    print("\n" + "=" * 80)
    print("CONFIGURING PPO MODEL FOR PARALLEL TRAINING")
    print("=" * 80)
    
    steps_per_env = 1024
    batch_size = 4096
    lr_schedule = linear_schedule(initial_value=3e-3, min_lr=1e-4)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        learning_rate=lr_schedule,
        n_steps=steps_per_env,
        batch_size=batch_size,
        n_epochs=10,
        ent_coef=0.001,
        gamma=0.99,
        gae_lambda=0.90,
        clip_range=0.2,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
        seed=seed,
    )
    
    print(f"Device: {device.upper()}")
    print("Policy: MlpPolicy with [256, 256] architecture")
    print("Learning Rate: Linear Schedule (3e-3 -> 1e-4)")
    print(f"Parallel Environments: {n_envs}")
    print(f"Steps per Environment: {steps_per_env}")
    print(f"Total Buffer Size per Update: {steps_per_env * n_envs}")
    print(f"Batch Size: {batch_size}")
    
    print("\n" + "=" * 80)
    print(f"STARTING PARALLEL TRAINING - {total_timesteps:,} timesteps")
    print("=" * 80 + "\n")
    
    callback = TqdmCallback(total_timesteps)
    model.learn(total_timesteps, callback=callback, progress_bar=False)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    model_folder = f"Models_{ticker}"
    os.makedirs(model_folder, exist_ok=True)
    
    save_path = f"{model_folder}/ppo_trading_model_{ticker}"
    model.save(save_path)
    env.save(f"{save_path}_vecnormalize.pkl")
    
    print(f"\nModel saved to: {save_path}.zip")
    print(f"VecNormalize stats saved to: {save_path}_vecnormalize.pkl")
    
    plot_training_metrics(callback, ticker)
    
    env.close()
    return model

def save_trade_plot(day_index: int, day_name: str, indicators_df: pd.DataFrame, 
                   trade_history: List[dict], ticker: str, save_folder: str = "test_trade_plots"):
    os.makedirs(save_folder, exist_ok=True)
    
    prices = indicators_df["_close"].values
    steps = list(range(len(prices)))
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(steps, prices, 'k-', linewidth=1.5, label='Price')
    
    for t in trade_history:
        s = int(t["step"])
        if s < 0 or s >= len(prices):
            continue
        
        price = prices[s]
        tt = t["trade_type"].replace(" ", "_")
        
        if tt == "OPEN_LONG":
            ax.scatter(s, price, color='lime', s=100, marker='^', label='Open Long', zorder=5)
        elif tt == "OPEN_SHORT":
            ax.scatter(s, price, color='red', s=100, marker='v', label='Open Short', zorder=5)
        elif tt == "CLOSE_LONG":
            ax.scatter(s, price, color='lightgreen', s=80, marker='o', label='Close Long', zorder=5)
        elif tt == "CLOSE_SHORT":
            ax.scatter(s, price, color='pink', s=80, marker='o', label='Close Short', zorder=5)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    ax.set_title(f"{ticker} - Day {day_index+1} - {day_name}")
    ax.set_xlabel("Step (2-min candles)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    
    out_path = f"{save_folder}/{ticker}_day_{day_index+1}_{day_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved chart -> {out_path}")

def plot_equity_and_drawdown(daily_pnls: List[float], ticker: str, save_folder: str = "test_results"):
    os.makedirs(save_folder, exist_ok=True)
    
    daily_returns = np.array(daily_pnls) / 10000.0
    equity_curve = 1.0 + np.cumsum(daily_returns)
    
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    axes[0].plot(equity_curve, linewidth=2, color='blue')
    axes[0].set_title(f'{ticker} - Equity Curve')
    axes[0].set_xlabel('Trading Days')
    axes[0].set_ylabel('Equity')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].fill_between(range(len(drawdowns)), drawdowns * 100, color='red', alpha=0.3)
    axes[1].plot(drawdowns * 100, linewidth=2, color='darkred')
    axes[1].set_title(f'{ticker} - Drawdown (%)')
    axes[1].set_xlabel('Trading Days')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{save_folder}/{ticker}_equity_drawdown.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Equity and drawdown plot saved to {save_path}")

def test_model(ticker: str, test_file_pairs: List[str], deterministic: bool = True, 
               specific_day: str = None):
    model_folder = f"Models_{ticker}"
    model_path = f"{model_folder}/ppo_trading_model_{ticker}.zip"
    vec_normalize_path = f"{model_folder}/ppo_trading_model_{ticker}_vecnormalize.pkl"
    output_file = f"test_results/test_results_{ticker}.txt"
    
    os.makedirs("test_results", exist_ok=True)
    shutil.rmtree(f"test_trade_plots") if os.path.exists(f"test_trade_plots") else None
    os.makedirs(f"test_trade_plots", exist_ok=True)
    shutil.rmtree(f"signals_{ticker}") if os.path.exists(f"signals_{ticker}") else None
    os.makedirs(f"signals_{ticker}", exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"LOADING TRAINED MODEL FOR {ticker}")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vec_normalize_path}")
    
    exit_params = get_exit_params('test')
    print(f"\nTesting Exit Parameters:")
    print(f"  Stop Loss: {exit_params['STOP_LOSS']}")
    print(f"  Trailing Stop: {exit_params['TRAIL_PCT']}")
    
    if specific_day:
        test_file_pairs = [f for f in test_file_pairs if re.search(rf'day{re.escape(specific_day)}\b', str(f))]
        if not test_file_pairs:
            print(f"Error: Day {specific_day} not found in test files")
            return
        print(f"Testing specific day: {specific_day}")
    
    print(f"Days to test: {len(test_file_pairs)}")
    
    model = PPO.load(model_path)
    print("\nModel loaded.\n")
    
    total_test_pnl = 0.0
    total_trades = 0
    winning_days = 0
    losing_days = 0
    daily_pnls = []
    all_trade_pnls = []
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"TEST RESULTS - {ticker}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"VecNormalize: {vec_normalize_path}\n")
        f.write(f"Deterministic: {deterministic}\n\n")
        
        for day_idx, file_120s in enumerate(tqdm(test_file_pairs, desc="Testing", unit="day")):
            day_name = Path(file_120s).stem
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"DAY {day_idx + 1}: {day_name}\n")
            f.write("=" * 80 + "\n")
            
            df_120s = load_ohlc_data(file_120s)
            indicators_df = calculate_all_indicators(df_120s)
            warmup_cutoff = indicators_df.index[0] + pd.Timedelta(minutes=PARAMS['WARMUP_MINUTES'])
            
            indicators_df_with_index = indicators_df[indicators_df.index > warmup_cutoff]
            original_timestamps = indicators_df_with_index.index.tolist()
            indicators_df = indicators_df_with_index.reset_index(drop=True)
            
            env = IntradayTradingEnv([indicators_df], mode='test')
            vec_env = DummyVecEnv([lambda e=env: e])
            vec_env = VecNormalize.load(vec_normalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            
            obs = vec_env.reset()
            done = np.array([False])
            
            step = 0
            prev_pos = 0
            day_pnl_bps = 0.0
            logged_trades = []
            open_positions = {}
            
            signals_data = []
            
            print("\n" + "=" * 80)
            print(f"TRADE LOG - {day_name}")
            print("=" * 80)
            print(f"{'Step':<6} {'Trade Type':<20} {'Price':<10} {'Pos':<10} {'MTM%':<8}")
            print("-" * 80)
            
            while not done[0]:
                env_real = vec_env.venv.envs[0]
                price_idx = env_real.current_step
                price = env_real.indicators_df.loc[env_real.current_step, "_close"]
                timestamp = original_timestamps[price_idx] if price_idx < len(original_timestamps) else None
                timestamp = timestamp + pd.Timedelta(minutes=2)
                entry = env_real.entry_price if env_real.entry_price > 0 else 0.0
                
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = vec_env.step(action)
                
                new_pos = vec_env.venv.envs[0].position
                
                buy_signal = 0
                sell_signal = 0
                exit_signal = 0
                
                if new_pos != prev_pos:
                    if prev_pos == 0 and new_pos == 1:
                        trade_type = "OPEN LONG"
                        mtm_pct = 0.0
                        logged_trades.append({"step": price_idx, "trade_type": trade_type})
                        open_positions["long"] = {"entry": price, "open_step": step}
                        buy_signal = 1
                        
                    elif prev_pos == 0 and new_pos == -1:
                        trade_type = "OPEN SHORT"
                        mtm_pct = 0.0
                        logged_trades.append({"step": price_idx, "trade_type": trade_type})
                        open_positions["short"] = {"entry": price, "open_step": step}
                        sell_signal = 1
                        
                    elif prev_pos == 1 and new_pos == 0:
                        trade_type = "CLOSE LONG"
                        mtm_pct = (price - entry) / entry * 100 if entry else 0.0
                        logged_trades.append({"step": price_idx, "trade_type": trade_type})
                        exit_signal = 1
                        
                        if "long" in open_positions:
                            entry_p = open_positions["long"]["entry"]
                            pnl_bps = ((price - entry_p) / entry_p * 10000) - PARAMS["TRANSACTION_COST"] * 10000 * 2
                            day_pnl_bps += pnl_bps
                            all_trade_pnls.append(pnl_bps)
                            total_trades += 1
                            del open_positions["long"]
                            
                    elif prev_pos == -1 and new_pos == 0:
                        trade_type = "CLOSE SHORT"
                        mtm_pct = (entry - price) / entry * 100 if entry else 0.0
                        logged_trades.append({"step": price_idx, "trade_type": trade_type})
                        exit_signal = 1
                        
                        if "short" in open_positions:
                            entry_p = open_positions["short"]["entry"]
                            pnl_bps = ((entry_p - price) / entry_p * 10000) - PARAMS["TRANSACTION_COST"] * 10000 * 2
                            day_pnl_bps += pnl_bps
                            all_trade_pnls.append(pnl_bps)
                            total_trades += 1
                            del open_positions["short"]
                    else:
                        trade_type = f"{prev_pos}->{new_pos}"
                        mtm_pct = 0.0
                    
                    pos_name = {0: "FLAT", 1: "LONG", -1: "SHORT"}[new_pos]
                    print(f"{step:<6} {trade_type:<20} {price:<10.4f} {pos_name:<10} {mtm_pct:<8.3f}")
                
                signals_data.append({
                    'Time': timestamp.strftime('%H:%M:%S') if timestamp else None,
                    'Price': price,
                    'BUY': buy_signal,
                    'SELL': sell_signal,
                    'EXIT': exit_signal
                })
                
                prev_pos = new_pos
                step += 1
            
            env_real = vec_env.venv.envs[0]
            last_step_idx = env_real.current_step - 1

            if last_step_idx >= 0 and last_step_idx < len(env_real.indicators_df):
                final_price = env_real.indicators_df.loc[last_step_idx, "_close"]
                final_timestamp = original_timestamps[last_step_idx] if last_step_idx < len(original_timestamps) else None
                
                if env_real.position == 1:
                    logged_trades.append({"step": last_step_idx, "trade_type": "CLOSE LONG"})
                    entry_p = env_real.entry_price
                    pnl_bps = ((final_price - entry_p) / entry_p * 10000) - PARAMS["TRANSACTION_COST"] * 10000 * 2
                    day_pnl_bps += pnl_bps
                    all_trade_pnls.append(pnl_bps)
                    total_trades += 1
                    print(f"{step:<6} {'CLOSE LONG (EOD)':<20} {final_price:<10.4f} {'FLAT':<10} {((final_price - entry_p) / entry_p * 100):<8.3f}")
                    
                elif env_real.position == -1:
                    logged_trades.append({"step": last_step_idx, "trade_type": "CLOSE SHORT"})
                    entry_p = env_real.entry_price
                    pnl_bps = ((entry_p - final_price) / entry_p * 10000) - PARAMS["TRANSACTION_COST"] * 10000 * 2
                    day_pnl_bps += pnl_bps
                    all_trade_pnls.append(pnl_bps)
                    total_trades += 1
                    print(f"{step:<6} {'CLOSE SHORT (EOD)':<20} {final_price:<10.4f} {'FLAT':<10} {((entry_p - final_price) / entry_p * 100):<8.3f}")
           
            save_trade_plot(day_idx, day_name, indicators_df, logged_trades, ticker)
            
            signals_df = pd.DataFrame(signals_data)
            signals_path = f"signals_{ticker}/{day_name}.csv"
            signals_df.to_csv(signals_path, index=False)
            
            print("-" * 80)
            print(f"Day PnL: {day_pnl_bps:+.2f} bps\n")
            
            f.write(f"Steps executed: {step}\n")
            f.write(f"Day PnL (bps): {day_pnl_bps:+.2f}\n")
            f.write(f"Signals saved to: {signals_path}\n")
            
            total_test_pnl += day_pnl_bps
            daily_pnls.append(day_pnl_bps)
            
            if day_pnl_bps > 0:
                winning_days += 1
            elif day_pnl_bps < 0:
                losing_days += 1
            
            vec_env.close()
        
        total_days = len(daily_pnls)
        flat_days = total_days - winning_days - losing_days
        avg_daily_pnl = float(np.mean(daily_pnls)) if total_days > 0 else 0.0
        
        if len(daily_pnls) > 0:
            daily_returns = np.array(daily_pnls) / 10000.0
            equity_curve = 1.0 + np.cumsum(daily_returns)
            
            peak = np.maximum.accumulate(equity_curve)
            drawdowns = (peak - equity_curve) / peak
            max_dd = float(drawdowns.max())
            
            total_simple_return = np.sum(daily_returns)
            annualized_return = total_simple_return * (252.0 / total_days) if total_days > 0 else 0.0
            calmar = annualized_return / max_dd if max_dd > 0 else 0.0
        else:
            max_dd = 0.0
            annualized_return = 0.0
            calmar = 0.0
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"FINAL SUMMARY - {ticker}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Days: {total_days}\n")
        f.write(f"Total PnL (bps): {total_test_pnl:+.2f}\n")
        f.write(f"Winning Days: {winning_days}\n")
        f.write(f"Losing Days: {losing_days}\n")
        f.write(f"Flat Days: {flat_days}\n")
        f.write(f"Average Daily PnL: {avg_daily_pnl:+.2f} bps\n")
        f.write(f"Total Trades: {total_trades}\n")
        
        if all_trade_pnls:
            avg_trade_pnl = float(np.mean(all_trade_pnls))
            f.write(f"Average Trade PnL: {avg_trade_pnl:+.2f} bps\n")
            
            wins = [p for p in all_trade_pnls if p > 0]
            losses = [p for p in all_trade_pnls if p < 0]
            if wins or losses:
                trade_win_rate = len(wins) / (len(wins) + len(losses)) * 100
                f.write(f"Winning Trades: {len(wins)}\n")
                f.write(f"Losing Trades: {len(losses)}\n")
                f.write(f"Trade Win Rate: {trade_win_rate:.2f}%\n")
                if wins:
                    f.write(f"Average Win: {np.mean(wins):+.2f} bps\n")
                if losses:
                    f.write(f"Average Loss: {np.mean(losses):+.2f} bps\n")
        
        f.write("\nRisk Metrics:\n")
        f.write(f"Annualized Return: {annualized_return * 100:.2f}%\n")
        f.write(f"Max Drawdown: {max_dd * 100:.2f}%\n")
        f.write(f"Calmar Ratio: {calmar:.3f}\n")
        f.write("=" * 80 + "\n")
    
    print("=" * 80)
    print(f"FINAL SUMMARY - {ticker}")
    print("=" * 80)
    print(f"Days tested: {total_days}")
    print(f"Total PnL: {total_test_pnl:+.2f} bps")
    print(f"Avg Daily PnL: {avg_daily_pnl:+.2f} bps")
    print(f"Winning Days: {winning_days}")
    print(f"Losing Days: {losing_days}")
    print(f"Flat Days: {flat_days}")
    print(f"Total Trades: {total_trades}")
    print(f"Annualized Return: {annualized_return * 100:.2f}%")
    print(f"Max Drawdown: {max_dd * 100:.2f}%")
    print(f"Calmar Ratio: {calmar:.3f}")
    print()
    
    plot_equity_and_drawdown(daily_pnls, ticker)
    
    print(f"Test results saved to {output_file}")
    

def my_broadcast_callback(state, ts):
    
    current_positions = backtest.position_map

    LOT = 100  # Fixed trade size

    for ticker, data in state.items():

        if "Price" not in data or data["Price"] == 0:
            continue

        current_pos = current_positions.get(ticker, 0)  # in shares

        # Position state based on quantity held
        is_long  = current_pos > 0
        is_short = current_pos < 0
        is_flat  = current_pos == 0

        try:
            buy_signal  = int(data.get("BUY", 0))
            sell_signal = int(data.get("SELL", 0))
            exit_signal = int(data.get("EXIT", 0))
        except (ValueError, TypeError):
            continue

        # --- PRIORITY: EXIT comes first ---
        if exit_signal == 1:
            if is_long:
                print(f"[{ts}] EXIT LONG {ticker} at {data['Price']}")
                backtest.place_order(ticker=ticker, qty=LOT, side=Side.SELL)
            elif is_short:
                print(f"[{ts}] EXIT SHORT {ticker} at {data['Price']}")
                backtest.place_order(ticker=ticker, qty=LOT, side=Side.BUY)

        # --- Entry Logic ---
        elif buy_signal == 1 and is_flat:
            print(f"[{ts}] BUY LONG {ticker} at {data['Price']}")
            backtest.place_order(ticker=ticker, qty=LOT, side=Side.BUY)

        elif sell_signal == 1 and is_flat:
            print(f"[{ts}] SELL SHORT {ticker} at {data['Price']}")
            backtest.place_order(ticker=ticker, qty=LOT, side=Side.SELL)

def on_timer(ts):
    print(f"\n[TIMER] Timestamp={ts}")
    backtest.position_manager.print_details()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("   python EBY.py train")
        print("   python EBY.py test")
        print("   python EBY.py test <day_number>")
        print("   python EBY.py backtest_ebullient")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    ticker = 'EBY'
    
    source_folder = ticker if Path(ticker).exists() else Path(PARAMS['SOURCE_FOLDER'])
    target_folder = f"{ticker}_{PARAMS['CANDLE_FREQUENCY']}"
    
    if command == "train":
        print(f"\n{'='*80}")
        print(f"TRAINING PIPELINE FOR {ticker}")
        print(f"{'='*80}\n")
        
        print("Step 1: Resampling to 2-min candles")
        resample_to_candles(source_folder, target_folder, PARAMS['CANDLE_FREQUENCY'])
        
        print("\nStep 2: Loading and splitting data")
        train_files, test_files = load_data(target_folder, ticker, PARAMS['TRAIN_RATIO'])
        
        print(f"\nTrain files: {len(train_files)}")
        print(f"Test files: {len(test_files)}")
        
        print("\nStep 3: Training model")
        train_model_parallel(train_files, ticker, PARAMS['EPISODES'], use_gpu=True, seed=PARAMS['SEED'])
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED FOR {ticker}")
        print(f"{'='*80}\n")
        
    elif command == "test":
        print(f"\n{'='*80}")
        print(f"TESTING PIPELINE FOR {ticker}")
        print(f"{'='*80}\n")
        
        train_files, test_files = load_saved_days(ticker)
        
        specific_day = None
        if len(sys.argv) > 3:
            specific_day = sys.argv[3]
            print(f"Testing specific day: {specific_day}")
        
        test_model(ticker, test_files, deterministic=True, specific_day=specific_day)
        
        print(f"\n{'='*80}")
        print(f"TESTING COMPLETED FOR {ticker}")
        print(f"{'='*80}\n")
        
    elif command == "backtest_ebullient":
        print(f"\n{'='*80}")
        print(f"EBULLIENT BACKTESTING FOR {ticker}")
        print(f"{'='*80}\n")
        
        signals_folder = f"signals_{ticker}"

        signal_files = sorted(glob.glob(f"{signals_folder}/*.csv"), 
                     key=lambda x: int(re.search(r'day(\d+)', Path(x).name).group(1)))
        if not signal_files:
            print(f"Error: No signal files found in {signals_folder}")
            sys.exit(1)
        
        print(f"Found {len(signal_files)} signal files")
        
        try:
            max_day_i = int(re.search(r'day(\d+)\.csv', Path(signal_files[-1]).name).group(1)) 
        except AttributeError:
            print("Error: Could not extract day number from signal file name. Check file format (day{i}.csv).")
            sys.exit(1)
        
        try: 
            config = {
                "data_path": '.',
                "start_date": 0,
                "end_date": max_day_i,
                "timer": 600,
                "tcost": 2,
                "broadcast": [f"signals_{ticker}"]
            }
            
            config_file = os.path.join("config.json")
            with open(config_file, "w") as f:
                json.dump(config, f, indent=4)
            
            print(f"Created temporary config at: {config_file}")
            print(f"Backtesting days 0 to {max_day_i}.")
            print("\nStarting backtest...\n")
            
            global backtest
            backtest = BacktesterIIT(config_file)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Backtest starting...")
            
            backtest.run(
                broadcast_callback=my_broadcast_callback,
                timer_callback=on_timer
            )
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Backtest finished.")
            
            print("\n" + "=" * 80)
            print("BACKTEST RESULTS")
            print("=" * 80)
            
        except Exception as e:
            print(f"\nError during backtesting: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'='*80}")
        print(f"EBULLIENT BACKTESTING COMPLETED FOR {ticker}")
        print(f"{'='*80}\n")
            
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: train, test, backtest_ebullient")
        sys.exit(1)