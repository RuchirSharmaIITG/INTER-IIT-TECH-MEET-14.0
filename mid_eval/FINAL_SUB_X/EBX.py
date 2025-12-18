#!/usr/bin/env python3
"""
EBX.py - Trading Strategy
Complete pipeline for training, testing, and backtesting trading strategies.

Usage:
    Training:
        python EBX.py train --days 250 --config config_EBX.json
        python EBX.py train --days MODELS/train_days_EBX.txt --config config_EBX.json
    
    Testing:
        python EBX.py test --config config_EBX.json
        python EBX.py test --days MODELS/test_days_EBX.txt --config config_EBX.json
        
    Backtesting:
        python EBX.py backtest_ebullient --config config_EBX.json
"""

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
import xgboost as xgb
import gc
import os

from pykalman import KalmanFilter
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
try:
    from alpha_research.backtesterIIT import BacktesterIIT, Side, PositionManager
    from alpha_research.resultReport import ResultReport
except ImportError:
    print("WARNING: Could not import 'backtesterIIT' or 'resultReport'. Backtesting command will fail.")
# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

CONFIG = {}

def load_config(config_path):
    """Load configuration from JSON file."""
    global CONFIG
    with open(config_path, 'r') as f:
        CONFIG = json.load(f)
    print(f"✓ Loaded configuration from {config_path}")
    return CONFIG

# =============================================================================
# DATA PREPARATION MODULE
# =============================================================================

def generate_kalman_signals(df_in: pd.DataFrame) -> pd.DataFrame:
    """Apply Kalman filter and generate trading signals based on slope."""
    df = df_in.copy()
    
    kf = KalmanFilter(
        transition_matrices=[[1]],
        observation_matrices=[[1]],
        transition_covariance=[[CONFIG['kalman']['q']]],
        observation_covariance=[[CONFIG['kalman']['r']]],
        initial_state_mean=df['Price'].iloc[0],
        initial_state_covariance=[[1.0]]
    )
    
    states, _ = kf.smooth(df['Price'].values)
    df['KP'] = states.flatten()
    df['Slope'] = df['KP'].diff(CONFIG['kalman']['slope_period'])
    df['StdS'] = df['Slope'].rolling(250).std()
    df['UpTh'] = df['StdS'] * CONFIG['kalman']['slope_std_mult']
    df['DnTh'] = df['StdS'] * -CONFIG['kalman']['slope_std_mult']

    pos = pd.Series(0, index=df.index, dtype=int)
    curr_pos = 0
    
    for i in range(1, len(df)):
        s = df['Slope'].iloc[i]
        up = df['UpTh'].iloc[i]
        dn = df['DnTh'].iloc[i]
        
        if curr_pos == 1 and s <= 0:
            curr_pos = 0
        elif curr_pos == -1 and s >= 0:
            curr_pos = 0
        
        if curr_pos == 0:
            if s > up:
                curr_pos = 1
            elif s < dn:
                curr_pos = -1
        
        pos.iloc[i] = curr_pos
    
    df['Pos'] = pos
    prev = df['Pos'].shift(1)
    curr = df['Pos']

    df['BuyL'] = ((curr == 1) & (prev == 0)).astype(int)
    df['ExitL'] = ((curr == 0) & (prev == 1)).astype(int)
    df['BuyS'] = ((curr == -1) & (prev == 0)).astype(int)
    df['ExitS'] = ((curr == 0) & (prev == -1)).astype(int)
    
    df['Signal'] = 0
    df.loc[df['BuyL'] == 1, 'Signal'] = 1
    df.loc[df['BuyS'] == 1, 'Signal'] = -1
    
    df.drop(columns=['Pos', 'KP', 'Slope', 'StdS', 'UpTh', 'DnTh'], inplace=True, errors='ignore')
    
    return df

def calc_regime(df_in: pd.DataFrame, t=7, lim=15, th=2, rst=8) -> pd.Series:
    """Calculate CUSUM regime signal."""
    pb10 = f"PB10_T{t}"
    pb11 = f"PB11_T{t}"
    pb10_r = f"PB10_T{rst}"
    pb11_r = f"PB11_T{rst}"

    cols = ["Price", pb10, pb11, pb10_r, pb11_r]
    missing = [c for c in cols if c not in df_in.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df = df_in[cols].copy()

    h10 = (df["Price"] >= df[pb10]).astype(int)
    h11 = (df["Price"] <= df[pb11]).astype(int)
    r10 = (df["Price"] <= df[pb10_r]).astype(int)
    r11 = (df["Price"] >= df[pb11_r]).astype(int)

    sup = np.zeros(len(df))
    sdn = np.zeros(len(df))

    for i in range(1, len(df)):
        sup[i] = max(0, sup[i-1] + h10.iloc[i])
        if r11.iloc[i] == 1:
            sup[i] = 0
        if sup[i] > lim:
            sup[i] = lim

        sdn[i] = min(0, sdn[i-1] - h11.iloc[i])
        if r10.iloc[i] == 1:
            sdn[i] = 0
        if sdn[i] < -lim:
            sdn[i] = -lim

    df["Sup"] = sup
    df["Sdn"] = sdn

    sig = pd.Series(np.zeros(len(df)), index=df.index)
    sig[df["Sup"] >= th] = 1
    sig[df["Sdn"] <= -th] = -1
    sig.name = "Regime"
    
    return sig

def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features."""
    df = df.copy()
    
    t = pd.to_datetime(df['Time']) if 'Time' in df.columns else pd.to_datetime(df.index)
    
    h = t.dt.hour
    m = t.dt.minute
    
    df['H_sin'] = np.sin(2 * np.pi * h / 24)
    df['H_cos'] = np.cos(2 * np.pi * h / 24)
    df['M_sin'] = np.sin(2 * np.pi * m / 60)
    df['M_cos'] = np.cos(2 * np.pi * m / 60)
    
    return df

def add_custom_feats(df: pd.DataFrame) -> pd.DataFrame:
    """Add custom feature combinations."""
    df = df.copy()
    
    feat_list = []
    
    if all(c in df.columns for c in ['BB9_T6', 'VB4_T6', 'PB4_T6']):
        df['F1'] = np.sqrt(df['BB9_T6'] * df['VB4_T6']) * df['PB4_T6']
        feat_list.append('F1')
    
    if all(c in df.columns for c in ['BB4_T3', 'BB4_T6']):
        df['F2'] = df['BB4_T3'] / (df['BB4_T6'] + 1e-10)
        feat_list.append('F2')
    
    if all(c in df.columns for c in ['BB4_T2', 'BB4_T6']):
        df['F3'] = df['BB4_T2'] / (df['BB4_T6'] + 1e-10)
        feat_list.append('F3')
    
    return df, feat_list

def get_t_feats(cols: list, pfx: list, max_t: int) -> list:
    """Filter columns by prefix and T-level."""
    t_lvls = {f"_T{i}" for i in range(1, max_t + 1)}
    
    feats = []
    for col in cols:
        for p in pfx:
            if col.startswith(p):
                for t in t_lvls:
                    if col.endswith(t):
                        feats.append(col)
                        break
                break
    
    return feats

def load_preprocess(path: str, day: int) -> tuple:
    """Load data, generate features and labels."""
    df = pd.read_csv(path)

    df = add_time_feats(df)
    df = generate_kalman_signals(df)
    
    try:
        reg = calc_regime(df.copy(), 
                         t=CONFIG['regime']['t'], 
                         rst=CONFIG['regime']['rst'],
                         lim=CONFIG['regime']['lim'],
                         th=CONFIG['regime']['th'])
        df = df.join(reg)
    except ValueError as e:
        print(f"Regime error: {e}")
        df['Regime'] = 0
    
    df, custom_feats = add_custom_feats(df)
    
    feats = get_t_feats(df.columns.tolist(), 
                        CONFIG['features']['families'],
                        CONFIG['features']['max_t'])
    
    if 'Regime' in df.columns:
        feats.append('Regime')
    
    for tf in ['H_sin', 'H_cos', 'M_sin', 'M_cos']:
        if tf in df.columns:
            feats.append(tf)
    
    feats.extend(custom_feats)
    
    tgt = 'Signal'
    df_m = df[feats + [tgt]].copy()
    
    del df
    
    df_m = df_m.iloc[CONFIG['features']['max_t_drop']:].copy()
    df_m.fillna(method='ffill', inplace=True)
    
    print(f"Final shape: {df_m.shape}")
    return df_m, feats, tgt

def create_sequences(df: pd.DataFrame, feats: list, tgt: str, lb: int, hr: float = 1.0) -> tuple:
    """Create time series sequences with sampled Hold signals."""
    X = df[feats].values
    y = df[tgt].values
    
    bl_idx = np.where(y == 1)[0]
    bs_idx = np.where(y == -1)[0]
    h_idx = np.where(y == 0)[0]
    
    print(f"BuyL={len(bl_idx)}, BuyS={len(bs_idx)}, Hold={len(h_idx)}")
    
    n_act = len(bl_idx) + len(bs_idx)
    n_hold = int(n_act * hr)
    
    if len(h_idx) > n_hold:
        np.random.seed(42)
        h_idx = np.random.choice(h_idx, size=n_hold, replace=False)
    
    fin_idx = np.concatenate([bl_idx, bs_idx, h_idx])
    fin_idx = fin_idx[fin_idx >= (lb - 1)]
    
    fin_idx = shuffle(fin_idx, random_state=42)
    
    Xf = []
    yf = []
    
    for i in fin_idx:
        win = X[i - (lb - 1) : i + 1]
        Xf.append(win)
        yf.append(y[i])
    
    Xf = np.array(Xf)
    yf = np.array(yf)

    print(f"X: {Xf.shape}, y: {yf.shape}")

    ym = np.zeros_like(yf, dtype=np.int32)
    ym[yf == 0] = 0
    ym[yf == 1] = 1
    ym[yf == -1] = 2
    
    return Xf, ym

def prepare_train_data(n_days):
    out_path = Path(CONFIG['paths']['train_data_dir'])
    out_path.mkdir(parents=True, exist_ok=True)
    
    data_path = Path(CONFIG['paths']['data_dir'])
    all_files = sorted(data_path.glob("day*.csv"))
    
    if len(all_files) == 0:
        return

    all_day_names = {f.stem for f in all_files}
    sel_day_names = []
    
    if isinstance(n_days, str) and n_days.endswith('.txt'):
        with open(n_days, 'r') as f:
            sel_day_names = [line.strip() for line in f if line.strip()]
        sel_files = [data_path / f"{day}.csv" for day in sel_day_names]
    else:
        required_days = int(n_days)
        train_days_file_path = Path(CONFIG['paths']['models_dir']) / Path(CONFIG['paths']['train_days_file'])
        test_days_file_path = Path(CONFIG['paths']['models_dir']) / Path(CONFIG['paths']['test_days_file'])
        all_day_names_set = set()
        if train_days_file_path.exists():
                with open(train_days_file_path, 'r') as f:
                    all_day_names_set.update(line.strip() for line in f if line.strip())
                    all_available_days = list(all_day_names_set)
                    
                if len(all_available_days) < required_days and test_days_file_path.exists():
                    with open(test_days_file_path, 'r') as f:
                        all_day_names_set.update(line.strip() for line in f if line.strip())
                        all_available_days = list(all_day_names_set)
                        
                if len(all_available_days) < required_days:
                    sel_day_names = all_available_days
                    print(f"Warning: Only {len(sel_day_names)} unique days available. Using all of them for training.")
                else :
                    sel_day_names = random.sample(all_available_days, required_days)
        sel_files = [data_path / f"{day}.csv" for day in sel_day_names if Path(data_path / f"{day}.csv").exists()]

    processed_day_names = []
    
    for data_file in tqdm(sel_files, desc="Processing Train Days"):
        day_num = int(data_file.stem.replace('day', ''))
        df_proc = None
        try:
            df_proc, feats, tgt = load_preprocess(
                path=str(data_file),
                day=day_num
            )
            
            if len(df_proc) > CONFIG['training']['lookback']:
                Xt, yt = create_sequences(
                    df=df_proc,
                    feats=feats,
                    tgt=tgt,
                    lb=CONFIG['training']['lookback'],
                    hr=CONFIG['training']['hold_ratio']
                )
                
                out_X = out_path / f'X_day{day_num}.npy'
                out_y = out_path / f'y_day{day_num}.npy'
                
                np.save(out_X, Xt)
                np.save(out_y, yt)
                
                processed_day_names.append(data_file.stem)
                
                del Xt, yt
            else:
                pass
        
        except Exception as e:
            pass
        finally:
            if df_proc is not None:
                del df_proc

    if processed_day_names:
        train_output_file_path = out_path / Path(CONFIG['paths']['train_days_file'])
        sorted_train_names = sorted(processed_day_names, key=lambda x: int(x.replace('day', '')))
        
        with open(train_output_file_path, 'w') as f:
            for day_name in sorted_train_names:
                f.write(f"{day_name}\n")

    processed_day_set = set(processed_day_names)
    test_day_names = all_day_names - processed_day_set
    
    test_output_file_path = out_path / Path(CONFIG['paths']['test_days_file'])
    sorted_test_day_names = sorted(test_day_names, key=lambda x: int(x.replace('day', '')))
    
    with open(test_output_file_path, 'w') as f:
        for day_name in sorted_test_day_names:
            f.write(f"{day_name}\n")
# =============================================================================
# TRAINING MODULE
# =============================================================================

def load_batches(data_dir: str, batch_sz: int):
    """Load data in batches to manage memory."""
    dp = Path(data_dir)
    if not dp.is_dir():
        return
    
    x_fs = sorted(list(dp.glob('X_day*.npy')))
    y_fs = sorted(list(dp.glob('y_day*.npy')))
    
    if not x_fs or not y_fs:
        return

    print(f"Found {len(x_fs)} files")
    
    xb = []
    yb = []
    cnt = 0
    
    for xf, yf in tqdm(zip(x_fs, y_fs), desc="Loading", total=len(x_fs)):
        try:
            xd = np.load(xf)
            yd = np.load(yf)
            
            if xd.shape[0] != yd.shape[0]:
                continue
            
            xb.append(xd)
            yb.append(yd)
            cnt += len(xd)
            
            del xd, yd
            gc.collect()
            
            if cnt >= batch_sz:
                xc = np.concatenate(xb, axis=0)
                yc = np.concatenate(yb, axis=0)
                
                yield xc, yc
                
                xb = []
                yb = []
                cnt = 0
                del xc, yc
                gc.collect()
                
        except Exception as e:
            continue
    
    if xb:
        xc = np.concatenate(xb, axis=0)
        yc = np.concatenate(yb, axis=0)
        yield xc, yc

def get_feat_names(data_dir: str) -> tuple:
    """Extract feature dimensions from first file."""
    dp = Path(data_dir)
    xfs = sorted(list(dp.glob('X_day*.npy')))
    
    if not xfs:
        raise FileNotFoundError("No data files found")
    
    xs = np.load(xfs[0])
    ns, nt, nf = xs.shape
    
    print(f"Shape: ({ns}, {nt}, {nf})")
    
    fnames = []
    for t in range(nt):
        for fi in range(nf):
            fnames.append(f"F{fi}_T{nt - 1 - t}")
    
    return fnames, nf

def flatten(X: np.ndarray) -> np.ndarray:
    """Flatten 3D to 2D."""
    ns, nt, nf = X.shape
    return X.reshape(ns, nt * nf)

def plot_feat_imp(imp_df: pd.DataFrame, out_dir: str, top: int = 30):
    """Plot top feature importance."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    topf = imp_df.head(top)
    
    cols = plt.cm.viridis(np.linspace(0, 1, len(topf)))
    plt.barh(range(len(topf)), topf['Imp'], color=cols)
    plt.yticks(range(len(topf)), topf['Feat'])
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Feature')
    plt.title(f'Top {top} Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/feat_imp_top{top}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {out_dir}/feat_imp_top{top}.png")

def plot_cm(yt, yp, out_dir: str, cls_names: list):
    """Plot confusion matrix."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(yt, yp)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cls_names, yticklabels=cls_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/cm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {out_dir}/cm.png")

def train_model():
    """Train XGBoost model."""
    print("="*80)
    print("XGBoost Training (3-Class: Hold/Long/Short)")
    print("="*80)
    
    data_dir = CONFIG['paths']['train_data_dir']
    plots_dir = CONFIG['paths']['plots_dir']
    
    print("\n[1/6] Getting feature structure...")
    fnames, nf = get_feat_names(data_dir)
    
    feat_path = CONFIG['paths']['feature_names_file']
    with open(feat_path, 'w') as f:
        for fn in fnames:
            f.write(f"{fn}\n")
    print(f"Saved {len(fnames)} features to {feat_path}")
    
    print("\n[2/6] Fitting scaler...")
    scaler = RobustScaler()
    
    dgen = load_batches(data_dir, batch_sz=CONFIG['training']['batch_size'])
    try:
        xb1, yb1 = next(dgen)
    except StopIteration:
        print("No data batches")
        return None

    ns, nt, nfa = xb1.shape
    print(f"Batch shape: {xb1.shape}")
    
    xf = flatten(xb1)
    scaler.fit(xf)
    
    print(f"Scaler fitted on {len(xf)} samples")
    del xf, xb1, yb1
    gc.collect()
    
    scaler_path = CONFIG['paths']['scaler_file']
    joblib.dump(scaler, scaler_path)
    print(f"Saved {scaler_path}")
    
    print("\n[3/6] Processing data...")
    
    xtr_all = []
    ytr_all = []
    xv_all = []
    yv_all = []
    
    tot = 0
    
    for xb, yb in load_batches(data_dir, batch_sz=CONFIG['training']['batch_size']):
        xf = flatten(xb)
        xs = scaler.transform(xf)
        
        spl = int(0.8 * len(xs))
        
        xtr_all.append(xs[:spl])
        ytr_all.append(yb[:spl])
        xv_all.append(xs[spl:])
        yv_all.append(yb[spl:])
        
        tot += len(xs)
        
        del xb, yb, xf, xs
        gc.collect()
    
    print(f"Total samples: {tot}")
    print("Concatenating...")
    
    xtr = np.concatenate(xtr_all, axis=0)
    ytr = np.concatenate(ytr_all, axis=0)
    xv = np.concatenate(xv_all, axis=0)
    yv = np.concatenate(yv_all, axis=0)
    
    del xtr_all, ytr_all, xv_all, yv_all
    gc.collect()
    
    print(f"Train: {len(xtr)}")
    print(f"Val: {len(xv)}")
    
    print("\n[4/6] Creating DMatrix...")
    
    dtr = xgb.DMatrix(xtr, label=ytr, feature_names=fnames)
    dv = xgb.DMatrix(xv, label=yv, feature_names=fnames)
    
    del xtr, ytr, xv, yv
    gc.collect()
    
    print("DMatrix created")
    
    print("\n[5/6] Training model...")
    print("="*80)
    
    evs = [(dtr, 'train'), (dv, 'val')]
    evals_res = {}
    
    xgb_params = CONFIG['xgboost']
    
    bst = xgb.train(
        params=xgb_params,
        dtrain=dtr,
        num_boost_round=xgb_params['n_estimators'],
        evals=evs,
        evals_result=evals_res,
        early_stopping_rounds=CONFIG['training']['early_stop'],
        verbose_eval=10
    )
    
    print("\n" + "="*80)
    print("Training complete")
    
    print("\n[6/6] Evaluating...")
    
    yprob = bst.predict(dv)
    ypred = np.argmax(yprob, axis=1)
    ytrue = dv.get_label()
    
    acc = accuracy_score(ytrue, ypred)
    
    print("="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    cls_names = ['Hold', 'Buy_Long', 'Buy_Short']
    print(classification_report(ytrue, ypred, target_names=cls_names, zero_division=0))
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    
    imp_dict = bst.get_score(importance_type='gain')
    imp_df = pd.DataFrame([
        {'Feat': k, 'Imp': v} 
        for k, v in imp_dict.items()
    ]).sort_values('Imp', ascending=False)
    
    os.makedirs(plots_dir, exist_ok=True)
    imp_df.to_csv(f'{plots_dir}/feat_imp.csv', index=False)
    print(f"Saved {plots_dir}/feat_imp.csv")
    
    print("\nTop 20 Features:")
    for i, row in imp_df.head(20).iterrows():
        print(f"{i + 1}. {row['Feat']}: {row['Imp']:.2f}")
    
    print("\nGenerating plots...")
    plot_feat_imp(imp_df, plots_dir, top=30)
    plot_cm(ytrue, ypred, plots_dir, cls_names)
    
    model_path = CONFIG['paths']['model_file']
    print(f"\nSaving model to {model_path}...")
    bst.save_model(model_path)
    
    print("\n" + "="*80)
    print("Training Complete")
    print("="*80)
    
    return bst

# =============================================================================
# TEST DATA PREPARATION MODULE
# =============================================================================

def load_scaler(path: str) -> tuple:
    """Load fitted scaler and infer feature count."""
    try:
        scl = joblib.load(path)
        
        if hasattr(scl, 'center_'):
            nf_tot = scl.center_.shape[0]
            nf_orig = nf_tot // CONFIG['training']['lookback']
            print(f"Scaler expects {nf_tot} features ({CONFIG['training']['lookback']} steps × {nf_orig} features)")
            return scl, nf_orig
        else:
            print("Error: Scaler missing center_")
            return None, None
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None, None

def process_test_day(day: int, scl: RobustScaler, exp_nf: int):
    """Process one test day following training pipeline."""
    fname = f'day{day}.csv'
    fpath = Path(CONFIG['paths']['data_dir']) / fname
    
    if not fpath.exists():
        print(f"  Skip {fname}: Not found")
        return
    
    df = pd.read_csv(fpath)
    if df.empty or 'Price' not in df.columns:
        print(f"  Skip {fname}: Empty or no Price")
        return
    
    print(f"  {fname}: {df.shape[0]} rows")
    
    df = add_time_feats(df)
    
    try:
        reg = calc_regime(df.copy(), 
                         t=CONFIG['regime']['t'], 
                         rst=CONFIG['regime']['rst'],
                         lim=CONFIG['regime']['lim'],
                         th=CONFIG['regime']['th'])
        df = df.join(reg)
    except ValueError as e:
        print(f"  Warning: {e}")
        df['Regime'] = 0
    
    df, cust_feats = add_custom_feats(df)
    
    all_cols = df.columns.tolist()
    feats = get_t_feats(all_cols, CONFIG['features']['families'], CONFIG['features']['max_t'])
    
    if 'Regime' in df.columns:
        feats.append('Regime')
    
    for tf in ['H_sin', 'H_cos', 'M_sin', 'M_cos']:
        if tf in df.columns:
            feats.append(tf)
    
    feats.extend(cust_feats)
    
    df_m = df[feats + ['Price']].copy()
    del df
    
    print(f"  Drop first {CONFIG['features']['max_t_drop']} rows")
    df_m = df_m.iloc[CONFIG['features']['max_t_drop']:].copy()
    df_m.fillna(method='ffill', inplace=True)
    
    nf_in_df = len(feats)
    if nf_in_df != exp_nf:
        print(f"  Error: Expected {exp_nf} features, got {nf_in_df}")
        return
    
    feat_arr = df_m[feats].values
    prices = df_m['Price'].values
    tot_t = feat_arr.shape[0]
    
    lookback = CONFIG['training']['lookback']
    stride = CONFIG['testing']['stride']
    first_valid = lookback - 1
    
    if first_valid >= tot_t:
        print(f"  Skip {fname}: Not enough data")
        return
    
    pred_idx = np.arange(first_valid, tot_t, stride)
    n_seq = len(pred_idx)
    
    if n_seq <= 0:
        print(f"  Skip {fname}: No sequences with stride={stride}")
        return
    
    X_day = np.empty((n_seq, lookback, nf_in_df), dtype=np.float32)
    pred_prices = np.empty(n_seq, dtype=np.float32)
    
    for seq_n, pidx in enumerate(pred_idx):
        start = pidx - lookback + 1
        end = pidx + 1
        
        seq = feat_arr[start:end]
        X_day[seq_n] = seq
        pred_prices[seq_n] = prices[pidx]
    
    X_flat = X_day.reshape(n_seq, -1)
    X_scaled = scl.transform(X_flat)
    
    out_path = Path(CONFIG['paths']['test_data_dir'])
    out_path.mkdir(exist_ok=True, parents=True)
    
    np.save(out_path / f'X_test_day{day:03d}.npy', X_scaled)
    
    abs_idx = pred_idx + CONFIG['features']['max_t_drop']
    np.save(out_path / f'indices_day{day:03d}.npy', abs_idx)
    np.save(out_path / f'prices_day{day:03d}.npy', pred_prices)
    
    print(f"  Done: {n_seq} sequences")

def prepare_test_data(test_days_path: str):
    """Prepare test sequences based on a list of days."""
    print("="*80)
    print("Test Sequence Preparation")
    print("="*80)
    
    print("\n[1/3] Loading scaler...")
    scl, exp_nf = load_scaler(CONFIG['paths']['scaler_file'])
    
    if scl is None or exp_nf is None:
        print("\nError: Could not load scaler")
        return

    print(f"\n[2/3] Loading test day list...")
    if isinstance(test_days_path, str) and test_days_path.endswith('.txt'):
        test_file = Path(test_days_path)
        print(f"Using provided file: {test_file}")
    else:
        test_file = Path(CONFIG['paths']['train_data_dir']) / Path(CONFIG['paths']['test_days_file'])
        print(f"Using generated file: {test_file}")
    
    if not test_file.exists():
        print(f"\nError: Test days file not found at {test_file}")
        return
    
    with open(test_file, 'r') as f:
        day_names = [line.strip() for line in f if line.strip()]
    
    day_numbers = []
    for name in day_names:
        try:
            day_numbers.append(int(name.replace('day', '')))
        except ValueError:
            print(f"  - Warning: Skipping malformed line: {name}")
    
    if not day_numbers:
        print("\nError: No valid days found in test file.")
        return
    
    print(f"Loaded {len(day_numbers)} days to process.")

    print(f"\n[3/3] Processing...")
    print("-"*80)
    
    for day in tqdm(day_numbers, desc="Days"):
        try:
            process_test_day(day, scl, exp_nf)
        except Exception as e:
            print(f" E  Error day {day}: {e}")
            continue
    
    print("\n" + "="*80)
    print("Complete")
    print("="*80)

# =============================================================================
# SIGNAL GENERATION MODULE
# =============================================================================

def create_candles(price_df, candle_minutes=5):
    """Create OHLC candles from tick data."""
    price_df = price_df.copy()
    price_df['candle_group'] = price_df.index.floor(f'{candle_minutes}T')
    
    candles = price_df.groupby('candle_group').agg({
        'Price': ['first', 'max', 'min', 'last', 'count']
    })
    
    candles.columns = ['open', 'high', 'low', 'close', 'tick_count']
    candles = candles.reset_index()
    candles.columns = ['timestamp', 'open', 'high', 'low', 'close', 'tick_count']
    
    return candles

def detect_whipsaw_day(candles_df, observation_minutes=60, opening_range_candles=3):
    """Detect if a day is a whipsaw day."""
    if len(candles_df) < opening_range_candles:
        return False, None, None, "Insufficient candles"
    
    opening_range_high = candles_df.iloc[:opening_range_candles]['high'].max()
    opening_range_low = candles_df.iloc[:opening_range_candles]['low'].min()
    
    opening_time = candles_df.iloc[0]['timestamp']
    observation_end_time = opening_time + timedelta(minutes=observation_minutes)
    
    observation_candles = candles_df[candles_df['timestamp'] <= observation_end_time]
    
    first_break_direction = 0
    is_whipsaw = False
    whipsaw_details = "No breakout"
    
    for idx in range(opening_range_candles, len(observation_candles)):
        candle = observation_candles.iloc[idx]
        
        if first_break_direction == 0:
            if candle['close'] > opening_range_high:
                first_break_direction = 1
                whipsaw_details = f"First break: Bullish at {candle['timestamp']}"
            elif candle['close'] < opening_range_low:
                first_break_direction = -1
                whipsaw_details = f"First break: Bearish at {candle['timestamp']}"
        else:
            if first_break_direction == 1 and candle['close'] < opening_range_low:
                is_whipsaw = True
                whipsaw_details = f"WHIPSAW: Bullish break reversed to bearish at {candle['timestamp']}"
                break
            elif first_break_direction == -1 and candle['close'] > opening_range_high:
                is_whipsaw = True
                whipsaw_details = f"WHIPSAW: Bearish break reversed to bullish at {candle['timestamp']}"
                break
    
    return is_whipsaw, opening_range_high, opening_range_low, whipsaw_details

class CandleBasedStrategy:
    """Simple candle-based strategy."""
    
    def __init__(self, candle_minutes=5, threshold_open=0, threshold_close=10, warmup_minutes=5):
        self.candle_minutes = candle_minutes
        self.threshold_open = threshold_open
        self.threshold_close = threshold_close
        self.warmup_minutes = warmup_minutes
        
        self.current_candle_predictions = []
        self.current_candle_start = None
        
        self.current_position = 0
        self.warmup_complete = False
        self.strategy_start_time = None
        
    def _get_candle_floor(self, timestamp):
        """Get the candle period for a timestamp."""
        return timestamp.floor(f'{self.candle_minutes}T')
    
    def _count_predictions(self):
        """Count long and short predictions in current candle."""
        if not self.current_candle_predictions:
            return {'long': 0, 'short': 0, 'hold': 0, 'total': 0}
        
        long_count = sum(1 for cls in self.current_candle_predictions if cls == 2)
        short_count = sum(1 for cls in self.current_candle_predictions if cls == 1)
        hold_count = sum(1 for cls in self.current_candle_predictions if cls == 0)
        
        return {
            'long': long_count,
            'short': short_count,
            'hold': hold_count,
            'total': long_count + short_count + hold_count
        }
    
    def add_prediction(self, timestamp, predicted_class):
        """Add a prediction to current candle."""
        candle_floor = self._get_candle_floor(timestamp)
        
        if self.current_candle_start is None:
            self.current_candle_start = candle_floor
            self.strategy_start_time = timestamp
        
        if candle_floor != self.current_candle_start:
            self.current_candle_predictions = []
            self.current_candle_start = candle_floor
        
        self.current_candle_predictions.append(predicted_class)
    
    def evaluate_at_candle_close(self, timestamp):
        """Evaluate strategy at candle close."""
        if not self.warmup_complete:
            if self.strategy_start_time:
                time_since_start = (timestamp - self.strategy_start_time).total_seconds() / 60
                if time_since_start >= self.warmup_minutes:
                    self.warmup_complete = True
                else:
                    return {
                        'desired_position': 0,
                        'counts': self._count_predictions(),
                        'warmup': True,
                        'signal_strength': 0
                    }
            else:
                return {
                    'desired_position': 0,
                    'counts': {'long': 0, 'short': 0, 'hold': 0, 'total': 0},
                    'warmup': True,
                    'signal_strength': 0
                }
        
        counts = self._count_predictions()
        diff = counts['long'] - counts['short']
        
        desired_position = 0
        
        if diff > self.threshold_open:
            desired_position = 1
        elif diff < -self.threshold_open:
            desired_position = -1
        
        if self.current_position == 1:
            if diff <= self.threshold_close:
                desired_position = 0
            else:
                desired_position = 1
        elif self.current_position == -1:
            if diff >= -self.threshold_close:
                desired_position = 0
            else:
                desired_position = -1
        
        return {
            'desired_position': desired_position,
            'counts': counts,
            'warmup': False,
            'signal_strength': diff
        }
    
    def update_position(self, position):
        """Update current position."""
        self.current_position = position
    
    def reset(self):
        """Reset strategy state."""
        self.current_candle_predictions = []
        self.current_candle_start = None
        self.current_position = 0
        self.warmup_complete = False
        self.strategy_start_time = None

def generate_signals_and_backtest(signal_df, price_df, candles_df, starting_capital, day_num):
    """Execute candle-based backtest and generate signal data."""
    
    strategy_params = CONFIG['strategy']
    
    strategy = CandleBasedStrategy(
        candle_minutes=strategy_params['candle_minutes'],
        threshold_open=strategy_params['threshold_open'],
        threshold_close=strategy_params['threshold_close'],
        warmup_minutes=strategy_params['warmup_minutes']
    )
    
    trades = []
    equity_curve = [{'timestamp': price_df.index[0], 'equity': starting_capital}]
    
    signals_output = pd.DataFrame({
        'Time': price_df.index.strftime('%H:%M:%S'),
        'Price': price_df['Price'].values,
        'BUY': 0,
        'SELL': 0,
        'EXIT': 0
    })
    
    current_capital = starting_capital
    entry_info = None
    trailing_stop_price = None
    highest_price_since_entry = None
    lowest_price_since_entry = None
    last_position_close_time = None
    
    POSITION_SIZE_PCT = 1.0
    
    signal_df = signal_df.copy()
    signal_df['candle_group'] = signal_df['timestamp'].dt.floor(f'{strategy_params["candle_minutes"]}T')
    
    position_changes = []
    
    for candle_idx, candle in candles_df.iterrows():
        candle_start = candle['timestamp']
        candle_close = candle_start + timedelta(minutes=strategy_params['candle_minutes'])
        close_price = candle['close']
        
        candle_signals = signal_df[signal_df['candle_group'] == candle_start]
        
        for _, signal in candle_signals.iterrows():
            strategy.add_prediction(signal['timestamp'], signal['predicted_class'])
        
        candle_ticks = price_df[(price_df.index >= candle_start) & (price_df.index < candle_close)]
        
        stop_hit = False
        exit_reason = None
        stop_exit_price = None
        stop_exit_time = None
        
        if entry_info is not None:
            for tick_time, tick_row in candle_ticks.iterrows():
                tick_price = tick_row['Price']
                
                time_in_position = (tick_time - entry_info['entry_time']).total_seconds()
                if time_in_position < CONFIG['risk']['min_holding_time_seconds']:
                    continue
                
                entry_price = entry_info['entry_price']
                position_type = entry_info['type']
                
                if position_type == 'long':
                    if highest_price_since_entry is None or tick_price > highest_price_since_entry:
                        highest_price_since_entry = tick_price
                        if CONFIG['risk']['trailing_stop_pct']:
                            new_trailing = highest_price_since_entry * (1 - CONFIG['risk']['trailing_stop_pct'])
                            if trailing_stop_price is None or new_trailing > trailing_stop_price:
                                trailing_stop_price = new_trailing
                    
                    if CONFIG['risk']['trailing_stop_pct'] and trailing_stop_price and tick_price <= trailing_stop_price:
                        stop_hit, exit_reason = True, 'trailing_stop'
                        stop_exit_price, stop_exit_time = tick_price, tick_time
                        break
                    elif CONFIG['risk']['stop_loss_pct'] and tick_price <= entry_price * (1 - CONFIG['risk']['stop_loss_pct']):
                        stop_hit, exit_reason = True, 'stop_loss'
                        stop_exit_price, stop_exit_time = tick_price, tick_time
                        break
                    elif CONFIG['risk']['take_profit_pct'] and tick_price >= entry_price * (1 + CONFIG['risk']['take_profit_pct']):
                        stop_hit, exit_reason = True, 'take_profit'
                        stop_exit_price, stop_exit_time = tick_price, tick_time
                        break
                
                else:  # short
                    if lowest_price_since_entry is None or tick_price < lowest_price_since_entry:
                        lowest_price_since_entry = tick_price
                        if CONFIG['risk']['trailing_stop_pct']:
                            new_trailing = lowest_price_since_entry * (1 + CONFIG['risk']['trailing_stop_pct'])
                            if trailing_stop_price is None or new_trailing < trailing_stop_price:
                                trailing_stop_price = new_trailing
                    
                    if CONFIG['risk']['trailing_stop_pct'] and trailing_stop_price and tick_price >= trailing_stop_price:
                        stop_hit, exit_reason = True, 'trailing_stop'
                        stop_exit_price, stop_exit_time = tick_price, tick_time
                        break
                    elif CONFIG['risk']['stop_loss_pct'] and tick_price >= entry_price * (1 + CONFIG['risk']['stop_loss_pct']):
                        stop_hit, exit_reason = True, 'stop_loss'
                        stop_exit_price, stop_exit_time = tick_price, tick_time
                        break
                    elif CONFIG['risk']['take_profit_pct'] and tick_price <= entry_price * (1 - CONFIG['risk']['take_profit_pct']):
                        stop_hit, exit_reason = True, 'take_profit'
                        stop_exit_price, stop_exit_time = tick_price, tick_time
                        break
        
        if stop_hit:
            shares = entry_info['shares']
            if entry_info['type'] == 'long':
                gross_pnl = (stop_exit_price - entry_info['entry_price']) * shares
            else:
                gross_pnl = (entry_info['entry_price'] - stop_exit_price) * shares
            
            entry_cost = shares * entry_info['entry_price'] * CONFIG['backtesting']['transaction_cost_rate']
            exit_cost = shares * stop_exit_price * CONFIG['backtesting']['transaction_cost_rate']
            net_pnl = gross_pnl - entry_cost - exit_cost
            current_capital += net_pnl
            
            trades.append({
                'day': day_num,
                'type': entry_info['type'],
                'entry_time': entry_info['entry_time'],
                'exit_time': stop_exit_time,
                'entry_price': entry_info['entry_price'],
                'exit_price': stop_exit_price,
                'shares': shares,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'hold_duration_sec': (stop_exit_time - entry_info['entry_time']).total_seconds(),
                'exit_reason': exit_reason
            })
            
            position_changes.append({
                'timestamp': stop_exit_time,
                'action': 'EXIT',
                'position_type': entry_info['type']
            })
            
            equity_curve.append({'timestamp': stop_exit_time, 'equity': current_capital})
            last_position_close_time = stop_exit_time
            
            entry_info = None
            trailing_stop_price = None
            highest_price_since_entry = None
            lowest_price_since_entry = None
            strategy.update_position(0)
        
        decision = strategy.evaluate_at_candle_close(candle_close)
        
        if decision['warmup']:
            continue
        
        desired_position = decision['desired_position']
        current_position = strategy.current_position
        
        if desired_position != current_position:
            
            can_close = True
            if entry_info is not None:
                time_in_position = (candle_close - entry_info['entry_time']).total_seconds()
                if time_in_position < CONFIG['risk']['min_holding_time_seconds']:
                    can_close = False
            
            if entry_info is not None and can_close:
                shares = entry_info['shares']
                if entry_info['type'] == 'long':
                    gross_pnl = (close_price - entry_info['entry_price']) * shares
                else:
                    gross_pnl = (entry_info['entry_price'] - close_price) * shares
                
                entry_cost = shares * entry_info['entry_price'] * CONFIG['backtesting']['transaction_cost_rate']
                exit_cost = shares * close_price * CONFIG['backtesting']['transaction_cost_rate']
                net_pnl = gross_pnl - entry_cost - exit_cost
                current_capital += net_pnl
                
                trades.append({
                    'day': day_num,
                    'type': entry_info['type'],
                    'entry_time': entry_info['entry_time'],
                    'exit_time': candle_close,
                    'entry_price': entry_info['entry_price'],
                    'exit_price': close_price,
                    'shares': shares,
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'hold_duration_sec': (candle_close - entry_info['entry_time']).total_seconds(),
                    'exit_reason': 'candle_signal'
                })
                
                position_changes.append({
                    'timestamp': candle_close,
                    'action': 'EXIT',
                    'position_type': entry_info['type']
                })
                
                equity_curve.append({'timestamp': candle_close, 'equity': current_capital})
                last_position_close_time = candle_close
                
                entry_info = None
                trailing_stop_price = None
                highest_price_since_entry = None
                lowest_price_since_entry = None
                strategy.update_position(0)
            
            if desired_position != 0:
                can_open = True
                entry_time = candle_close
                entry_price = close_price
                
                if last_position_close_time is not None:
                    time_since_close = (candle_close - last_position_close_time).total_seconds()
                    if time_since_close < CONFIG['risk']['min_time_between_trades_seconds']:
                        can_open = False
                        
                        potential_entry_time = last_position_close_time + timedelta(seconds=CONFIG['risk']['min_time_between_trades_seconds'])
                        
                        if potential_entry_time < candle_close:
                            future_ticks = price_df[(price_df.index >= potential_entry_time) & (price_df.index < candle_close)]
                            if not future_ticks.empty:
                                entry_time = future_ticks.index[0]
                                entry_price = future_ticks.iloc[0]['Price']
                                can_open = True
                
                if can_open and entry_info is None:
                    shares = (current_capital * POSITION_SIZE_PCT) / entry_price
                    position_type = 'long' if desired_position == 1 else 'short'
                    
                    entry_info = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'shares': shares,
                        'type': position_type
                    }
                    
                    position_changes.append({
                        'timestamp': entry_time,
                        'action': 'BUY' if position_type == 'long' else 'SELL',
                        'position_type': position_type
                    })
                    
                    if position_type == 'long':
                        highest_price_since_entry = entry_price
                        lowest_price_since_entry = None
                        if CONFIG['risk']['trailing_stop_pct']:
                            trailing_stop_price = entry_price * (1 - CONFIG['risk']['trailing_stop_pct'])
                    else:
                        lowest_price_since_entry = entry_price
                        highest_price_since_entry = None
                        if CONFIG['risk']['trailing_stop_pct']:
                            trailing_stop_price = entry_price * (1 + CONFIG['risk']['trailing_stop_pct'])
                    
                    strategy.update_position(desired_position)
    
    if entry_info:
        exit_price = candles_df['close'].iloc[-1]
        exit_time = candles_df['timestamp'].iloc[-1] + timedelta(minutes=strategy_params['candle_minutes'])
        shares = entry_info['shares']
        
        if entry_info['type'] == 'long':
            gross_pnl = (exit_price - entry_info['entry_price']) * shares
        else:
            gross_pnl = (entry_info['entry_price'] - exit_price) * shares
        
        entry_cost = shares * entry_info['entry_price'] * CONFIG['backtesting']['transaction_cost_rate']
        exit_cost = shares * exit_price * CONFIG['backtesting']['transaction_cost_rate']
        net_pnl = gross_pnl - entry_cost - exit_cost
        current_capital += net_pnl
        
        trades.append({
            'day': day_num,
            'type': entry_info['type'],
            'entry_time': entry_info['entry_time'],
            'exit_time': exit_time,
            'entry_price': entry_info['entry_price'],
            'exit_price': exit_price,
            'shares': shares,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'hold_duration_sec': (exit_time - entry_info['entry_time']).total_seconds(),
            'exit_reason': 'EOD'
        })
        
        position_changes.append({
            'timestamp': exit_time,
            'action': 'EXIT',
            'position_type': entry_info['type']
        })
        
        equity_curve.append({'timestamp': exit_time, 'equity': current_capital})
    
    for change in position_changes:
        time_str = change['timestamp'].strftime('%H:%M:%S')
        mask = signals_output['Time'] == time_str
        
        if mask.any():
            idx = signals_output[mask].index[0]
        else:
            time_diffs = abs(pd.to_datetime(signals_output['Time'], format='%H:%M:%S') - 
                           pd.to_datetime(time_str, format='%H:%M:%S'))
            idx = time_diffs.idxmin()
        
        if change['action'] == 'BUY':
            signals_output.loc[idx, 'BUY'] = 1
        elif change['action'] == 'SELL':
            signals_output.loc[idx, 'SELL'] = 1
        elif change['action'] == 'EXIT':
            signals_output.loc[idx, 'EXIT'] = 1
    
    return trades, equity_curve, signals_output

def load_feature_names(path: str) -> list:
    """Load feature names saved during training."""
    try:
        with open(path, 'r') as f:
            feature_names = [line.strip() for line in f if line.strip()]
        print(f"✓ Loaded {len(feature_names)} feature names from {path}")
        return feature_names
    except Exception as e:
        print(f"✗ Error loading feature names: {e}")
        return None

def process_day_with_signals(x_file, idx_file, price_file, csv_path, bst, 
                             day_index, starting_capital, feature_names):
    """Process a single day with candle-based strategy and generate signals."""
    X_day = np.load(x_file)
    prediction_indices = np.load(idx_file)
    prediction_prices = np.load(price_file)
    
    total_samples = X_day.shape[0]
    
    y_pred_class = np.zeros(total_samples, dtype=np.int32)
    max_proba = np.zeros(total_samples, dtype=np.float32)
    
    batch_size = CONFIG['testing']['prediction_batch_size']
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        
        X_batch = X_day[start_idx:end_idx]
        dtest_batch = xgb.DMatrix(X_batch, feature_names=feature_names)
        
        try:
            y_proba_batch = bst.predict(dtest_batch, output_margin=False)
            
            if y_proba_batch.ndim == 2:
                y_pred_batch = np.argmax(y_proba_batch, axis=1)
                confidences = np.max(y_proba_batch, axis=1)
            else:
                y_pred_batch = y_proba_batch.astype(np.int32)
                confidences = np.full_like(y_pred_batch, CONFIG['testing']['confidence_threshold'] + 0.05, dtype=np.float32)
        except Exception as e:
            print(f"\n  ⚠ Prediction Error: {e}")
            y_pred_batch = bst.predict(dtest_batch).astype(np.int32)
            confidences = np.full_like(y_pred_batch, CONFIG['testing']['confidence_threshold'] + 0.05, dtype=np.float32)
        
        y_pred_class[start_idx:end_idx] = y_pred_batch
        max_proba[start_idx:end_idx] = confidences
        
        del X_batch, dtest_batch
        gc.collect()
    
    del X_day
    gc.collect()
    
    df_day = pd.read_csv(csv_path)
    if 'Time' in df_day.columns:
        df_day['Time'] = pd.to_datetime(df_day['Time'])
        df_day.set_index('Time', inplace=True)
    
    candles_df = create_candles(df_day, CONFIG['strategy']['candle_minutes'])
    
    is_whipsaw_day = False
    whipsaw_info = {}
    
    if CONFIG['strategy']['whipsaw_detection']:
        is_whipsaw, or_high, or_low, details = detect_whipsaw_day(
            candles_df, 
            observation_minutes=CONFIG['strategy']['whipsaw_observation_minutes'],
            opening_range_candles=CONFIG['strategy']['opening_range_candles']
        )
        is_whipsaw_day = is_whipsaw
        whipsaw_info = {
            'is_whipsaw': is_whipsaw,
            'opening_range_high': or_high,
            'opening_range_low': or_low,
            'details': details
        }
    
    if is_whipsaw_day:
        return [], [{'timestamp': df_day.index[0], 'equity': starting_capital}], whipsaw_info, None
    
    signal_df = pd.DataFrame({
        'price_idx': prediction_indices,
        'predicted_class': y_pred_class,
        'confidence': max_proba,
        'timestamp': df_day.index[prediction_indices].values,
        'price': df_day['Price'].iloc[prediction_indices].values
    })
    
    signal_df_filtered = signal_df[signal_df['confidence'] >= CONFIG['testing']['confidence_threshold']].copy()
    
    if len(signal_df_filtered) == 0:
        return [], [{'timestamp': df_day.index[0], 'equity': starting_capital}], whipsaw_info, None
    
    day_trades, day_equity, signals_output = generate_signals_and_backtest(
        signal_df_filtered[['timestamp', 'predicted_class', 'price']],
        df_day,
        candles_df,
        starting_capital,
        day_index
    )
    
    del signal_df, signal_df_filtered, df_day
    gc.collect()
    
    return day_trades, day_equity, whipsaw_info, signals_output

def generate_signals(args):
    """Generate signals for all test days."""
    print("="*80)
    print("SIGNAL GENERATION WITH CANDLE STRATEGY")
    print("="*80)
    
    feature_names = load_feature_names(CONFIG['paths']['feature_names_file'])
    if feature_names is None:
        print("✗ Cannot proceed without feature names")
        return None
    
    if args.days: 
        test_days_file = Path(args.days)
    else:
        test_days_file = Path(CONFIG['paths']['train_data_dir']) / Path(CONFIG['paths']['test_days_file'])
    if not test_days_file.exists():
        print(f"✗ Test days file not found: {test_days_file}")
        return None
    with open(test_days_file, 'r') as f:
        test_day_names = [line.strip() for line in f if line.strip()]
    
    test_days = [int(name.replace('day', '')) for name in test_day_names]
    
    try:
        bst = xgb.Booster()
        bst.load_model(CONFIG['paths']['model_file'])
        print(f"✓ Loaded model: {CONFIG['paths']['model_file']}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    sequence_path = Path(CONFIG['paths']['test_data_dir'])
    signals_path = Path(CONFIG['paths']['signals_dir'])
    signals_path.mkdir(exist_ok=True, parents=True)
    
    print(f"✓ Processing {len(test_days)} test days")
    print("="*80)
    
    all_trades = []
    daily_results = []
    cumulative_capital = CONFIG['backtesting']['initial_capital']
    
    whipsaw_days = 0
    tradable_days = 0
    
    for day_index in tqdm(test_days, desc="Processing Days"):
        x_file = sequence_path / f'X_test_day{day_index:03d}.npy'
        idx_file = sequence_path / f'indices_day{day_index:03d}.npy'
        price_file = sequence_path / f'prices_day{day_index:03d}.npy'
        csv_path = Path(CONFIG['paths']['data_dir']) / f'day{day_index}.csv'
        
        if not x_file.exists() or not idx_file.exists() or not price_file.exists() or not csv_path.exists():
            print(f"\n  ✗ Missing files for day {day_index}")
            continue
        
        try:
            day_trades, day_equity, whipsaw_info, signals_output = process_day_with_signals(
                x_file, idx_file, price_file, csv_path, bst, day_index, 
                cumulative_capital, feature_names
            )
            
            if whipsaw_info.get('is_whipsaw', False):
                whipsaw_days += 1
                print(f"\n  Day {day_index}: WHIPSAW - SKIPPED")
                
                daily_results.append({
                    'day': day_index,
                    'trades': 0,
                    'pnl': 0.0,
                    'ending_capital': cumulative_capital,
                    'status': 'whipsaw'
                })
                continue
            
            tradable_days += 1
            
            if signals_output is not None:
                signal_file = signals_path / f'day{day_index}_signals.csv'
                signals_output.to_csv(signal_file, index=False)
            
            all_trades.extend(day_trades)
            
            if day_equity:
                cumulative_capital = day_equity[-1]['equity']
            
            day_pnl = sum(t['net_pnl'] for t in day_trades)
            
            daily_results.append({
                'day': day_index,
                'trades': len(day_trades),
                'pnl': day_pnl,
                'ending_capital': cumulative_capital,
                'status': 'traded'
            })
            
            pnl_color = "+" if day_pnl >= 0 else ""
            print(f"  Day {day_index} PnL: {pnl_color}${day_pnl:,.2f} | Trades: {len(day_trades)} | Equity: ${cumulative_capital:,.2f}")
            
        except Exception as e:
            print(f"\n  ✗ Error processing day {day_index}: {e}")
            daily_results.append({
                'day': day_index,
                'trades': 0,
                'pnl': 0.0,
                'ending_capital': cumulative_capital,
                'status': 'error'
            })
        
        gc.collect()
    
    return {
        'all_trades': all_trades,
        'daily_results': daily_results,
        'cumulative_capital': cumulative_capital,
        'whipsaw_days': whipsaw_days,
        'tradable_days': tradable_days
    }

# =============================================================================
# PERFORMANCE METRICS MODULE
# =============================================================================

def calculate_performance_metrics(trades_df, daily_results_df, initial_capital):
    """Calculate comprehensive performance metrics."""
    metrics = {}
    
    # Basic metrics
    total_pnl = daily_results_df['pnl'].sum()
    total_return = ((daily_results_df['ending_capital'].iloc[-1] - initial_capital) / initial_capital) * 100
    
    metrics['total_pnl'] = total_pnl
    metrics['total_return_pct'] = total_return
    metrics['initial_capital'] = initial_capital
    metrics['final_capital'] = daily_results_df['ending_capital'].iloc[-1]
    
    # Daily metrics
    traded_days = daily_results_df[daily_results_df['status'] == 'traded']
    metrics['total_days'] = len(daily_results_df)
    metrics['traded_days'] = len(traded_days)
    metrics['winning_days'] = len(traded_days[traded_days['pnl'] > 0])
    metrics['losing_days'] = len(traded_days[traded_days['pnl'] < 0])
    metrics['daily_win_rate'] = (metrics['winning_days'] / metrics['traded_days'] * 100) if metrics['traded_days'] > 0 else 0
    metrics['avg_daily_pnl'] = traded_days['pnl'].mean() if len(traded_days) > 0 else 0
    
    # Trade metrics
    if len(trades_df) > 0:
        metrics['total_trades'] = len(trades_df)
        metrics['winning_trades'] = len(trades_df[trades_df['net_pnl'] > 0])
        metrics['losing_trades'] = len(trades_df[trades_df['net_pnl'] < 0])
        metrics['trade_win_rate'] = (metrics['winning_trades'] / metrics['total_trades'] * 100)
        
        if metrics['winning_trades'] > 0:
            metrics['avg_win'] = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean()
            metrics['max_win'] = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].max()
        else:
            metrics['avg_win'] = 0
            metrics['max_win'] = 0
            
        if metrics['losing_trades'] > 0:
            metrics['avg_loss'] = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean()
            metrics['max_loss'] = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].min()
        else:
            metrics['avg_loss'] = 0
            metrics['max_loss'] = 0
        
        # Profit factor
        total_wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
        total_losses = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Average holding time
        metrics['avg_hold_duration_sec'] = trades_df['hold_duration_sec'].mean()
        metrics['avg_hold_duration_min'] = metrics['avg_hold_duration_sec'] / 60
    else:
        metrics['total_trades'] = 0
        metrics['winning_trades'] = 0
        metrics['losing_trades'] = 0
        metrics['trade_win_rate'] = 0
        metrics['avg_win'] = 0
        metrics['max_win'] = 0
        metrics['avg_loss'] = 0
        metrics['max_loss'] = 0
        metrics['profit_factor'] = 0
        metrics['avg_hold_duration_sec'] = 0
        metrics['avg_hold_duration_min'] = 0
    
    # Sharpe Ratio (annualized)
    if len(traded_days) > 1:
        daily_returns = traded_days['pnl'] / traded_days['ending_capital'].shift(1)
        daily_returns = daily_returns.dropna()
        
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            metrics['sharpe_ratio'] = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
    else:
        metrics['sharpe_ratio'] = 0
    
    # Maximum Drawdown
    equity_curve = traded_days['ending_capital'].values
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    metrics['max_drawdown_pct'] = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
    
    # Calmar Ratio
    if metrics['max_drawdown_pct'] > 0:
        metrics['calmar_ratio'] = total_return / metrics['max_drawdown_pct']
    else:
        metrics['calmar_ratio'] = 0
    
    return metrics

def print_performance_summary(results):
    """Print comprehensive performance summary."""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    if not results or 'daily_results' not in results:
        print("No results to display")
        return
    
    daily_results_df = pd.DataFrame(results['daily_results'])
    
    print(f"\nDAILY RESULTS:")
    print("-"*80)
    for _, row in daily_results_df.iterrows():
        if row['status'] == 'traded':
            pnl_sign = "+" if row['pnl'] >= 0 else ""
            print(f"Day {row['day']:3d}: {pnl_sign}${row['pnl']:>10,.2f} | Trades: {row['trades']:>3d} | Equity: ${row['ending_capital']:>12,.2f}")
        else:
            print(f"Day {row['day']:3d}: {row['status'].upper():>12} | Equity: ${row['ending_capital']:>12,.2f}")
    
    print("\n" + "="*80)
    
    if len(results['all_trades']) > 0:
        trades_df = pd.DataFrame(results['all_trades'])
        metrics = calculate_performance_metrics(trades_df, daily_results_df, CONFIG['backtesting']['initial_capital'])
        
        print(f"OVERALL STATISTICS:")
        print("-"*80)
        print(f"  Initial Capital:              ${metrics['initial_capital']:>15,.2f}")
        print(f"  Final Equity:                 ${metrics['final_capital']:>15,.2f}")
        print(f"  Total PnL:                    ${metrics['total_pnl']:>15,.2f}")
        print(f"  Total Return:                 {metrics['total_return_pct']:>15.2f}%")
        print(f"  Average Daily PnL:            ${metrics['avg_daily_pnl']:>15,.2f}")
        print(f"  Days Traded:                  {metrics['traded_days']:>15d}")
        print(f"  Winning Days:                 {metrics['winning_days']:>15d}")
        print(f"  Losing Days:                  {metrics['losing_days']:>15d}")
        print(f"  Daily Win Rate:               {metrics['daily_win_rate']:>15.2f}%")
        print(f"  Days Skipped (Whipsaw):       {results['whipsaw_days']:>15d}")
        
        print(f"\nTRADE STATISTICS:")
        print("-"*80)
        print(f"  Total Trades:                 {metrics['total_trades']:>15d}")
        print(f"  Winning Trades:               {metrics['winning_trades']:>15d}")
        print(f"  Losing Trades:                {metrics['losing_trades']:>15d}")
        print(f"  Trade Win Rate:               {metrics['trade_win_rate']:>15.2f}%")
        print(f"  Average Win:                  ${metrics['avg_win']:>15,.2f}")
        print(f"  Average Loss:                 ${metrics['avg_loss']:>15,.2f}")
        print(f"  Max Win:                      ${metrics['max_win']:>15,.2f}")
        print(f"  Max Loss:                     ${metrics['max_loss']:>15,.2f}")
        print(f"  Profit Factor:                {metrics['profit_factor']:>15.2f}")
        print(f"  Avg Hold Duration:            {metrics['avg_hold_duration_min']:>15.2f} min")
        
        print(f"\nRISK METRICS:")
        print("-"*80)
        print(f"  Sharpe Ratio (Annualized):    {metrics['sharpe_ratio']:>15.2f}")
        print(f"  Maximum Drawdown:             {metrics['max_drawdown_pct']:>15.2f}%")
        print(f"  Calmar Ratio:                 {metrics['calmar_ratio']:>15.2f}")
    
    print("\n" + "="*80)

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def train_pipeline(args):
    """Execute training pipeline."""
    print("\n" + "="*80)
    print("TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Prepare training data
    print("\n[Step 1/2] Preparing Training Data...")
    prepare_train_data(args.days)
    
    # Step 2: Train model
    print("\n[Step 2/2] Training Model...")
    train_model()
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE")
    print("="*80)

def test_pipeline(args):
    """Execute testing pipeline."""
    print("\n" + "="*80)
    print("TESTING PIPELINE")
    print("="*80)
    
    # Determine test days file
    if args.days:
        test_days_file = args.days
    else:
        test_days_file = Path(CONFIG['paths']['train_data_dir']) / CONFIG['paths']['test_days_file']
        if not test_days_file.exists():
            print(f"\nError: Test days file not found at {test_days_file}")
            print("Please run training first or provide --days argument")
            return
        
    # Step 1: Prepare test data
    print("\n[Step 1/3] Preparing Test Data...")
    prepare_test_data(test_days_file)
    
    # Step 2: Generate signals
    print("\n[Step 2/3] Generating Signals...")
    results = generate_signals(args)
    
    # Step 3: Print results
    print("\n[Step 3/3] Results...")
    print_performance_summary(results)
    
    print("\n" + "="*80)
    print("TESTING PIPELINE COMPLETE")
    print("="*80)
    print(f"\nSignal files saved to: {Path(CONFIG['paths']['signals_dir']).resolve()}")
    print("="*80)
def load_test_days(filepath: str):
    """Load test days from text file."""
    test_days = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('day'):
                day_num = int(line.replace('day', ''))
                test_days.append(day_num)
    return sorted(test_days)

def load_signals(day: int, signals_dir: str):
    """Load signal CSV for a specific day and create time-indexed dictionary."""
    signal_file = Path(signals_dir) / f'day{day}_signals.csv'
    
    if not signal_file.exists():
        raise FileNotFoundError(f"Signal file not found: {signal_file}")
    
    df = pd.read_csv(signal_file)
    print(f"  Loaded {len(df)} signals from {signal_file.name}")
    
    # Create dictionary with Time as key
    signals_dict = {}
    for idx, row in df.iterrows():
        # Ensure Time is string for dictionary key lookup
        time_key = str(row['Time']) 
        signals_dict[time_key] = {
            'BUY': int(row['BUY']),
            'SELL': int(row['SELL']),
            'EXIT': int(row['EXIT']),
            'Price': float(row['Price'])
        }
    
    buy_count = df['BUY'].sum()
    sell_count = df['SELL'].sum()
    exit_count = df['EXIT'].sum()
    print(f"    BUY: {buy_count}, SELL: {sell_count}, EXIT: {exit_count}")
    
    return df, signals_dict

# Callbacks
def signal_broadcast_callback(state, ts_str):
    """Broadcast callback that reads signals from CSV using dictionary lookup."""
    global backtest, signals_dict
    
    TICKER_NAME = CONFIG['paths']['signals_dir']
    
    curr_pos = backtest.position_map.get(TICKER_NAME, 0)
    
    if ts_str in signals_dict:
        signal_data = signals_dict[ts_str]
        
        buy = signal_data['BUY']
        sell = signal_data['SELL']
        exit_signal = signal_data['EXIT']
        price = signal_data['Price']
        
        if exit_signal == 1:
            if curr_pos == 1:
                # print(f"[{ts_str}] EXIT LONG {TICKER_NAME} @ {price:.2f}")
                backtest.place_order(TICKER_NAME, 1, Side.SELL)
            elif curr_pos == -1:
                # print(f"[{ts_str}] EXIT SHORT {TICKER_NAME} @ {price:.2f}")
                backtest.place_order(TICKER_NAME, 1, Side.BUY)
        
        elif buy == 1 and curr_pos == 0:
            # print(f"[{ts_str}] OPEN LONG {TICKER_NAME} @ {price:.2f}")
            backtest.place_order(TICKER_NAME, 1, Side.BUY)
        
        elif sell == 1 and curr_pos == 0:
            # print(f"[{ts_str}] OPEN SHORT {TICKER_NAME} @ {price:.2f}")
            backtest.place_order(TICKER_NAME, 1, Side.SELL)

def on_timer(ts):
    """Timer callback - kept minimal for now."""
    # print(f"\n[TIMER] Timestamp={ts}")
    pass

def plot_day_results(day: int, signals_df: pd.DataFrame, pm: PositionManager, plots_dir: Path):
    """Plot day results."""
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    TICKER_NAME = CONFIG['paths']['signals_dir']
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    fig.suptitle(f'Day {day} - Backtest Results', fontsize=18, fontweight='bold')
    
    # Plot 1: Price with signals
    ax1 = axes[0]
    ax1.plot(range(len(signals_df)), signals_df['Price'], 
             linewidth=1.5, color='black', label='Price')
    
    buy_signals = signals_df[signals_df['BUY'] == 1]
    if not buy_signals.empty:
        ax1.scatter(buy_signals.index, buy_signals['Price'], 
                    c='green', marker='^', s=200, edgecolors='black', 
                    linewidths=2, label='Buy', zorder=5)
    
    sell_signals = signals_df[signals_df['SELL'] == 1]
    if not sell_signals.empty:
        ax1.scatter(sell_signals.index, sell_signals['Price'], 
                    c='red', marker='v', s=200, edgecolors='black', 
                    linewidths=2, label='Sell', zorder=5)
    
    exit_signals = signals_df[signals_df['EXIT'] == 1]
    if not exit_signals.empty:
        ax1.scatter(exit_signals.index, exit_signals['Price'], 
                    c='orange', marker='x', s=200, linewidths=3, 
                    label='Exit', zorder=5)
    
    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.set_title('Price & Trading Signals', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signal counts (simplified)
    ax2 = axes[1]
    window_size = 20
    buy_rolling = signals_df['BUY'].rolling(window=window_size).sum()
    sell_rolling = signals_df['SELL'].rolling(window=window_size).sum()
    
    ax2.plot(range(len(buy_rolling)), buy_rolling, color='green', linewidth=2, label='Buy Rolling')
    ax2.plot(range(len(sell_rolling)), sell_rolling, color='red', linewidth=2, label='Sell Rolling')
    
    ax2.set_ylabel('Signal Count', fontsize=12, fontweight='bold')
    ax2.set_title('Rolling Signal Counts', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: PnL curve
    ax3 = axes[2]
    if pm.position_map and TICKER_NAME in pm.position_map:
        pos = pm.position_map[TICKER_NAME]
        if pos.pnl_list:
            ax3.plot(pos.pnl_list, linewidth=2, color='blue', label='PnL')
            ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            
            final_pnl = pos.pnl_list[-1]
            pnl_color = 'green' if final_pnl >= 0 else 'red'
            ax3.text(0.02, 0.98, f'Final PnL: %{final_pnl:,.2f}', 
                     transform=ax3.transAxes, fontsize=12, fontweight='bold',
                     verticalalignment='top', color=pnl_color,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Tick', fontsize=12, fontweight='bold')
    ax3.set_ylabel('PnL (%)', fontsize=12, fontweight='bold')
    ax3.set_title('PnL Curve', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'day_{day:03d}_backtest.png', dpi=150, bbox_inches='tight')
    plt.close()

# Main pipeline runner
def backtest_ebullient_pipeline(args):
    """Run backtest using signal CSVs from the testing phase."""
    global backtest, current_signals_df, signals_dict
    
    # Load configuration parameters
    TEST_DAYS_FILE = CONFIG['paths']['test_days_file']
    SIGNALS_DIR = CONFIG['paths']['signals_dir']
    PLOTS_DIR = CONFIG['paths']['plots_dir']
    TCOST = CONFIG['backtest']['tcost']
    TICKER_NAME = CONFIG['backtest']['ticker_name']
    DATA_ROOT_PATH_FOR_BACKTESTER = '.' 
    DATA_ROOT_PATH_FOR_CHECK = CONFIG['paths']['data_dir']
    
    print("="*80)
    print("BACKTEST RUNNER - Using Signal CSVs (Ebullient Strategy)")
    print("="*80)
    signals_path = Path(SIGNALS_DIR)
    test_days = []

    if signals_path.exists():
        for file_path in signals_path.glob("day*_signals.csv"):
            try:
                day_str = file_path.stem.split('_')[0].lstrip('day')
                test_days.append(int(day_str))
            except ValueError:
                print(f"Warning: Skipping file with non-numeric day identifier: {file_path.name}")
        test_days.sort()
        if test_days:
            print(f"\n loaded {len(test_days)} test days from {signals_path} folder.")
        else:
            print(f"\n Error: Found signals folder, but no signal files matching 'day*_signals.csv' inside {signals_path}. Did the 'test' command run correctly?")
    else:
        print(f"\n Error: Signals directory {signals_path} not found. Please run the 'test' command first.")
        test_days = []
        
    plots_dir = Path(PLOTS_DIR) / "ebullient_backtest" # Create a subfolder for these plots
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n✓ Signals Directory: {Path(SIGNALS_DIR).resolve()}")
    print(f"✓ Output Directory: {plots_dir.resolve()}")
    print(f"✓ Ticker: {TICKER_NAME}, TCost: {TCOST}")
    print("="*80)
    
    final_report = ResultReport()
    total_pnl = 0.0
    successful_days = 0
    failed_days = 0
    
    for day in test_days:
        print(f"\n{'='*80}")
        print(f"Processing Day {day}")
        print(f"{'='*80}")
        
        config_file = f"temp_config_day_{day}.json"
        
        try:
            current_signals_df, signals_dict = load_signals(day, SIGNALS_DIR)
            
            # --- 1. Create temporary configuration file ---
            config = {
                "data_path": DATA_ROOT_PATH_FOR_BACKTESTER,
                "start_date": day,
                "end_date": day,
                "timer": CONFIG['backtest']['timer_seconds'],
                "tcost": TCOST,
                "broadcast": [SIGNALS_DIR]
            }
            with open(config_file, 'w') as f:
                json.dump(config, f)
            
            backtest = BacktesterIIT(config_file)
            
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Starting backtest...")
            
            backtest.run(
                broadcast_callback=signal_broadcast_callback,
                timer_callback=on_timer
            )
            
            day_pnl = sum(pos.net_realised_pnl 
                          for pos in backtest.position_manager.position_map.values())
            
            total_pnl += day_pnl
            successful_days += 1
            
            pnl_sign = "+" if day_pnl >= 0 else ""
            print(f"\n  Day {day} PnL: {pnl_sign}${day_pnl:,.2f}")
            print(f"  Cumulative PnL: ${total_pnl:,.2f}")
            
            backtest.position_manager.print_details()
            
            plot_day_results(day, current_signals_df, backtest.position_manager, plots_dir)
            
            final_report.update(backtest.position_manager, day)
            
        except FileNotFoundError as e:
            print(f"  ✗ Signal file not found for day {day}: {e}")
            failed_days += 1
        except Exception as e:
            print(f"  ✗ Error processing day {day}: {e}")
            import traceback
            traceback.print_exc()
            failed_days += 1
        finally:
            if os.path.exists(config_file):
                os.remove(config_file)
            
            # Cleanup for next iteration
            current_signals_df = None
            signals_dict = {}
            backtest = None
            gc.collect()
    
    print(f"\n{'='*80}")
    print("DETAILED REPORT")
    print(f"{'='*80}")
    print(final_report.generate_report())
    
    print(f"\n{'='*80}")
    print(f"✓ Plots saved to: {plots_dir.resolve()}")
    print(f"{'='*80}")

# =============================================================================
# MAIN EXECUTION (Updated to include backtest_ebullient command)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EBX.py - Unified Trading System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--days', required=True,
                             help='Number of days (int) or path to train_days_EBX.txt file')
    train_parser.add_argument('--config', required=True,
                             help='Path to configuration JSON file')
    
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('--days', default=None,
                            help='Path to test_days_EBX.txt file (optional, uses default if not provided)')
    test_parser.add_argument('--config', required=True,
                            help='Path to configuration JSON file')

    backtest_parser = subparsers.add_parser('backtest_ebullient', 
                                            help='Run backtest using signals generated by the test command.')
    backtest_parser.add_argument('--config', required=True,
                                 help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)

    load_config(args.config)

    if args.command == 'train':
        train_pipeline(args) 
    elif args.command == 'test':
        test_pipeline(args) 
    elif args.command == 'backtest_ebullient':
        backtest_ebullient_pipeline(args)
