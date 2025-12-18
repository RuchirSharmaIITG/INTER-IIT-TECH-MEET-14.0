# RL Trading System - Complete Guide

### Note : The startegy returns depends on the random selection of days. In some combinations few days with very high return might get included (like day87 in EBX and day104 in EBY) resulting in higher returns in some runs, while lower in other runs in which these days are not included in testing. For different runs change the seed in PARAMS['SEED'] to get different results.

## Quick Start 
### Install Dependencies
```bash
pip install numpy pandas gymnasium stable-baselines3 torch tqdm matplotlib
```

### Prepare Data
Place your raw tick data CSV files in a folder (e.g., `EBX/`) or Specify the dataset folder in the PARAMS['SOURCE_FOLDER'] in the code:
```
EBX/
├── day1.csv
├── day2.csv
└── day3.csv
```

## Commands Explained

### Command 1: `python <Ticker>.py train`

**What Happens:**

1. **Data Resampling** (2-3 mins)
   - Reads tick data from `EBX/` folder
   - Converts to 2-minute OHLC candles
   - Saves to `EBX_2min/` (skips if already exists)
   - Creates `train_days_EBX.txt` and `test_days_EBX.txt` by randomly selecting days

2. **Indicator Calculation** (1 min)
   - Precomputes 60+ technical indicators for ALL training days
   - Applies 30-minute warmup window (discards first 30 mins of each day)

3. **Model Training** (5-10 mins depending on CPU/GPU)
   - Launches parallel environments
   - Trains PPO model
   - Prints training progress with tqdm bar
   - Monitors: entropy loss, explained variance, policy loss
   - GPU auto-detects and uses if available

4. **Model Saving** 
   - Saves trained model: `Models_EBX/ppo_trading_model_EBX.zip`
   - Saves normalization stats: `Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl`
   - Generates training plots: `training_plots/EBX_training_metrics.png`
   - Generates feature info: `feature_info_EBX.txt`

---

### Command 2: `python <Ticker>.py test`

**What Happens:**

1. **Model Loading** 
   - Loads trained model from `Models_EBX/ppo_trading_model_EBX.zip`
   - Loads normalization stats from `Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl`
   - Verifies both files exist

2. **Per-Day Backtesting** 
   - For each test day:
     - Loads 2-min candle data
     - Calculates indicators (with 30-min warmup)
     - Records every trade entry/exit with price and timestamp
     - Calculates daily P&L in basis points (bps)
     - Generates signals (BUY, SELL, EXIT)

3. **Output Generation** 
   - Saves all signals to CSV: `signals_EBX/day123.csv`
   - Generates price charts: `test_trade_plots/EBX_day_1_day123.png`
   - Calculates equity curve and drawdown
   - Saves equity plot: `test_results/EBX_equity_drawdown.png`
   - Writes report: `test_results/test_results_EBX.txt`
   - **Prints to console:** Trade log with timestamps, prices, positions

**Expected Bugs & Solutions:**

| Bug | Cause | Solution |
|-----|-------|----------|
| "VecNormalize file not found" | Didn't run train command first | Run `python <Ticker>.py train` first |
| All trades losing | Model overtrained on train set (overfitting) | Train on more diverse data or reduce training episodes |
| 0 trades executed | Model learned to always hold | Increase `TRADE_ENTRY_PENALTY` (currently -5) or check reward scaling |

---

### Command 3: `python <Ticker>.py test 123`

**What Happens:**

1. **Specific Day Filtering** 
   - Searches for `day123` in test file list
   - Only tests that single day (not all test days)
   - Useful for debugging specific days

---

### Command 4: `python <Ticker>.py backtest_ebullient`

**What Happens:**

1. **Backtest Execution** 
   - Initializes BacktesterIIT with config
   - Runs Ebullient's market simulator
   - For each signal:
     - EXIT signal → Closes position
     - BUY signal → Opens long (100 shares)
     - SELL signal → Opens short (100 shares)
   - Prints backtest results

**Expected Bugs & Solutions:**

| Bug | Cause | Solution |
|-----|-------|----------|
| "day(\d+)" regex error | Signal file naming doesn't match pattern | Check files are named like `day1.csv`, `day2.csv` (not `day_1.csv`) |
| Config file error | JSON formatting issue | Manually inspect `config.json` created in root |

---

## Common Issues & Global Solutions

### Issue 1: "PARAMS mismatch between training and testing"
**Problem:** You changed stop loss/trailing stop but didn't retrain
**Solution:** 
- Training uses `STOP_LOSS_TR`, `TRAIL_PCT_TR`
- Testing uses `STOP_LOSS_TE`, `TRAIL_PCT_TE` (can be different!)
- Model learns exits based on TRAINING params
- Testing params determine what exits are ENFORCED during test
- If you change testing params, results will differ (but model hasn't relearned)

### Issue 2: "Too many/too few trades"
**Problem:** Model behavior doesn't match expectations
**Solutions:**
- Too many: Increase `TRADE_ENTRY_PENALTY` (currently -5) to -10 or -15
- Too few: Decrease `TRADE_ENTRY_PENALTY` to 0 or -2
- Too many stops: Decrease `STOP_LOSS` from -0.0004 to -0.0002
- Retrain after changing parameters

---

## File Checklist After Running

```
After train:
✓ Models_EBX/ppo_trading_model_EBX.zip (2-5MB)
✓ Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl (100KB)
✓ feature_info_EBX.txt (50KB)
✓ train_days_EBX.txt (list of days)
✓ test_days_EBX.txt (list of days)
✓ training_plots/EBX_training_metrics.png (chart)
✓ EBX_2min/ (folder with 2-min candles - auto-created)

After test:
✓ test_results/test_results_EBX.txt (report)
✓ test_results/EBX_equity_drawdown.png (equity chart)
✓ test_trade_plots/ (folder with per-day charts)
✓ signals_EBX/ (folder with signal CSVs)
```



