# Trading System Code Guide

This guide provides step-by-step instructions for running the complete trading system pipeline, from training models to backtesting strategies on both EBX and EBY datasets. Before starting make sure to have the dataset folder 'EBX' in 'FINAL_SUB_X' folder and 'EBY'and'EBX' both in the 'FINAL_SUB_Y' folder .
Also run:

```bash
pip install requirements.txt
```

to install the neccessary libraries.


## Quick Start: Getting Backtest Results First

If you want to see backtest results immediately using pre-trained models and pre-generated signals for both EBY and EBX:

```bash
python Backtest_EBX_EBY.py config.json
```

This runs the **combined backtest** using signals from both `FINAL_SUB_X/SIGNALS_EBX` and `FINAL_SUB_Y/SIGNALS_EBY` as specified in `config.json`.

**For individual strategy backtests:**

```bash
# EBX backtest with ebullient strategy
cd FINAL_SUB_X
python EBX.py backtest_ebullient --config config_EBX.json

# EBY backtest with ebullient strategy
cd FINAL_SUB_Y
python EBY.py backtest_ebullient --config config_EBY.json~
```

### What Happens:
1. Loads pre-generated signal CSV files from `SIGNALS_EBX/` or `SIGNALS_EBY/`
2. Executes trades based on BUY/SELL/EXIT signals
3. Calculates PnL, Ratios and performance metrics
4. Generates plots and detailed reports
5. Outputs results to `PHOTOS_EBX/ebullient_backtest/` or `PHOTOS_EBY/ebullient_backtest/`

---

## Understanding the System

**Strategy Flow:**
```
Raw Data → Feature Engineering → Model Training → 
Signal Generation → Candle-Based Strategy → Backtesting
```

**Candle-Based Strategy:**
- Aggregates tick-level predictions into candle periods (20min for EBX, 25min for EBY)
- Counts long/short predictions within each candle
- Opens positions when signal strength exceeds threshold
- Closes positions based on opposite signals or risk management rules

---

## Complete Pipeline Walkthrough

### Step 1: Train the Model

Training builds an XGBoost model that predicts trading signals from engineered features.

**For EBX:**
```bash
python EBX.py train --days MODELS/train_days_EBX.txt --config config_EBX.json
or 
python EBX.py train --days 250 --config config_EBX.json
```

**For EBY:**
```bash
python EBY.py train --days MODELS/train_days_EBY.txt --config config_EBY.json
or 
python EBY.py train --days 250 --config config_EBY.json
```

**What Happens:**

1. **Data Preparation (`prepare_train_data`)**
   - Loads raw CSV files from `EBX/` or `EBY/` directory
   - For EBY: Randomly selects from EBY days 0-278, supplements with EBX if needed
   - For EBX: Randomly selects from EBX days 0-509
   - Applies feature engineering pipeline:
     - **Time features**: Cyclical encoding (hour/minute sin/cos)
     - **Kalman filter**: Smooths price, generates slope-based signals
     - **Regime detection**: CUSUM algorithm for market state
     - **Custom features**: Composite indicators from price/volume bands
   - Creates sequences with lookback window (30 timesteps)
   - Balances classes using hold_ratio sampling
   - Saves preprocessed arrays to `train_data_EBX/` or `train_data_EBY/`
   - **Outputs:**
     - `train_data_EBX/X_day*.npy` - Feature sequences
     - `train_data_EBX/y_day*.npy` - Label sequences  
     - `train_data_EBX/train_days_EBX.txt` - List of training days
     - `train_data_EBX/test_days_EBX.txt` - Remaining days for testing

2. **Model Training (`train_model`)**
   - Loads preprocessed data in batches (memory-efficient)
   - Fits RobustScaler on first batch
   - Flattens 3D sequences (samples, timesteps, features) → 2D (samples, timesteps×features)
   - Splits data 80/20 for train/validation
   - Trains XGBoost classifier (3 classes: Hold, Long, Short)
   - Uses early stopping to prevent overfitting
   - **Outputs:**
     - `MODELS/xgb_model_EBX.json` or `MODELS/xgb_model_EBY.json` - Trained model
     - `MODELS/scaler_EBX.pkl` or `MODELS/scaler_EBY.pkl` - Fitted scaler
     - `MODELS/features_EBX.txt` or `MODELS/features_EBY.txt` - Feature names

---

### Step 2: Generate Test Signals

Testing generates trading signals on unseen data using the trained model.

**For EBX:**
```bash
python EBX.py test --days MODELS/test_days_EBX.txt --config config_EBX.json
or 
python EBX.py test --config config_EBX.json  # if already ran train
```
**For EBY:**
```bash
python EBY.py test --days MODELS/test_days_EBY.txt --config config_EBY.json
or 
python EBY.py test --config config_EBY.json  # if already ran train
```

1. **Test Data Preparation (`prepare_test_data`)**
   - Loads test days from `MODELS/test_days_EBX.txt`
   - Applies same feature engineering pipeline as training
   - Creates sequences with stride=4 (one prediction every 4 timesteps)
   - Scales features using saved scaler
   - **Outputs:**
     - `test_data_EBX/X_test_day*.npy` - Scaled feature sequences
     - `test_data_EBX/indices_day*.npy` - Original row indices for each prediction
     - `test_data_EBX/prices_day*.npy` - Corresponding price values

2. **Signal Generation (`generate_signals`)**
   - Loads trained model and feature names
   - Processes each test day:
     - Makes predictions in batches
     - Filters by confidence threshold (default 0.5)
     - Creates OHLC candles from tick data
     - Optionally detects whipsaw days (disabled by default)
     - Runs candle-based strategy simulation:
       - Aggregates predictions within candle periods
       - Calculates signal strength (long_count - short_count)
       - Opens positions when strength exceeds `threshold_open`
       - Closes positions when strength falls below `threshold_close`
       - Applies risk management (stops, take-profits, trailing stops)
     - Generates BUY/SELL/EXIT signals aligned with price timestamps
   - **Outputs:**
     - `SIGNALS_EBX/day*_signals.csv` - Trading signals for each day
       ```csv
       Time,Price,BUY,SELL,EXIT
       09:15:00,100.25,1,0,0
       09:40:00,100.50,0,0,1
       ```
     - Console performance summary

---

### Step 3: Run Backtest

Backtesting validates the generated signals using the BacktesterIIT framework.

**Individual Strategy Backtest:**
```bash
python EBX.py backtest_ebullient --config config_EBX.json
python EBY.py backtest_ebullient --config config_EBY.json
```

**Combined Strategy Backtest:**
```bash
python Backtest_EBX_EBY.py config.json
```

**What Happens:**

1. **Signal Loading**
   - Reads signal CSV files from `SIGNALS_EBX/` or `SIGNALS_EBY/`
   - Creates time-indexed dictionary for fast lookup
   - Maps signal timestamps to price ticks

2. **Backtesting Engine**
   - For each day:
     - Creates temporary config file
     - Initializes BacktesterIIT with transaction costs
     - Processes each tick chronologically:
       - Checks for matching signal timestamp
       - Executes BUY/SELL/EXIT orders
       - Updates positions and PnL
       - Records trade history
     - Generates per-day plots showing:
       - Price chart with entry/exit markers
       - Rolling signal counts
       - Cumulative PnL curve

3. **Results Aggregation**
   - Combines results across all days
   - Calculates comprehensive metrics:
     - Returns, Sharpe ratio, max drawdown
     - Win rates, profit factors
     - Average trade statistics
   - Generates detailed report
   - **Outputs:**
     - `PHOTOS_EBX/ebullient_backtest/day_*_backtest.png` - Per-day visualizations
     - Console report with detailed statistics

---

## Directory Structure Reference

After running the complete pipeline, you'll have the following structure:

```
FINAL_SUB_X/
│
├── EBX/                         # Raw data for EBX
│   ├── day0.csv
│   ├── day1.csv
│   └── ...
│
├── MODELS/                      # Trained models and artifacts
│   ├── xgb_model_EBX.json       # XGBoost model for EBX
│   ├── scaler_EBX.pkl           # Feature scaler for EBX
│   └── features_EBX.txt         # Feature names
│
├── train_data_X/               # EBX training data
│   ├── X_day*.npy              # Feature sequences (samples, 30, num_features)
│   ├── y_day*.npy              # Labels (samples,) - values: 0, 1, 2
│   ├── train_days_X.txt        # List of days used for training
│   └── test_days_X.txt         # Remaining days for testing
│
├── test_data_X/                # EBX test sequences
│   ├── X_test_day*.npy         # Scaled features for prediction
│   ├── indices_day*.npy        # Original row indices
│   └── prices_day*.npy         # Corresponding prices
│
├── SIGNALS_EBX/                # Generated signals for EBX
│   ├── day0_signals.csv        # Columns: Time, Price, BUY, SELL, EXIT
│   ├── day1_signals.csv        # BUY/SELL/EXIT: binary flags (0 or 1)
│   └── ...
│
├── PHOTOS_EBX/                 # EBX visualizations and reports
│   ├── feat_imp.csv            # Feature importance scores
│   ├── feat_imp_top30.png      # Top 30 features bar chart
│   ├── cm.png                  # Confusion matrix heatmap
│   └── ebullient_backtest/     # Backtest results
│       ├── day_000_backtest.png  # Price + signals + PnL plots
│       ├── day_001_backtest.png
│       └── ...
│
├── EBX.py                       # Main script for EBX pipeline
└──  config_EBX.json             # Configuration for EBX
```

### File Role Explanations

#### Raw Data Files (`EBX/`, `EBY/`)
- **day*.csv**: Tick-level price data with pre-computed features
  - Columns include: `Time`, `Price`, `PB*_T*`, `VB*_T*`, `BB*`, etc.
  - Each row represents one market tick

#### Model Files (`MODELS/`)
- **xgb_model_*.json**: Serialized XGBoost model (readable JSON format)
- **scaler_*.pkl**: Fitted RobustScaler for feature normalization
- **features_*.txt**: Ordered list of feature names used by model
  - Format: `F0_T29, F0_T28, ..., F0_T0, F1_T29, ...`
  - Flattened from (timesteps, features) → (timesteps × features)

#### Training Data (`train_data_EBX/`, `train_data_EBY/`)
- **X_day*.npy**: 3D arrays `(samples, lookback, features)`
  - Each sample is a sequence of 30 consecutive timesteps
  - Features include engineered indicators, time encodings, regime signals
- **y_day*.npy**: 1D arrays `(samples,)` with class labels
  - `0`: Hold (no action)
  - `1`: Buy Short (EBX) / Buy Long (EBY)
  - `2`: Buy Long (EBX) / Buy Short (EBY)
- **train_days_*.txt**: Training set day numbers
- **test_days_*.txt**: Test set day numbers

#### Test Data (`test_data_EB/`, `test_data_EB/`)
- **X_test_day*.npy**: 2D arrays `(samples, timesteps × features)`
  - Flattened and scaled, ready for prediction
  - Created with stride=4 (one prediction per 4 ticks)
- **indices_day*.npy**: Maps predictions back to original CSV rows
- **prices_day*.npy**: Price at each prediction point

#### Signal Files (`SIGNALS_EBX/`, `SIGNALS_EBY/`)
- **day*_signals.csv**: Trading signals aligned with price data
  - `Time`: Timestamp (HH:MM:SS format)
  - `Price`: Market price at signal time
  - `BUY`: 1 if open long position, else 0
  - `SELL`: 1 if open short position, else 0
  - `EXIT`: 1 if close current position, else 0
  - Only one flag is 1 per row; most rows are all 0s

#### Visualization Files (`PHOTOS_EBX/`, `PHOTOS_EBY/`)
- **feat_imp.csv**: Feature importance sorted by gain
- **feat_imp_top30.png**: Bar chart of most important features
- **cm.png**: Confusion matrix for validation set
- **ebullient_backtest/day_*_backtest.png**: Three-panel plots
  1. Price with BUY (green ^), SELL (red v), EXIT (orange x) markers
  2. Rolling 20-candle signal counts
  3. Cumulative PnL curve with final value annotation

---

## Configuration Parameters

### Key Parameters in `config_X.json` and `config_Y.json`

#### Paths
```json
"paths": {
  "data_dir": "EBX",                    // Raw data directory
  "train_data_dir": "train_data_EBX",     // Preprocessed training data
  "test_data_dir": "test_data_EBX",       // Preprocessed test data
  "signals_dir": "SIGNALS_EBX",         // Generated signals output
  "plots_dir": "PHOTOS_EBX",            // Visualizations output
  "model_file": "MODELS/xgb_model_EBX.json",
  "scaler_file": "MODELS/scaler_EBX.pkl",
  "feature_names_file": "MODELS/features_EBX.txt"
}
```

#### Feature Engineering
```json
"features": {
  "families": ["PB1_T", "PB2_T", ...],  // Feature prefixes to extract
  "max_t": 6,                            // Maximum T-level to include
  "max_t_drop": 120                      // Initial rows to drop (warmup)
}
```

#### Kalman Filter
```json
"kalman": {
  "q": 0.00001,                          // Process noise
  "r": 1.0,                              // Observation noise
  "slope_period": 5,                     // Periods for slope calculation
  "slope_std_mult": 1.0                  // Threshold multiplier
}
```

#### Regime Detection (CUSUM)
```json
"regime": {
  "t": 7,                                // Primary T-level
  "rst": 8,                              // Reset T-level
  "lim": 15,                             // CUSUM limit
  "th": 2                                // Regime threshold
}
```

#### Training
```json
"training": {
  "lookback": 30,                        // Sequence length (timesteps)
  "hold_ratio": 1.0,                     // Hold samples / Action samples
  "batch_size": 60000,                   // Samples per batch
  "early_stop": 400                      // Early stopping rounds
}
```

#### XGBoost
```json
"xgboost": {
  "objective": "multi:softprob",         // Multi-class classification
  "num_class": 3,                        // Hold, Long, Short
  "max_depth": 3,                        // Tree depth
  "learning_rate": 0.04,                 // Step size
  "n_estimators": 1200,                  // Number of trees
  "subsample": 0.7,                      // Row sampling
  "colsample_bytree": 0.7                // Column sampling
}
```

#### Testing
```json
"testing": {
  "stride": 4,                           // Prediction interval
  "confidence_threshold": 0.5,           // Minimum prediction confidence
  "prediction_batch_size": 10000         // Samples per prediction batch
}
```

#### Candle Strategy
```json
"strategy": {
  "candle_minutes": 20,                  // Candle period (EBX: 20, EBY: 25)
  "threshold_open": 10,                  // Signal strength to open
  "threshold_close": 15,                 // Signal strength to close
  "warmup_minutes": 5,                   // Initial warmup period
  "whipsaw_detection": false             // Skip whipsaw days (optional)
}
```

#### Risk Management
```json
"risk": {
  "stop_loss_pct": 0.002,                // Stop loss (0.2%)
  "take_profit_pct": 0.004,              // Take profit (0.4%)
  "trailing_stop_pct": 0.0035,           // Trailing stop (0.35%)
  "min_holding_time_seconds": 15,        // Minimum hold time
  "min_time_between_trades_seconds": 15  // Cool-down period
}
```

#### Backtesting
```json
"backtesting": {
  "initial_capital": 100000,             // Starting capital
  "transaction_cost_rate": 0.0002        // Transaction cost (0.02%)
},
"backtest": {
  "ticker_name": "EBX",                  // Ticker symbol
  "tcost": 2,                            // Transaction cost for BacktesterIIT
  "timer_seconds": 600                   // Timer interval
}
```

### Combined Backtest Configuration (`config.json`)

```json
{
  "data_path": ".",                      // Root directory for data lookup
  "start_date": 0,                       // First day to backtest
  "end_date": 509,                       // Last day to backtest
  "timer": 600,                          // Timer callback interval (seconds)
  "tcost": 2,                            // Transaction cost
  "broadcast": [                         // Signal directories to monitor
    "FINAL_SUB_X/SIGNALS_EBX",
    "FINAL_SUB_Y/SIGNALS_EBY"
  ]
}
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "No data files found" during training
**Problem**: `EBX/` or `EBY/` directory is empty or misnamed.

**Solution**:
```bash
# Check directory exists and contains day*.csv files
ls EBX/day*.csv | head
ls EBY/day*.csv | head
```

#### 2. "Missing columns" error during feature engineering
**Problem**: Raw CSV doesn't contain expected feature columns.

**Solution**:
- Verify CSV has columns matching `config_*.json` feature families
- Check `max_t` setting matches available T-levels in data

#### 3. Model prediction fails with shape mismatch
**Problem**: Test data features don't match training features.

**Solution**:
- Ensure same `config_*.json` used for train and test
- Delete `test_data_*/` and regenerate with matching config
- Check `features_*.txt` matches expected count

#### 4. Signal file not found during backtest
**Problem**: Test step didn't complete or signals weren't generated.

**Solution**:
```bash
# Check if signal files exist
ls SIGNALS_EBX/*.csv | wc -l

# Regenerate signals if missing
python EBX.py test --config config_EBX.json
```

#### 5. Memory errors during training
**Problem**: Insufficient RAM for large datasets.

**Solution**:
- Reduce `training.batch_size` in config (e.g., 30000)
- Use fewer training days
- Close other applications

#### 6. Poor model performance (accuracy < 50%)
**Problem**: Inadequate training or poor hyperparameters.

**Solution**:
- Increase training days (e.g., `--days 300`)
- Adjust XGBoost parameters (learning_rate, max_depth)
- Check class balance in training data
- Review feature importance - remove weak features

#### 7. Backtest PnL doesn't match test results
**Problem**: Different execution logic or transaction costs.

**Solution**:
- Verify `tcost` matches in config files
- Check risk management settings (stops, take-profits)
- Compare signal files with backtest execution logs
- Ensure timestamps align correctly

#### 8. "Train_days.txt not found" during testing
**Problem**: Training step wasn't completed successfully.

**Solution**:
```bash
# Retrain to generate day lists
python EBX.py train --days MODELS/train_day_EBY.txt --config config_EBX.json
```
---