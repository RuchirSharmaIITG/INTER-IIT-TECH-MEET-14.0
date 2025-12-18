# In mainIIT.py
import sys
from datetime import datetime
from alpha_research.backtesterIIT import BacktesterIIT, Side, Ticker # Make sure Side is imported

# You do NOT need the 'previous_signal_state' dictionary for this logic.
# DELETE the line: previous_signal_state = {}

def my_broadcast_callback(state, ts):
    """
    This function reads your BUY, SELL, and EXIT columns directly from the CSV.
    """
    
    # Get the backtester's current position map
    current_positions = backtest.position_map

    for ticker, data in state.items():
        
        # 1. Skip if there's no valid price data
        if 'Price' not in data or data['Price'] == 0:
            continue

        # 2. Get the current position for this specific ticker (defaults to 0)
        current_pos = current_positions.get(ticker, 0)

        # 3. Check your signal columns (use .get() for safety, defaulting to 0)
        try:
            # Read the signals from the data row
            buy_signal = int(data.get("BUY", 0))
            sell_signal = int(data.get("SELL", 0))
            exit_signal = int(data.get("EXIT", 0))
        except (ValueError, TypeError):
            print(f"[{ts}] Invalid signal data for {ticker}. Skipping.")
            continue # Skip this tick if data is bad

        # --- Your Strategy Logic ---

        # 4. Check for EXIT Signal first (Highest Priority)
        if exit_signal == 1:
            if current_pos == 1: # We are long, exit by selling
                print(f"[{ts}] (EXIT) EXITING LONG for {ticker} at {data['Price']}")
                backtest.place_order(ticker=ticker, qty=1, side=Side.SELL)
            elif current_pos == -1: # We are short, exit by buying
                print(f"[{ts}] (EXIT) EXITING SHORT for {ticker} at {data['Price']}")
                backtest.place_order(ticker=ticker, qty=1, side=Side.BUY)

        # 5. Check for BUY Signal (only if not exiting)
        elif buy_signal == 1:
            if current_pos == 0: # Only buy if we are flat
                print(f"[{ts}] (BUY) PLACING BUY for {ticker} at {data['Price']}")
                backtest.place_order(ticker=ticker, qty=1, side=Side.BUY)

        # 6. Check for SELL Signal (only if not exiting or buying)
        elif sell_signal == 1:
            if current_pos == 0: # Only sell (short) if we are flat
                print(f"[{ts}] (SELL) PLACING SELL for {ticker} at {data['Price']}")
                backtest.place_order(ticker=ticker, qty=1, side=Side.SELL)


def on_timer(ts):
    """
    This function is called based on the 'timer' in your config file.
    """
    print(f"\n[TIMER] Timestamp={ts}")
    backtest.position_manager.print_details()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config.json>")
        sys.exit(1)

    config_file = sys.argv[1]
    backtest = BacktesterIIT(config_file)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Backtest starting...")
    
    # This line "registers" your functions as the callbacks
    backtest.run(
        broadcast_callback=my_broadcast_callback,
        timer_callback=on_timer
    )
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Backtest finished.")