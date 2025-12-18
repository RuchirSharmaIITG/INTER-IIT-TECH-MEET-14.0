# Usage : python Backtest_EBX_EBY.py config.json
import os
import sys
from datetime import datetime
from alpha_research.backtesterIIT import BacktesterIIT, Side, Ticker # Make sure Side is imported

def my_broadcast_callback(state, ts):
    
    current_positions = backtest.position_map

    for ticker, data in state.items():
        if 'Price' not in data or data['Price'] == 0:
            continue
        current_pos = current_positions.get(ticker, 0)
        try:
            buy_signal = int(data.get("BUY", 0))
            sell_signal = int(data.get("SELL", 0))
            exit_signal = int(data.get("EXIT", 0))
        except (ValueError, TypeError):
            print(f"[{ts}] Invalid signal data for {ticker}. Skipping.")
            continue
        if exit_signal == 1:
            if current_pos == 1: # We are long, exit by selling
                print(f"[{ts}] (EXIT) EXITING LONG for {ticker} at {data['Price']}")
                backtest.place_order(ticker=ticker, qty=1, side=Side.SELL)
            elif current_pos == -1: # We are short, exit by buying
                print(f"[{ts}] (EXIT) EXITING SHORT for {ticker} at {data['Price']}")
                backtest.place_order(ticker=ticker, qty=1, side=Side.BUY)
        elif buy_signal == 1:
            if current_pos == 0: # Only buy if we are flat
                print(f"[{ts}] (BUY) PLACING BUY for {ticker} at {data['Price']}")
                backtest.place_order(ticker=ticker, qty=1, side=Side.BUY)
        elif sell_signal == 1:
            if current_pos == 0: # Only sell (short) if we are flat
                print(f"[{ts}] (SELL) PLACING SELL for {ticker} at {data['Price']}")
                backtest.place_order(ticker=ticker, qty=1, side=Side.SELL)


def on_timer(ts):
    print(f"\n[TIMER] Timestamp={ts}")
    backtest.position_manager.print_details()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Backtest_EBX_EBY.py <config.json>")
        sys.exit(1)

    config_file = os.path.abspath(sys.argv[1])
    backtest = BacktesterIIT(config_file)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Backtest starting...")
    backtest.run(
        broadcast_callback=my_broadcast_callback,
        timer_callback=on_timer
    )
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Backtest finished.")