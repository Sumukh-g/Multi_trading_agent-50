"""Test download with just a few tickers."""
import yfinance as yf
import pandas as pd
import time
import random

def test_download(tk):
    """Test downloading a single ticker."""
    print(f"Testing {tk}...", end=" ", flush=True)
    try:
        # Method 1: yf.download
        df = yf.download(tk, period="1y", interval="1d", auto_adjust=True, progress=False, show_errors=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(0, axis=1)
        
        if df is None or df.empty:
            print("❌ Empty")
            return False
        
        df = df.rename(columns=str.title)
        required = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required):
            print("❌ Missing columns")
            return False
        
        print(f"✅ {len(df)} rows")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)[:40]}")
        return False

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    print("Testing download methods...\n", flush=True)
    for tk in test_tickers:
        result = test_download(tk)
        sys.stdout.flush()
        time.sleep(1)
    print("\nTest complete!", flush=True)

