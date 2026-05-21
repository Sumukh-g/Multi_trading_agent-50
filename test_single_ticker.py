"""Test downloading a single ticker to verify the fix works."""
import yfinance as yf
import pandas as pd
import warnings
import sys
from io import StringIO

warnings.filterwarnings('ignore')

class SuppressStderr:
    def __init__(self):
        self.original_stderr = sys.stderr
    def __enter__(self):
        sys.stderr = StringIO()
        return self
    def __exit__(self, *args):
        sys.stderr = self.original_stderr

def test_ticker(tk):
    print(f"Testing {tk}...", flush=True)
    
    # Method 1: yf.download
    print("  Method 1: yf.download...", end=" ", flush=True)
    try:
        with SuppressStderr():
            df = yf.download(tk, period="2y", interval="1d", auto_adjust=True, progress=False, show_errors=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(0, axis=1)
            if df is not None and not df.empty and len(df) > 50:
                print(f"✓ SUCCESS ({len(df)} rows)")
                return True
    except:
        pass
    print("FAILED")
    
    # Method 2: Ticker object
    print("  Method 2: Ticker.history...", end=" ", flush=True)
    try:
        with SuppressStderr():
            ticker = yf.Ticker(tk)
            df = ticker.history(period="2y", interval="1d", auto_adjust=True, timeout=60, raise_errors=False)
            if df is not None and not df.empty and len(df) > 50:
                print(f"✓ SUCCESS ({len(df)} rows)")
                return True
    except:
        pass
    print("FAILED")
    
    return False

if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT"]
    print("Testing download methods with error suppression...\n")
    for tk in test_tickers:
        result = test_ticker(tk)
        print()
    print("If both methods failed, there may be a network/API issue.")
    print("Try again in a few minutes or check your internet connection.")

