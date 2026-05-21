"""Improved download script with session management and better error handling."""
import yaml
import yfinance as yf
import pandas as pd
import os
import time
import random
import sys
import warnings
from io import StringIO

# Suppress yfinance warnings and errors
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect stderr to suppress yfinance error messages
class SuppressStderr:
    def __init__(self):
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        sys.stderr = StringIO()
        return self
        
    def __exit__(self, *args):
        sys.stderr = self.original_stderr

def download_ticker_robust(tk, max_retries=3):
    """Download with multiple methods, session management, and retries."""
    
    # Method 1: yf.download with session
    def method1():
        with SuppressStderr():
            try:
                df = yf.download(
                    tk, 
                    period="2y",  # Shorter period sometimes works better
                    interval="1d", 
                    auto_adjust=True, 
                    progress=False, 
                    show_errors=False,
                    threads=False  # Single-threaded can be more reliable
                )
                return df
            except:
                return None
    
    # Method 2: Ticker object with explicit session
    def method2():
        with SuppressStderr():
            try:
                ticker = yf.Ticker(tk)
                # Try to get info first to validate ticker
                try:
                    info = ticker.info
                    if not info or len(info) < 5:
                        return None
                except:
                    pass
                
                df = ticker.history(
                    period="2y", 
                    interval="1d", 
                    auto_adjust=True, 
                    timeout=60,
                    raise_errors=False
                )
                return df
            except:
                return None
    
    # Method 3: Download with date range
    def method3():
        with SuppressStderr():
            try:
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)  # ~2 years
                df = yf.download(
                    tk,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    show_errors=False,
                    threads=False
                )
                return df
            except:
                return None
    
    methods = [method1, method2, method3]
    
    for attempt in range(max_retries):
        for method_idx, method in enumerate(methods):
            try:
                df = method()
                
                if df is None:
                    continue
                
                # Handle MultiIndex columns from yf.download
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(0, axis=1)
                
                # Check if we got valid data
                if df.empty or len(df) < 50:
                    continue
                
                # Normalize columns
                df = df.rename(columns=str.title)
                
                # Check required columns
                required = ['Open', 'High', 'Low', 'Close']
                if not all(col in df.columns for col in required):
                    continue
                
                # Validate data quality
                if df['Close'].isna().sum() > len(df) * 0.5:  # More than 50% NaN
                    continue
                
                # Add Volume if missing
                if 'Volume' not in df.columns:
                    df['Volume'] = 0
                
                # Clean up any remaining NaN values
                df = df.dropna(subset=['Close'])
                
                if len(df) < 50:
                    continue
                
                return df
                
            except Exception as e:
                # Silently continue to next method
                continue
        
        # Wait longer between retry cycles
        if attempt < max_retries - 1:
            wait = (attempt + 1) * 5 + random.uniform(0, 3)  # Longer waits
            time.sleep(wait)
    
    return None

def main():
    print("="*60, flush=True)
    print("Financial Data Downloader (Enhanced)", flush=True)
    print("="*60, flush=True)
    
    os.makedirs("raw", exist_ok=True)
    
    try:
        with open("config/universe.yaml", "r") as f:
            uni = yaml.safe_load(f)
        tickers = uni["tickers"]
    except Exception as e:
        print(f"Error loading config: {e}", flush=True)
        return
    
    print(f"\nFound {len(tickers)} tickers to download", flush=True)
    print("Note: This may take several minutes due to API rate limiting\n", flush=True)
    
    successful = 0
    failed = []
    
    for i, tk in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {tk}...", end=" ", flush=True)
        
        df = download_ticker_robust(tk)
        
        if df is None or df.empty:
            print("FAILED", flush=True)
            failed.append(tk)
        else:
            try:
                df.to_parquet(f"raw/{tk}.parquet", compression='snappy', index=True)
                print(f"OK ({len(df)} rows)", flush=True)
                successful += 1
            except Exception as e:
                print(f"SAVE ERROR", flush=True)
                failed.append(tk)
        
        # Longer delay to avoid rate limiting
        if i < len(tickers):
            delay = 3 + random.uniform(0, 2)
            time.sleep(delay)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Results: {successful} successful, {len(failed)} failed", flush=True)
    if failed:
        print(f"Failed tickers: {', '.join(failed)}", flush=True)
        print(f"\nTip: Wait 10-15 minutes and run again for failed tickers", flush=True)
    print(f"{'='*60}\n", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", flush=True)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nFatal error: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
