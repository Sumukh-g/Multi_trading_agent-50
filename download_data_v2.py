"""Improved download script with better error handling."""
import yaml
import yfinance as yf
import pandas as pd
import os
import time
import random
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

def download_ticker_robust(tk, max_retries=3):
    """Download with multiple methods and retries."""
    methods = [
        lambda: yf.download(tk, period="3y", interval="1d", auto_adjust=True, progress=False, show_errors=False),
        lambda: yf.Ticker(tk).history(period="3y", interval="1d", auto_adjust=True, timeout=30),
        lambda: yf.download(tk, start="2021-01-01", end=None, interval="1d", auto_adjust=True, progress=False, show_errors=False),
    ]
    
    for attempt in range(max_retries):
        for method_idx, method in enumerate(methods):
            try:
                df = method()
                
                # Handle MultiIndex columns from yf.download
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(0, axis=1)
                
                if df is None or df.empty or len(df) < 100:
                    continue
                
                # Normalize columns
                df = df.rename(columns=str.title)
                
                # Check required columns
                required = ['Open', 'High', 'Low', 'Close']
                if not all(col in df.columns for col in required):
                    continue
                
                # Add Volume if missing
                if 'Volume' not in df.columns:
                    df['Volume'] = 0
                
                return df
                
            except Exception as e:
                if attempt < max_retries - 1 or method_idx < len(methods) - 1:
                    time.sleep(1 + random.uniform(0, 1))
                    continue
                return None
        
        # Wait longer between retry cycles
        if attempt < max_retries - 1:
            wait = (attempt + 1) * 3 + random.uniform(0, 2)
            time.sleep(wait)
    
    return None

def main():
    print("="*60, flush=True)
    print("Financial Data Downloader", flush=True)
    print("="*60, flush=True)
    
    os.makedirs("raw", exist_ok=True)
    
    try:
        with open("config/universe.yaml", "r") as f:
            uni = yaml.safe_load(f)
        tickers = uni["tickers"]
    except Exception as e:
        print(f"Error loading config: {e}", flush=True)
        return
    
    print(f"\nFound {len(tickers)} tickers to download\n", flush=True)
    
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
                df.to_parquet(f"raw/{tk}.parquet", compression='snappy')
                print(f"OK ({len(df)} rows)", flush=True)
                successful += 1
            except Exception as e:
                print(f"SAVE ERROR: {str(e)[:30]}", flush=True)
                failed.append(tk)
        
        # Delay to avoid rate limiting
        if i < len(tickers):
            time.sleep(2 + random.uniform(0, 1))
    
    print(f"\n{'='*60}", flush=True)
    print(f"Results: {successful} successful, {len(failed)} failed", flush=True)
    if failed:
        print(f"Failed: {', '.join(failed)}", flush=True)
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

