import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from risk_engine.risk_model import expected_returns, ledoit_wolf_cov
from risk_engine.optimizer import mv_optimize
import numpy as np

def fetch_hist(tickers, period="1y"):
    frames = {}
    for tk in tickers:
        try:
            df = yf.download(tk, period=period, interval="1d", auto_adjust=True, progress=False)
            if df is None or df.empty: 
                continue
            close = df['Close']
            # Handle multi-level columns
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            frames[tk] = close
        except Exception as e:
            continue
    
    if len(frames) == 0:
        return pd.DataFrame()
    
    px = pd.DataFrame(frames).dropna(how='all')
    rets = px.pct_change().dropna()
    return rets

def main():
    import os
    print("="*60, flush=True)
    print("CREATING OPTIMIZED PORTFOLIO", flush=True)
    print("="*60, flush=True)
    
    sig = pd.read_csv("data/signals_latest.csv")
    tradables = sig[sig['actionable']].head(20)  # top 20 actionable
    
    if tradables.empty:
        print("\n⚠️  No actionable signals found.", flush=True)
        print("Creating portfolio with all signals instead...", flush=True)
        tradables = sig.head(20)
        if tradables.empty:
            print("❌ No signals available at all!", flush=True)
            return
    
    print(f"\nProcessing {len(tradables)} signals...", flush=True)
    
    tks = tradables['ticker'].tolist()
    print(f"Fetching historical data for optimization...", flush=True)
    rets = fetch_hist(tks, "1y")
    
    if rets.empty:
        print("❌ No historical data available for optimization!", flush=True)
        # Create equal weight portfolio
        tradables['weight'] = 1.0 / len(tradables)
        tradables.to_csv("data/portfolio_latest.csv", index=False)
        print("✅ Created equal-weight portfolio instead", flush=True)
        return
    
    # Only use tickers that have return data
    available_tks = [tk for tk in tks if tk in rets.columns]
    if len(available_tks) == 0:
        print("❌ No return data available!", flush=True)
        return
    
    print(f"Optimizing portfolio with {len(available_tks)} assets...", flush=True)
    mu = expected_returns(rets)
    Sigma, cols = ledoit_wolf_cov(rets[available_tks])
    w = mv_optimize(mu.reindex(available_tks).fillna(0), Sigma, available_tks, lam=10.0, w_cap=0.05)
    out = tradables.merge(w.rename("weight"), left_on="ticker", right_index=True, how="left").fillna(0)
    
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/portfolio_latest.csv", index=False)
    
    print(f"\n{'='*60}", flush=True)
    print(f"✅ Portfolio created successfully", flush=True)
    print(f"   Positions: {(out['weight'] > 0).sum()}", flush=True)
    print(f"   Total weight: {out['weight'].sum():.2%}", flush=True)
    print(f"   Saved to: data/portfolio_latest.csv", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
