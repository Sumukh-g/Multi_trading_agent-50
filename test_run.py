"""Test script to verify the system works."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("Testing Quantitative Trading System")
print("="*60)

# Test 1: Load features
print("\n1. Testing feature loading...")
try:
    import yfinance as yf
    from src.features.builder import make_features
    
    df = yf.download("AAPL", period="1y", interval="1d", auto_adjust=True, progress=False)
    df = df.rename(columns=str.title)
    X = make_features(df)
    print(f"   ✓ Features loaded: {len(X.columns)} features, {len(X)} samples")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Load models
print("\n2. Testing model loading...")
try:
    import joblib
    import json
    
    models_exist = []
    for name in ["lgb_clf.pkl", "xgb_clf.pkl", "meta_logit.pkl", "lgb_reg.pkl"]:
        path = f"models/{name}"
        if os.path.exists(path):
            models_exist.append(name)
    
    print(f"   ✓ Found {len(models_exist)} models: {', '.join(models_exist)}")
    
    if os.path.exists("models/uncertainty.json"):
        with open("models/uncertainty.json") as f:
            unc = json.load(f)
        print(f"   ✓ Uncertainty loaded: tau={unc.get('selective_threshold', 'N/A'):.3f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Test inference
print("\n3. Testing inference...")
try:
    from src.modelling.infer import score_latest
    
    result = score_latest(df, mode="balanced")
    print(f"   ✓ Inference works")
    print(f"     - Decision: {result['decision']}")
    print(f"     - Soft Decision: {result['soft_decision']}")
    print(f"     - Confidence: {result['confidence']:.3f}")
    print(f"     - Expected Return: {result['exp_return']*100:+.2f}%")
    print(f"     - Signal Strength: {result['signal_strength']:.1f}/100")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check signals file
print("\n4. Checking signals file...")
try:
    import pandas as pd
    
    if os.path.exists("data/signals_latest.csv"):
        sig = pd.read_csv("data/signals_latest.csv")
        print(f"   ✓ Signals file exists: {len(sig)} rows")
        print(f"   Columns: {', '.join(sig.columns[:10])}...")
        
        # Check for new columns
        new_cols = ['action', 'soft_decision', 'signal_strength', 'prob_2pct', 'prob_10pct']
        missing = [c for c in new_cols if c not in sig.columns]
        if missing:
            print(f"   ⚠️  Missing new columns: {missing}")
            print(f"   → Run: python -m src.live.run_signals")
        else:
            print(f"   ✓ All new columns present")
    else:
        print("   ⚠️  No signals file found")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("Test complete!")
print("="*60)

