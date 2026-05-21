"""Test if data loading and feature building works."""
import pandas as pd
from src.utils.io import load_df
from src.features.builder import make_features
from src.labels.labels import label_bigmove_30d

print("Testing data loading...")
try:
    df = load_df("raw/AAPL.parquet")
    print(f"✓ Loaded AAPL: {df.shape}")
    print(f"  Columns: {df.columns.tolist()[:5]}...")
    print(f"  Index type: {type(df.index)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    print("\nTesting feature building...")
    X = make_features(df)
    print(f"✓ Features built: {X.shape}")
    print(f"  Feature columns: {len(X.columns)}")
    
    print("\nTesting label creation...")
    y = label_bigmove_30d(df)
    print(f"✓ Labels created: {y.shape}")
    print(f"  Non-null labels: {y['y_bigmove'].notna().sum()}")
    
    # Test alignment
    common_idx = X.index.intersection(y.index)
    y_aligned = y.reindex(common_idx).dropna()
    X_aligned = X.reindex(y_aligned.index)
    
    print(f"\n✓ After alignment:")
    print(f"  X shape: {X_aligned.shape}")
    print(f"  y shape: {y_aligned.shape}")
    print(f"  y_bigmove distribution: {y_aligned['y_bigmove'].value_counts().to_dict()}")
    
    if len(X_aligned) >= 100:
        print("\n✅ SUCCESS: Data loading and processing works!")
    else:
        print(f"\n⚠️  WARNING: Only {len(X_aligned)} rows after processing (need 100+)")
        
except Exception as e:
    import traceback
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()

