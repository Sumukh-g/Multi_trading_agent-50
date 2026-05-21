import sys
print("Python version:", sys.version)
print("Testing imports...")
try:
    import yaml
    print("✓ yaml")
    import yfinance
    print("✓ yfinance")
    import pandas
    print("✓ pandas")
    import numpy
    print("✓ numpy")
    import sklearn
    print("✓ sklearn")
    import lightgbm
    print("✓ lightgbm")
    import xgboost
    print("✓ xgboost")
    print("\nAll imports successful!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

