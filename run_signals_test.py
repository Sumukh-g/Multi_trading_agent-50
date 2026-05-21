"""Quick test to generate signals."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run
from src.live.run_signals import main

try:
    main(mode="balanced", use_ai=False)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

