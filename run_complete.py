"""Run the complete pipeline: download data, train, generate signals, create portfolio, render."""
import sys
import os
import subprocess
import time

def run_step(name, command):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print()
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=False,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print(f"\n⚠️  {name} completed with exit code {result.returncode}")
            return False
        print(f"\n✅ {name} completed successfully")
        return True
    except Exception as e:
        print(f"\n❌ Error in {name}: {e}")
        return False

def main():
    python_exe = r"C:\Users\cenas\AppData\Local\Programs\Python\Python311\python.exe"
    
    steps = [
        ("Download Data", f'{python_exe} download_data.py'),
        ("Train Models", f'{python_exe} -m src.modelling.train'),
        ("Generate Signals", f'{python_exe} -m src.live.run_signals'),
        ("Create Portfolio", f'{python_exe} -m src.live.portfolio_from_signals'),
        ("Render Sheet", f'{python_exe} -m src.live.render_sheet'),
    ]
    
    print("Starting complete pipeline...")
    print(f"Python: {python_exe}")
    
    for name, cmd in steps:
        success = run_step(name, cmd)
        if not success and name != "Download Data":  # Allow download to fail, training will download on fly
            print(f"\n⚠️  Pipeline stopped at {name}")
            print("You can continue manually or fix the issue.")
            break
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print("Pipeline execution complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

