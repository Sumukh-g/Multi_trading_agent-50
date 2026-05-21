"""
Complete Pipeline Runner for the Quantitative Trading System

This script runs the entire pipeline:
1. Train models (multiple thresholds: 2%, 5%, 10%)
2. Generate signals (with soft decisions)
3. Create portfolio
4. Render visualization
5. Launch dashboard (optional)

Usage:
    python run_complete_pipeline.py
    python run_complete_pipeline.py --mode aggressive --with-ai
    python run_complete_pipeline.py --skip-training --dashboard
"""

import sys
import subprocess
import os
import argparse
from datetime import datetime


def run_step(name: str, command: list, required: bool = False, env: dict = None) -> bool:
    """
    Run a pipeline step with error handling.
    
    Args:
        name: Step name for display
        command: Command to run as list
        required: Whether failure should stop the pipeline
        env: Optional environment variables
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"STEP: {name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(command)}\n")
    
    # Merge environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    
    try:
        result = subprocess.run(
            command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            check=False,
            env=run_env
        )
        
        if result.returncode != 0:
            if required:
                print(f"\n❌ {name} failed with exit code {result.returncode}")
                print("Pipeline stopped.")
                return False
            else:
                print(f"\n⚠️  {name} completed with warnings (exit code {result.returncode})")
                return True
        else:
            print(f"\n✅ {name} completed successfully")
            return True
            
    except Exception as e:
        print(f"\n❌ Error in {name}: {e}")
        if required:
            return False
        return True


def main():
    parser = argparse.ArgumentParser(description="Run the complete trading pipeline")
    parser.add_argument("--mode", type=str, default="balanced",
                       choices=["conservative", "balanced", "aggressive"],
                       help="Prediction mode for signal generation")
    parser.add_argument("--with-ai", action="store_true",
                       help="Enable AI explanations (requires Gemini API key)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="Gemini API key for AI explanations")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model training step")
    parser.add_argument("--skip-portfolio", action="store_true",
                       help="Skip portfolio optimization")
    parser.add_argument("--dashboard", action="store_true",
                       help="Launch Streamlit dashboard after pipeline")
    
    args = parser.parse_args()
    
    # Determine Python executable
    python_exe = sys.executable
    
    print("="*70)
    print("🚀 QUANTITATIVE TRADING MODEL - COMPLETE PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {python_exe}")
    print(f"Mode: {args.mode}")
    print(f"AI Explanations: {'Enabled' if args.with_ai else 'Disabled'}")
    
    steps_run = 0
    steps_passed = 0
    
    # Step 1: Train Models
    if not args.skip_training:
        success = run_step(
            "Train Models (Multi-threshold: 2%, 5%, 10%)",
            [python_exe, "-m", "src.modelling.train"],
            required=True
        )
        steps_run += 1
        if success:
            steps_passed += 1
        else:
            print(f"\n{'='*70}")
            print("Pipeline stopped due to training failure.")
            print(f"{'='*70}")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping training step (--skip-training)")
    
    # Step 2: Generate Signals
    signal_cmd = [python_exe, "-m", "src.live.run_signals", "--mode", args.mode]
    if not args.with_ai:
        signal_cmd.append("--no-ai")
    if args.api_key:
        signal_cmd.extend(["--api-key", args.api_key])
    
    env = {}
    if args.api_key:
        env["GEMINI_API_KEY"] = args.api_key
    
    success = run_step(
        "Generate Trading Signals",
        signal_cmd,
        required=True,
        env=env if env else None
    )
    steps_run += 1
    if success:
        steps_passed += 1
    else:
        print(f"\n{'='*70}")
        print("Pipeline stopped due to signal generation failure.")
        print(f"{'='*70}")
        sys.exit(1)
    
    # Step 3: Create Portfolio
    if not args.skip_portfolio:
        success = run_step(
            "Create Optimized Portfolio",
            [python_exe, "-m", "src.live.portfolio_from_signals"],
            required=False
        )
        steps_run += 1
        if success:
            steps_passed += 1
    else:
        print("\n⏭️  Skipping portfolio step (--skip-portfolio)")
    
    # Step 4: Render Visualization
    success = run_step(
        "Render Visualization",
        [python_exe, "-m", "src.live.render_sheet"],
        required=False
    )
    steps_run += 1
    if success:
        steps_passed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"Steps completed: {steps_passed}/{steps_run}")
    print(f"\n📁 Generated files:")
    
    files_to_check = [
        ("models/", "Model files"),
        ("data/signals_latest.csv", "Trading signals"),
        ("data/signals_latest.json", "Signals (JSON)"),
        ("data/signals_summary.json", "Summary statistics"),
        ("data/portfolio_latest.csv", "Portfolio weights"),
        ("web/signals.html", "HTML visualization")
    ]
    
    for path, description in files_to_check:
        if os.path.exists(path):
            if os.path.isdir(path):
                file_count = len([f for f in os.listdir(path) if f.endswith(('.pkl', '.json'))])
                print(f"   📁 {path} ({file_count} files)")
            else:
                print(f"   📄 {path}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Launch dashboard if requested
    if args.dashboard:
        print(f"\n{'='*70}")
        print("🌐 Launching Dashboard...")
        print(f"{'='*70}")
        print("Dashboard will open in your browser.")
        print("Press Ctrl+C to stop the dashboard.\n")
        
        subprocess.run([python_exe, "-m", "streamlit", "run", "dashboard.py"])


if __name__ == "__main__":
    main()
