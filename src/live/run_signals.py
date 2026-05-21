"""
Advanced Signal Generation Module

Generates comprehensive trading signals with:
- Multiple threshold predictions (2%, 5%, 10%)
- Direction predictions
- Soft decisions (always show recommendation)
- Multiple prediction modes
- AI-powered explanations (via Gemini)
"""

import yaml
import pandas as pd
import numpy as np
import yfinance as yf
import os
import json
from datetime import datetime
from typing import Dict, Optional

from src.modelling.infer import score_latest, PredictionMode
from src.rules.trade_engine import atr_stop, position_size
from src.features.builder import make_features
from src.rules.rulebook import load_rulebook

# Try to import Gemini agent
try:
    from src.agents.gemini_agent import create_agent
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def passes_gates(row: Dict, feats: pd.DataFrame, rules: Dict) -> tuple:
    """
    Check if signal passes all technical gates.
    
    Returns:
        Tuple of (passed: bool, reason: str)
    """
    dma = int(rules['gates']['min_price_above_dma']['dma'])
    dma_col = f'dma{dma}'
    
    # Price above DMA for longs
    if dma_col in feats.columns:
        if feats[dma_col].iloc[-1] < 0:
            if rules['gates']['min_price_above_dma']['require'] and row.get('soft_decision') == 'LONG':
                return False, f"Price below DMA{dma}"
    
    # ADX check
    if 'adx14' in feats.columns:
        if feats['adx14'].iloc[-1] < rules['gates']['min_adx']['threshold']:
            return False, "ADX too low (weak trend)"
    
    # Momentum check
    mom_threshold = rules['gates']['min_momentum_60d']['threshold']
    if 'mom_60' in feats.columns:
        mom = feats['mom_60'].iloc[-1]
        if row.get('soft_decision') == 'LONG' and mom < mom_threshold:
            return False, f"Momentum below {mom_threshold*100:.0f}%"
        if row.get('soft_decision') == 'SHORT' and mom > -mom_threshold:
            return False, "Momentum not negative enough"
    
    # ATR cap
    if row.get('atr_pct', 0) > rules['gates']['max_atr_pct']['threshold']:
        return False, "ATR% too high (too volatile)"
    
    # Expected move check
    min_move = rules['decisions']['min_expected_move']
    if abs(row.get('exp_30d_return', 0)) < min_move:
        return False, f"Expected move < {min_move*100:.0f}%"
    
    return True, ""


def build_row(tk: str, df: pd.DataFrame, rules: Dict, mode: str = "balanced") -> Dict:
    """
    Build a comprehensive signal row for a ticker.
    
    Args:
        tk: Ticker symbol
        df: OHLCV DataFrame
        rules: Rulebook dictionary
        mode: Prediction mode ("conservative", "balanced", "aggressive")
        
    Returns:
        Dictionary with all signal data
    """
    # Get predictions
    s = score_latest(df, mode=mode)
    
    # Extract price data safely
    close_col = df["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.iloc[:, 0]
    high_col = df["High"]
    if isinstance(high_col, pd.DataFrame):
        high_col = high_col.iloc[:, 0]
    low_col = df["Low"]
    if isinstance(low_col, pd.DataFrame):
        low_col = low_col.iloc[:, 0]
    
    last = float(close_col.iloc[-1])
    atr_val = (high_col - low_col).rolling(14).mean().iloc[-1]
    atr_pct = float(atr_val / last) if last > 0 else 0.0
    
    # Direction for risk calculations
    direction = 1 if s["exp_return"] > 0 else -1
    
    # Get features for gate checks
    feats = make_features(df).iloc[-1:]
    
    # Calculate risk parameters
    stop = atr_stop(last, atr_pct, mult=rules['risk']['hard_stop_mult'], direction=direction)
    size = position_size(
        prob=s["prob_5pct"], 
        exp_ret_abs=abs(s["exp_return"]), 
        vol_est=atr_pct
    ) * rules['regimes']['normal']['size_multiplier']
    target1 = last * (1 + rules['risk']['scale_out_at'] * direction)
    
    # Extract feature values safely
    dma200_val = float(feats['dma200'].iloc[-1]) if 'dma200' in feats.columns else 0.0
    adx14_val = float(feats['adx14'].iloc[-1]) if 'adx14' in feats.columns else 0.0
    mom_60_val = float(feats['mom_60'].iloc[-1]) if 'mom_60' in feats.columns else 0.0
    rsi14_val = float(feats['rsi14'].iloc[-1]) if 'rsi14' in feats.columns else 50.0
    
    # Build base row
    row = {
        "ticker": tk,
        "price": last,
        "date": datetime.now().strftime("%Y-%m-%d"),
        
        # Multi-threshold probabilities
        "prob_2pct": s["prob_2pct"],
        "prob_5pct_30d": s["prob_5pct"],
        "prob_10pct": s["prob_10pct"],
        "prob_direction": s["prob_direction"],
        
        # Expected return and intervals
        "exp_30d_return": s["exp_return"],
        "pi_low": s["pi_low"],
        "pi_high": s["pi_high"],
        "pi_width": s["pi_width"],
        
        # Confidence metrics
        "confidence": s["confidence"],
        "tau": s["tau"],
        "signal_strength": s["signal_strength"],
        "model_agreement": s["model_agreement"],
        
        # Decisions (hard and soft)
        "decision": s["decision"],
        "soft_decision": s["soft_decision"],
        "soft_strength": s["soft_strength"],
        "reason": s["reason"],
        "mode": mode,
        
        # Risk parameters
        "atr_pct": atr_pct,
        "size": size,
        "stop": stop,
        "target1": target1,
        "risk_reward": abs(target1 - last) / abs(last - stop) if stop != last else 0,
        
        # Technical indicators
        "dma200": dma200_val,
        "adx14": adx14_val,
        "mom_60": mom_60_val,
        "rsi14": rsi14_val
    }
    
    # Check gates
    ok, why = passes_gates(row, feats, rules)
    row["gates_pass"] = ok
    row["gates_reason"] = "" if ok else why
    
    return row


def generate_ai_explanations(signals: pd.DataFrame, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Add AI-powered explanations to signals using Gemini.
    
    Args:
        signals: DataFrame of signals
        api_key: Optional Gemini API key
        
    Returns:
        DataFrame with ai_explanation column added
    """
    if not GEMINI_AVAILABLE:
        signals['ai_explanation'] = "AI explanations require: pip install google-generativeai"
        signals['market_context'] = ""
        return signals
    
    agent = create_agent(api_key)
    
    explanations = []
    for _, row in signals.iterrows():
        signal_dict = row.to_dict()
        explanation = agent.explain_signal(signal_dict)
        explanations.append(explanation)
    
    signals['ai_explanation'] = explanations
    
    # Add market context
    all_signals = signals.to_dict('records')
    signals['market_context'] = agent.get_market_context(all_signals)
    
    return signals


def main(mode: str = "balanced", use_ai: bool = True, gemini_api_key: Optional[str] = None):
    """
    Main signal generation function.
    
    Args:
        mode: Prediction mode ("conservative", "balanced", "aggressive")
        use_ai: Whether to generate AI explanations
        gemini_api_key: Optional API key for Gemini
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    print("=" * 60, flush=True)
    print("🚀 GENERATING ADVANCED TRADING SIGNALS", flush=True)
    print("=" * 60, flush=True)
    print(f"Mode: {mode.upper()}", flush=True)
    print(f"AI Explanations: {'Enabled' if use_ai else 'Disabled'}", flush=True)
    
    rules = load_rulebook()
    uni = yaml.safe_load(open("config/universe.yaml"))
    tickers = [t for t in uni["tickers"] if not t.startswith("^")]
    
    print(f"\nProcessing {len(tickers)} tickers...\n", flush=True)
    
    rows = []
    for i, tk in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {tk}...", end=" ", flush=True)
        try:
            df = yf.download(tk, period="3y", interval="1d", auto_adjust=True, progress=False)
            if df is None or df.empty or len(df) < 260:
                print("⚠️ Insufficient data", flush=True)
                continue
            df = df.rename(columns=str.title)
            row = build_row(tk, df, rules, mode=mode)
            rows.append(row)
            
            # Print summary
            decision = row['soft_decision']
            strength = row['signal_strength']
            exp_ret = row['exp_30d_return'] * 100
            symbol = "📈" if decision == "LONG" else "📉"
            print(f"✓ {symbol} {decision} | Str:{strength:.0f} | Exp:{exp_ret:+.1f}%", flush=True)
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:30]}", flush=True)
            continue
    
    if len(rows) == 0:
        print("\n❌ No signals generated!", flush=True)
        return
    
    sig = pd.DataFrame(rows)
    
    # Determine actionable signals
    sig['actionable'] = (sig['decision'] != 'UNSURE') & (sig['gates_pass'])
    
    # Add action recommendation column
    sig['action'] = sig.apply(lambda r: 
        "🟢 BUY" if r['soft_decision'] == 'LONG' and r['gates_pass'] else
        "🔴 SELL" if r['soft_decision'] == 'SHORT' and r['gates_pass'] else
        "⚪ HOLD" if r['gates_pass'] else "⚠️ AVOID",
        axis=1
    )
    
    # Sort by signal strength
    sig = sig.sort_values(
        ["actionable", "signal_strength", "confidence"], 
        ascending=[False, False, False]
    )
    
    # Add AI explanations if requested
    if use_ai and GEMINI_AVAILABLE:
        print("\n🤖 Generating AI explanations...", flush=True)
        sig = generate_ai_explanations(sig, gemini_api_key)
    else:
        # Add placeholder columns
        sig['ai_explanation'] = ""
        sig['market_context'] = ""
    
    # Save signals
    os.makedirs("data", exist_ok=True)
    sig.to_csv("data/signals_latest.csv", index=False)
    
    # Save as JSON for dashboard
    sig.to_json("data/signals_latest.json", orient="records", indent=2)
    
    # Create summary statistics
    summary = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "total_signals": len(sig),
        "actionable": int(sig['actionable'].sum()),
        "buy_signals": int((sig['action'] == '🟢 BUY').sum()),
        "sell_signals": int((sig['action'] == '🔴 SELL').sum()),
        "hold_signals": int((sig['action'] == '⚪ HOLD').sum()),
        "avg_confidence": float(sig['confidence'].mean()),
        "avg_signal_strength": float(sig['signal_strength'].mean()),
        "top_picks": sig.head(5)[['ticker', 'action', 'signal_strength', 'exp_30d_return']].to_dict('records')
    }
    
    with open("data/signals_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"✅ Generated {len(sig)} signals", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"   🟢 Buy signals:  {summary['buy_signals']}", flush=True)
    print(f"   🔴 Sell signals: {summary['sell_signals']}", flush=True)
    print(f"   ⚪ Hold signals: {summary['hold_signals']}", flush=True)
    print(f"   📊 Actionable:   {summary['actionable']}", flush=True)
    print(f"\n   Avg Confidence:     {summary['avg_confidence']*100:.1f}%", flush=True)
    print(f"   Avg Signal Strength: {summary['avg_signal_strength']:.1f}/100", flush=True)
    
    print(f"\n📁 Files saved:", flush=True)
    print(f"   • data/signals_latest.csv", flush=True)
    print(f"   • data/signals_latest.json", flush=True)
    print(f"   • data/signals_summary.json", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Print top picks
    print(f"\n🏆 TOP 5 SIGNALS:", flush=True)
    for i, pick in enumerate(summary['top_picks'], 1):
        print(f"   {i}. {pick['ticker']}: {pick['action']} "
              f"(Str: {pick['signal_strength']:.0f}, Exp: {pick['exp_30d_return']*100:+.1f}%)", flush=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate trading signals")
    parser.add_argument("--mode", type=str, default="balanced",
                       choices=["conservative", "balanced", "aggressive"],
                       help="Prediction mode")
    parser.add_argument("--no-ai", action="store_true",
                       help="Disable AI explanations")
    parser.add_argument("--api-key", type=str, default=None,
                       help="Gemini API key")
    
    args = parser.parse_args()
    main(mode=args.mode, use_ai=not args.no_ai, gemini_api_key=args.api_key)
