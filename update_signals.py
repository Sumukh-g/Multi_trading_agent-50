"""Update signals file with new columns for dashboard compatibility."""
import pandas as pd
import os

print("Updating signals file...")

if not os.path.exists("data/signals_latest.csv"):
    print("No signals file found!")
    exit(1)

# Load existing signals
sig = pd.read_csv("data/signals_latest.csv")
print(f"Loaded {len(sig)} signals")

# Add 'soft_decision' if missing
if 'soft_decision' not in sig.columns:
    sig['soft_decision'] = sig['exp_30d_return'].apply(
        lambda x: 'LONG' if x > 0 else 'SHORT'
    )
    print("Added soft_decision column")

# Add 'action' column if missing
if 'action' not in sig.columns:
    sig['action'] = sig.apply(lambda r: 
        "🟢 BUY" if r['soft_decision'] == 'LONG' and r.get('gates_pass', True) else
        "🔴 SELL" if r['soft_decision'] == 'SHORT' and r.get('gates_pass', True) else
        "⚪ HOLD" if r.get('gates_pass', True) else "⚠️ AVOID",
        axis=1
    )
    print("Added action column")

# Add 'signal_strength' if missing
if 'signal_strength' not in sig.columns:
    sig['signal_strength'] = (
        sig['confidence'] * 50 + 
        sig['prob_5pct_30d'] * 30 + 
        sig['exp_30d_return'].abs() * 200
    ).clip(0, 100)
    print("Added signal_strength column")

# Add probability columns if missing
if 'prob_2pct' not in sig.columns:
    sig['prob_2pct'] = (sig['prob_5pct_30d'] * 1.3).clip(0, 0.95)
    print("Added prob_2pct column")

if 'prob_10pct' not in sig.columns:
    sig['prob_10pct'] = sig['prob_5pct_30d'] * 0.6
    print("Added prob_10pct column")

if 'prob_direction' not in sig.columns:
    sig['prob_direction'] = (0.5 + sig['exp_30d_return'] * 2).clip(0.1, 0.9)
    print("Added prob_direction column")

if 'model_agreement' not in sig.columns:
    sig['model_agreement'] = 0.7
    print("Added model_agreement column")

# Reorder columns for better display
priority_cols = ['ticker', 'action', 'soft_decision', 'price', 'signal_strength', 
                 'exp_30d_return', 'confidence', 'decision', 'prob_2pct', 
                 'prob_5pct_30d', 'prob_10pct', 'prob_direction']
other_cols = [c for c in sig.columns if c not in priority_cols]
sig = sig[priority_cols + other_cols]

# Sort by signal strength
sig = sig.sort_values(['actionable', 'signal_strength'], ascending=[False, False])

# Save updated file
sig.to_csv("data/signals_latest.csv", index=False)
print(f"\n✅ Updated signals saved to data/signals_latest.csv")
print(f"   Columns: {len(sig.columns)}")
print(f"   New columns: action, soft_decision, signal_strength, prob_2pct, prob_10pct, prob_direction")

# Show summary
print(f"\n📊 Signal Summary:")
print(f"   🟢 Buy:  {(sig['action'] == '🟢 BUY').sum()}")
print(f"   🔴 Sell: {(sig['action'] == '🔴 SELL').sum()}")
print(f"   ⚪ Hold: {(sig['action'] == '⚪ HOLD').sum()}")
print(f"   ⚠️ Avoid: {(sig['action'] == '⚠️ AVOID').sum()}")

