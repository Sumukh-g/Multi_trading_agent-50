"""
⏱️ QuantEdge Pro - Short-Term Trading Dashboard
1-Week Prediction Horizon | Hourly Analysis | Multi-Agent System

Run with: streamlit run dashboard_shortterm.py --server.port 8502
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import subprocess
import sys
from datetime import datetime
import json

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="QuantEdge Pro | Short-Term Trading",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS STYLING
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a1f2e;
        --accent-orange: #f59e0b;
        --accent-purple: #8b5cf6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-cyan: #00d4ff;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border-color: #2d3748;
    }
    
    .stApp { background: linear-gradient(135deg, #0a0e17 0%, #2d1f3d 100%); }
    .main .block-container { padding: 1rem 2rem; max-width: 100%; }
    #MainMenu, footer, header { visibility: hidden; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0f2e 0%, #2d1f3d 100%);
        border-right: 1px solid #4a3f5d;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f59e0b 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .horizon-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f59e0b, #8b5cf6);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    .agent-container {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .agent-card {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .agent-card:hover {
        transform: translateX(5px);
        border-color: #8b5cf6;
    }
    
    .agent-name {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.25rem;
    }
    
    .agent-description {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    
    .best-pick-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 2px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .best-pick-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.2);
    }
    
    .best-pick-card.sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
        border-color: rgba(239, 68, 68, 0.4);
    }
    
    .best-pick-card.sell:hover {
        box-shadow: 0 15px 40px rgba(239, 68, 68, 0.2);
    }
    
    .ticker-xl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
    }
    
    .price-xl {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        color: #94a3b8;
    }
    
    .signal-badge-lg {
        display: inline-block;
        padding: 10px 25px;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .signal-badge-lg.buy {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .signal-badge-lg.sell {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .signal-badge-lg.neutral {
        background: linear-gradient(135deg, #64748b, #475569);
        color: white;
    }
    
    .metric-row {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .metric-box {
        background: rgba(0,0,0,0.3);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-value.positive { color: #10b981; }
    .metric-value.negative { color: #ef4444; }
    .metric-value.highlight { color: #f59e0b; }
    
    .risk-section {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .profit-section {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .explanation-panel {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .explanation-title {
        color: #00d4ff;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }
    
    .explanation-text {
        color: #94a3b8;
        line-height: 1.8;
    }
    
    .all-signals-row {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.2s ease;
    }
    
    .all-signals-row:hover {
        border-color: #8b5cf6;
        transform: translateX(3px);
    }
    
    .section-divider {
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #8b5cf6, transparent);
        margin: 2rem 0;
    }
    
    .indicator-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.75rem;
    }
    
    .indicator-card {
        background: rgba(0,0,0,0.2);
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    
    .indicator-card.bullish {
        border-color: rgba(16, 185, 129, 0.4);
        background: rgba(16, 185, 129, 0.1);
    }
    
    .indicator-card.bearish {
        border-color: rgba(239, 68, 68, 0.4);
        background: rgba(239, 68, 68, 0.1);
    }
    
    .regime-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .regime-badge.strong_trend { background: #3b82f6; color: white; }
    .regime-badge.mean_reverting { background: #8b5cf6; color: white; }
    .regime-badge.volatility_breakout { background: #f59e0b; color: white; }
    .regime-badge.illiquid_noisy { background: #64748b; color: white; }
    .regime-badge.normal { background: #374151; color: white; }
    
    .thesis-box {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .key-factor {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10b981;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .risk-factor {
        background: rgba(239, 68, 68, 0.1);
        border-left: 3px solid #ef4444;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_short_term_signals():
    if os.path.exists('data/short_term_signals.json'):
        try:
            with open('data/short_term_signals.json', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content or content == '':
                    return None
                return json.loads(content)
        except (json.JSONDecodeError, ValueError, IOError) as e:
            # If JSON is corrupted, delete it and return None
            try:
                os.remove('data/short_term_signals.json')
            except:
                pass
            return None
    return None

def load_advanced_signals():
    if os.path.exists('data/advanced_signals.json'):
        try:
            with open('data/advanced_signals.json', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content or content == '':
                    return None
                return json.loads(content)
        except (json.JSONDecodeError, ValueError, IOError) as e:
            # If JSON is corrupted, delete it and return None
            try:
                os.remove('data/advanced_signals.json')
            except:
                pass
            return None
    return None

def get_signal_class(signal):
    if 'BUY' in str(signal).upper():
        return 'buy'
    elif 'SELL' in str(signal).upper():
        return 'sell'
    return 'neutral'

def run_short_term_analysis():
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-m", "src.agents.short_term.orchestrator"],
            capture_output=True, text=True, timeout=600
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def run_advanced_analysis():
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-m", "src.agents.short_term.advanced_orchestrator"],
            capture_output=True, text=True, timeout=600
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem;">
            <div style="font-size: 3rem;">⏱️</div>
            <div style="font-family: 'Inter'; font-size: 1.3rem; font-weight: 800; 
                        background: linear-gradient(135deg, #f59e0b, #8b5cf6);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Short-Term Trading
            </div>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">
                1-Week Predictions | Hourly Data
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio("Navigation", [
            "🏠 Overview", 
            "🏆 Best Predictions", 
            "🤖 Agent Analysis", 
            "🧠 Advanced Agent",  # NEW
            "📊 All Signals", 
            "📖 Understanding", 
            "⚙️ Run Analysis"
        ], label_visibility="collapsed")
        
        st.markdown("---")
        
        # Quick stats
        signals = load_short_term_signals()
        adv_signals = load_advanced_signals()
        
        if signals:
            st.markdown("### 📊 Quick Stats")
            st.metric("Analyzed", len(signals))
            bullish = len([s for s in signals if 'BUY' in s.get('signal', '')])
            bearish = len([s for s in signals if 'SELL' in s.get('signal', '')])
            st.metric("Bullish", bullish)
            st.metric("Bearish", bearish)
            avg_conf = np.mean([s.get('confidence', 0) for s in signals])
            st.metric("Avg Confidence", f"{avg_conf*100:.0f}%")
        
        if adv_signals:
            st.markdown("### 🧠 Advanced Stats")
            regimes = {}
            for s in adv_signals:
                r = s.get('regime', 'unknown')
                regimes[r] = regimes.get(r, 0) + 1
            for r, c in regimes.items():
                st.caption(f"• {r.replace('_', ' ').title()}: {c}")
        
        st.markdown("---")
        st.markdown(f"<div style='color: #64748b; font-size: 0.75rem; text-align: center;'>"
                   f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)
    
    # Main content
    if page == "🏠 Overview":
        show_overview()
    elif page == "🏆 Best Predictions":
        show_best_predictions()
    elif page == "🤖 Agent Analysis":
        show_agent_analysis()
    elif page == "🧠 Advanced Agent":
        show_advanced_agent()
    elif page == "📊 All Signals":
        show_all_signals()
    elif page == "📖 Understanding":
        show_understanding()
    elif page == "⚙️ Run Analysis":
        show_run_analysis()

def show_overview():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">⏱️ Short-Term Trading Dashboard</h1>
        <p class="main-subtitle">Multi-Agent AI System for 1-Week Stock Predictions</p>
        <div class="horizon-badge">🎯 1-Week Prediction Horizon | Hourly Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    signals = load_short_term_signals()
    
    if signals is None:
        st.warning("No short-term signals available. Run the analysis first from the sidebar.")
        return
    
    # Summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📊 Total Analyzed", len(signals))
    with col2:
        bullish = len([s for s in signals if 'BUY' in s.get('signal', '')])
        st.metric("🟢 Bullish", bullish)
    with col3:
        bearish = len([s for s in signals if 'SELL' in s.get('signal', '')])
        st.metric("🔴 Bearish", bearish)
    with col4:
        neutral = len(signals) - bullish - bearish
        st.metric("⚪ Neutral", neutral)
    with col5:
        avg_conf = np.mean([s.get('confidence', 0) for s in signals])
        st.metric("📈 Avg Confidence", f"{avg_conf*100:.0f}%")
    with col6:
        avg_rr = np.mean([s.get('risk_reward_ratio', 0) for s in signals])
        st.metric("⚖️ Avg R/R Ratio", f"{avg_rr:.2f}")
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Best picks preview
    col1, col2 = st.columns(2)
    
    # Sort by composite score
    sorted_signals = sorted(signals, key=lambda x: x.get('strength', 0) * x.get('confidence', 0), reverse=True)
    buy_signals = [s for s in sorted_signals if 'BUY' in s.get('signal', '')]
    sell_signals = [s for s in sorted_signals if 'SELL' in s.get('signal', '')]
    
    with col1:
        st.markdown("### 🟢 Top Buy Opportunity")
        if buy_signals:
            top = buy_signals[0]
            st.markdown(f"""
            <div class="best-pick-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="ticker-xl">{top['ticker']}</div>
                        <div class="price-xl">${top.get('current_price', 0):.2f}</div>
                    </div>
                    <span class="signal-badge-lg buy">🚀 {top['signal']}</span>
                </div>
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-label">1-Week Return</div>
                        <div class="metric-value positive">{top.get('pred_return_1w', 0)*100:+.2f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{top.get('confidence', 0)*100:.0f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Strength</div>
                        <div class="metric-value highlight">{top.get('strength', 0):.0f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Risk/Reward</div>
                        <div class="metric-value">{top.get('risk_reward_ratio', 0):.2f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Position Size</div>
                        <div class="metric-value">{top.get('position_size_pct', 0)*100:.0f}%</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                    <div class="risk-section">
                        <div style="color: #ef4444; font-weight: 600;">⛔ Stop Loss</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">${top.get('stop_loss', 0):.2f}</div>
                    </div>
                    <div class="profit-section">
                        <div style="color: #10b981; font-weight: 600;">🎯 Take Profit</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">${top.get('take_profit_1', 0):.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No buy signals at this time")
    
    with col2:
        st.markdown("### 🔴 Top Sell/Short Opportunity")
        if sell_signals:
            top = sell_signals[0]
            st.markdown(f"""
            <div class="best-pick-card sell">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="ticker-xl">{top['ticker']}</div>
                        <div class="price-xl">${top.get('current_price', 0):.2f}</div>
                    </div>
                    <span class="signal-badge-lg sell">📉 {top['signal']}</span>
                </div>
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-label">1-Week Return</div>
                        <div class="metric-value negative">{top.get('pred_return_1w', 0)*100:+.2f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{top.get('confidence', 0)*100:.0f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Strength</div>
                        <div class="metric-value highlight">{top.get('strength', 0):.0f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Risk/Reward</div>
                        <div class="metric-value">{top.get('risk_reward_ratio', 0):.2f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Position Size</div>
                        <div class="metric-value">{top.get('position_size_pct', 0)*100:.0f}%</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                    <div class="risk-section">
                        <div style="color: #ef4444; font-weight: 600;">⛔ Stop Loss</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #ef4444;">${top.get('stop_loss', 0):.2f}</div>
                    </div>
                    <div class="profit-section">
                        <div style="color: #10b981; font-weight: 600;">🎯 Take Profit</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">${top.get('take_profit_1', 0):.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No sell signals at this time")
    
    # Multi-horizon predictions
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### 📅 Multi-Horizon Predictions")
    
    df = pd.DataFrame(signals)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 1-Day Outlook")
        if 'pred_return_1d' in df.columns:
            avg_1d = df['pred_return_1d'].mean() * 100
            color = '#10b981' if avg_1d > 0 else '#ef4444'
            st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: {color};'>{avg_1d:+.2f}%</div>", unsafe_allow_html=True)
            st.caption("Average expected return")
    
    with col2:
        st.markdown("#### 3-Day Outlook")
        if 'pred_return_3d' in df.columns:
            avg_3d = df['pred_return_3d'].mean() * 100
            color = '#10b981' if avg_3d > 0 else '#ef4444'
            st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: {color};'>{avg_3d:+.2f}%</div>", unsafe_allow_html=True)
            st.caption("Average expected return")
    
    with col3:
        st.markdown("#### 1-Week Outlook")
        if 'pred_return_1w' in df.columns:
            avg_1w = df['pred_return_1w'].mean() * 100
            color = '#10b981' if avg_1w > 0 else '#ef4444'
            st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: {color};'>{avg_1w:+.2f}%</div>", unsafe_allow_html=True)
            st.caption("Average expected return")

def show_best_predictions():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🏆 Best Predictions</h1>
        <p class="main-subtitle">Top-ranked signals from our multi-agent AI system</p>
    </div>
    """, unsafe_allow_html=True)
    
    signals = load_short_term_signals()
    if signals is None:
        st.warning("No signals available.")
        return
    
    # Sort by composite score
    sorted_signals = sorted(signals, key=lambda x: x.get('strength', 0) * x.get('confidence', 0), reverse=True)
    
    # Top Buys
    st.markdown("## 🟢 TOP BUY RECOMMENDATIONS")
    
    st.markdown("""
    <div class="explanation-panel">
        <div class="explanation-title">💡 Understanding Buy Signals</div>
        <div class="explanation-text">
            These stocks have the highest probability of upward movement over the next week.
            The multi-agent system has identified positive signals across multiple indicators:
            <ul style="margin-top: 0.5rem;">
                <li><b>Trend Agent</b>: Upward price trend with strong momentum</li>
                <li><b>Momentum Agent</b>: RSI, Stochastic showing bullish conditions</li>
                <li><b>Volume Agent</b>: Accumulation patterns detected</li>
                <li><b>Pattern Agent</b>: Bullish candlestick/chart patterns</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    buy_signals = [s for s in sorted_signals if 'BUY' in s.get('signal', '')]
    
    for i, sig in enumerate(buy_signals[:5], 1):
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #10b981, #059669); border-radius: 16px; padding: 1.5rem; text-align: center; height: 100%;">
                    <div style="font-size: 2rem; font-weight: 800; color: white;">#{i}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: white; margin-top: 0.5rem;">{sig['ticker']}</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">${sig.get('current_price', 0):.2f}</div>
                    <div style="margin-top: 1rem; font-weight: 600; color: white;">{sig['signal']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**📊 Prediction Details**")
                mcols = st.columns(6)
                mcols[0].metric("1D Return", f"{sig.get('pred_return_1d', 0)*100:+.2f}%")
                mcols[1].metric("3D Return", f"{sig.get('pred_return_3d', 0)*100:+.2f}%")
                mcols[2].metric("1W Return", f"{sig.get('pred_return_1w', 0)*100:+.2f}%")
                mcols[3].metric("Confidence", f"{sig.get('confidence', 0)*100:.0f}%")
                mcols[4].metric("Strength", f"{sig.get('strength', 0):.0f}")
                mcols[5].metric("R/R Ratio", f"{sig.get('risk_reward_ratio', 0):.2f}")
                
                st.markdown("**💰 Risk Management**")
                rcols = st.columns(4)
                rcols[0].metric("Stop Loss", f"${sig.get('stop_loss', 0):.2f}")
                rcols[1].metric("Take Profit 1", f"${sig.get('take_profit_1', 0):.2f}")
                rcols[2].metric("Take Profit 2", f"${sig.get('take_profit_2', 0):.2f}")
                rcols[3].metric("Position Size", f"{sig.get('position_size_pct', 0)*100:.0f}%")
        
        st.markdown("---")
    
    # Top Sells
    st.markdown("## 🔴 TOP SELL/SHORT RECOMMENDATIONS")
    
    st.markdown("""
    <div class="explanation-panel" style="border-color: rgba(239, 68, 68, 0.3); background: rgba(239, 68, 68, 0.1);">
        <div class="explanation-title" style="color: #ef4444;">💡 Understanding Sell Signals</div>
        <div class="explanation-text">
            These stocks show the highest probability of downward movement.
            Consider reducing long positions or initiating short positions.
            The agents have identified:
            <ul style="margin-top: 0.5rem;">
                <li>Weakening price trends</li>
                <li>Overbought momentum conditions</li>
                <li>Distribution patterns in volume</li>
                <li>Bearish chart patterns</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    sell_signals = [s for s in sorted_signals if 'SELL' in s.get('signal', '')]
    
    for i, sig in enumerate(sell_signals[:5], 1):
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ef4444, #dc2626); border-radius: 16px; padding: 1.5rem; text-align: center; height: 100%;">
                    <div style="font-size: 2rem; font-weight: 800; color: white;">#{i}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: white; margin-top: 0.5rem;">{sig['ticker']}</div>
                    <div style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">${sig.get('current_price', 0):.2f}</div>
                    <div style="margin-top: 1rem; font-weight: 600; color: white;">{sig['signal']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**📊 Prediction Details**")
                mcols = st.columns(6)
                mcols[0].metric("1D Return", f"{sig.get('pred_return_1d', 0)*100:+.2f}%")
                mcols[1].metric("3D Return", f"{sig.get('pred_return_3d', 0)*100:+.2f}%")
                mcols[2].metric("1W Return", f"{sig.get('pred_return_1w', 0)*100:+.2f}%")
                mcols[3].metric("Confidence", f"{sig.get('confidence', 0)*100:.0f}%")
                mcols[4].metric("Strength", f"{sig.get('strength', 0):.0f}")
                mcols[5].metric("R/R Ratio", f"{sig.get('risk_reward_ratio', 0):.2f}")
                
                st.markdown("**💰 Risk Management**")
                rcols = st.columns(4)
                rcols[0].metric("Stop Loss", f"${sig.get('stop_loss', 0):.2f}")
                rcols[1].metric("Take Profit", f"${sig.get('take_profit_1', 0):.2f}")
        
        st.markdown("---")

def show_agent_analysis():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🤖 Multi-Agent Analysis</h1>
        <p class="main-subtitle">Understanding how our 5 specialized AI agents work together</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Agent explanations
    agents = [
        {
            "name": "📈 Trend Agent",
            "color": "#3b82f6",
            "weight": "23%",
            "focus": "Trend Detection & Following",
            "indicators": ["EMA (8/21/55)", "MACD", "ADX", "Supertrend"],
            "indicator_weights": {"EMA": "30%", "MACD": "30%", "ADX": "25%", "Supertrend": "15%"},
            "description": "Identifies and follows the primary price trend. Uses research-backed weights favoring EMA and MACD for their proven reliability."
        },
        {
            "name": "🚀 Momentum Agent",
            "color": "#8b5cf6",
            "weight": "19%",
            "focus": "Momentum & Overbought/Oversold",
            "indicators": ["RSI (14)", "Stochastic K/D", "Rate of Change", "Divergences"],
            "indicator_weights": {"RSI": "35%", "Stochastic": "25%", "ROC": "20%", "Divergences": "20%"},
            "description": "Detects overbought/oversold conditions and momentum shifts. RSI weighted highest due to extensive research validation."
        },
        {
            "name": "📊 Volatility Agent",
            "color": "#f59e0b",
            "weight": "19%",
            "focus": "Volatility & Mean Reversion",
            "indicators": ["Bollinger Bands", "Keltner Channels", "Squeeze Detection"],
            "indicator_weights": {"Bollinger": "35%", "Keltner": "25%", "Squeeze": "40%"},
            "description": "Analyzes volatility regimes and squeeze breakouts. Squeeze detection heavily weighted as it signals high-probability moves."
        },
        {
            "name": "📦 Volume Agent",
            "color": "#10b981",
            "weight": "17%",
            "focus": "Volume & Accumulation/Distribution",
            "indicators": ["OBV", "VWAP", "Chaikin Money Flow", "Volume Climax"],
            "indicator_weights": {"OBV": "25%", "VWAP": "30%", "CMF": "25%", "Climax": "20%"},
            "description": "Tracks smart money flow. VWAP weighted highest as institutional benchmark. Climax signals potential reversals."
        },
        {
            "name": "🎯 Pattern Agent",
            "color": "#ec4899",
            "weight": "21%",
            "focus": "Price Patterns & Levels",
            "indicators": ["Candlesticks", "Support/Resistance", "Breakouts"],
            "indicator_weights": {"Candlesticks": "30%", "S/R": "35%", "Breakouts": "35%"},
            "description": "Recognizes chart patterns and key levels. Breakouts and S/R equally weighted as they often trigger significant moves."
        }
    ]
    
    for agent in agents:
        st.markdown(f"""
        <div class="agent-container" style="border-color: {agent['color']}40; background: linear-gradient(135deg, {agent['color']}10 0%, {agent['color']}05 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {agent['color']};">{agent['name']}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">{agent['focus']}</div>
                </div>
                <div style="background: {agent['color']}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: 600;">
                    Global Weight: {agent['weight']}
                </div>
            </div>
            <div style="color: #f8fafc; margin-bottom: 1rem;">{agent['description']}</div>
            <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
        """, unsafe_allow_html=True)
        
        # Show indicators with weights
        ind_html = ""
        for ind, weight in agent['indicator_weights'].items():
            ind_html += f'<span style="background: {agent["color"]}30; color: {agent["color"]}; padding: 4px 12px; border-radius: 15px; font-size: 0.85rem;">{ind} ({weight})</span> '
        
        st.markdown(f"""
            {ind_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # How they work together
    st.markdown("---")
    st.markdown("### 🔗 How Agents Combine Signals")
    
    st.markdown("""
    <div class="explanation-panel">
        <div class="explanation-title">Signal Combination Process</div>
        <div class="explanation-text">
            <ol style="margin-top: 0.5rem;">
                <li><b>Individual Analysis</b>: Each agent analyzes the stock independently using its specialized indicators</li>
                <li><b>Weighted Scoring</b>: Indicators are scored and weighted according to research-backed priors</li>
                <li><b>Agent Aggregation</b>: Agent scores are combined using global weights (Trend 23%, Pattern 21%, etc.)</li>
                <li><b>Regime Adjustment</b>: Weights are dynamically adjusted based on market regime</li>
                <li><b>Final Signal</b>: Combined score generates BUY/SELL/NEUTRAL with confidence</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Global weights table
    st.markdown("### 📊 Global Indicator Weights")
    
    weights_data = {
        "Indicator": ["EMA", "MACD", "ADX", "Supertrend", "RSI", "Stochastic", "ROC", "Divergences",
                      "Bollinger", "Keltner", "Squeeze", "OBV", "VWAP", "CMF", "Vol Climax",
                      "Candlesticks", "S/R", "Breakouts"],
        "Agent": ["Trend", "Trend", "Trend", "Trend", "Momentum", "Momentum", "Momentum", "Momentum",
                  "Volatility", "Volatility", "Volatility", "Volume", "Volume", "Volume", "Volume",
                  "Pattern", "Pattern", "Pattern"],
        "Weight": [6.9, 6.9, 5.8, 3.5, 6.7, 4.8, 3.8, 3.8, 6.7, 4.8, 7.7, 4.3, 5.2, 4.3, 3.5, 6.3, 7.4, 7.4]
    }
    
    df = pd.DataFrame(weights_data)
    
    fig = px.bar(df, x='Indicator', y='Weight', color='Agent',
                 color_discrete_map={
                     'Trend': '#3b82f6', 'Momentum': '#8b5cf6', 
                     'Volatility': '#f59e0b', 'Volume': '#10b981', 'Pattern': '#ec4899'
                 },
                 title="Research-Backed Indicator Weights (%)")
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        title_font_color='#f8fafc',
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)


def show_advanced_agent():
    """Show the advanced weighted agent analysis."""
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🧠 Advanced Agent Analysis</h1>
        <p class="main-subtitle">18 Indicators | Dynamic Weights | Regime Detection | Trade Explanations</p>
    </div>
    """, unsafe_allow_html=True)
    
    signals = load_advanced_signals()
    
    if signals is None or not isinstance(signals, list) or len(signals) == 0:
        st.warning("No advanced signals available. Run the Advanced Analysis first.")
        
        if st.button("🚀 Run Advanced Analysis Now", type="primary"):
            with st.spinner("Running advanced multi-agent analysis..."):
                success, output = run_advanced_analysis()
                if success:
                    st.success("Advanced analysis complete!")
                    st.rerun()
                else:
                    st.error(f"Analysis failed: {output}")
        return
    
    # Ticker selector
    try:
        tickers = [s.get('ticker', 'UNKNOWN') for s in signals if isinstance(s, dict)]
        if not tickers:
            st.error("No valid tickers found in signals data.")
            return
        
        selected_ticker = st.selectbox("🎯 Select Stock for Detailed Analysis", tickers)
        
        sig = next((s for s in signals if isinstance(s, dict) and s.get('ticker') == selected_ticker), None)
        if sig is None:
            st.error("Signal not found")
            return
    except Exception as e:
        st.error(f"Error loading signal data: {e}")
        return
    
    # Main signal card
    signal_class = get_signal_class(sig['signal'])
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        border_color = '#10b981' if signal_class == 'buy' else '#ef4444' if signal_class == 'sell' else '#64748b'
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {border_color}20, {border_color}05); 
                    border: 2px solid {border_color}; border-radius: 16px; padding: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="ticker-xl">{sig['ticker']}</div>
                    <div class="price-xl">${sig['current_price']:.2f}</div>
                </div>
                <div>
                    <span class="signal-badge-lg {signal_class}">{sig['signal']}</span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.5rem;">
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{sig['confidence']*100:.0f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Strength</div>
                    <div class="metric-value highlight">{sig['strength']:.1f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">1W Return</div>
                    <div class="metric-value {'positive' if sig['pred_return_1w'] > 0 else 'negative'}">{sig['pred_return_1w']*100:+.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">R/R Ratio</div>
                    <div class="metric-value">{sig['risk_reward_ratio']:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        regime = sig['regime'].replace('_', ' ').title()
        regime_class = sig['regime']
        st.markdown(f"""
        <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid rgba(139, 92, 246, 0.3); 
                    border-radius: 16px; padding: 1.5rem; height: 100%;">
            <div style="color: #8b5cf6; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.5rem;">MARKET REGIME</div>
            <span class="regime-badge {regime_class}">{regime}</span>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 1rem; line-height: 1.6;">
                {sig.get('regime_explanation', '')[:150]}...
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sector = sig['sector'].replace('_', ' ').title()
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); 
                    border-radius: 16px; padding: 1.5rem; height: 100%;">
            <div style="color: #f59e0b; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.5rem;">SECTOR</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">{sector}</div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 1rem; line-height: 1.6;">
                {sig.get('sector_impact', '')[:150]}...
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Trade Thesis
    st.markdown("### 📝 Trade Thesis & Explanation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="thesis-box">
            <div style="font-size: 1.1rem; font-weight: 700; color: #00d4ff; margin-bottom: 1rem;">
                💡 Why This Trade?
            </div>
            <div style="color: #f8fafc; line-height: 1.8; white-space: pre-wrap;">
{sig.get('trade_thesis', 'No thesis available')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Key factors
        st.markdown("#### ✅ Key Bullish/Bearish Factors")
        key_factors = sig.get('key_factors', [])
        for factor in key_factors[:5]:
            st.markdown(f"""<div class="key-factor">{factor}</div>""", unsafe_allow_html=True)
        
        # Risk factors
        st.markdown("#### ⚠️ Risk Factors")
        risk_factors = sig.get('risk_factors', [])
        for factor in risk_factors[:5]:
            st.markdown(f"""<div class="risk-factor">{factor}</div>""", unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Agent Breakdown
    st.markdown("### 🤖 Agent Contributions")
    
    agent_scores = sig.get('agent_scores', {})
    
    cols = st.columns(5)
    agent_colors = {
        'trend': '#3b82f6', 'momentum': '#8b5cf6', 
        'volatility': '#f59e0b', 'volume': '#10b981', 'pattern': '#ec4899'
    }
    
    for i, (agent_key, agent_data) in enumerate(agent_scores.items()):
        color = agent_colors.get(agent_key, '#64748b')
        signal_badge = "🟢" if agent_data['signal'] == "BUY" else "🔴" if agent_data['signal'] == "SELL" else "⚪"
        
        with cols[i]:
            st.markdown(f"""
            <div style="background: {color}15; border: 1px solid {color}40; border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="font-size: 1.5rem;">{signal_badge}</div>
                <div style="color: {color}; font-weight: 700; font-size: 0.9rem; margin: 0.5rem 0;">
                    {agent_data['name'].replace(' Agent', '')}
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc;">
                    {agent_data['score']*100:+.1f}
                </div>
                <div style="color: #64748b; font-size: 0.8rem;">
                    Weight: {agent_data['weight']*100:.0f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.5rem;">
                    Conf: {agent_data['confidence']*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Indicator Grid
    st.markdown("### 📊 All 18 Indicators")
    
    indicator_details = sig.get('indicator_details', [])
    
    # Show indicators by agent
    tabs = st.tabs(["📈 Trend", "🚀 Momentum", "📊 Volatility", "📦 Volume", "🎯 Pattern"])
    
    agent_names = ['trend', 'momentum', 'volatility', 'volume', 'pattern']
    
    for tab, agent_name in zip(tabs, agent_names):
        with tab:
            agent_indicators = [ind for ind in indicator_details if ind['agent'] == agent_name]
            
            cols = st.columns(len(agent_indicators) if agent_indicators else 1)
            
            for i, ind in enumerate(agent_indicators):
                bullish_class = "bullish" if ind['is_bullish'] else "bearish"
                icon = "🟢" if ind['is_bullish'] else "🔴"
                
                with cols[i]:
                    st.markdown(f"""
                    <div class="indicator-card {bullish_class}">
                        <div style="font-size: 1.5rem;">{icon}</div>
                        <div style="font-weight: 700; color: #f8fafc; margin: 0.5rem 0;">{ind['name']}</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: {'#10b981' if ind['is_bullish'] else '#ef4444'};">
                            {ind['score']*100:+.1f}
                        </div>
                        <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.5rem;">
                            Weight: {ind['weight']*100:.1f}%
                        </div>
                        <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.5rem; line-height: 1.4;">
                            {ind['explanation'][:60]}{'...' if len(ind['explanation']) > 60 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Dynamic Weight Adjustments
    st.markdown("### ⚖️ Dynamic Weight Adjustments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Regime Multipliers")
        regime_mults = sig.get('regime_multipliers', {})
        if regime_mults:
            df_regime = pd.DataFrame([
                {"Agent": k.title(), "Multiplier": f"{v:.1f}x", "Impact": "↑ Boosted" if v > 1 else "↓ Reduced" if v < 1 else "→ Normal"}
                for k, v in regime_mults.items()
            ])
            st.dataframe(df_regime, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### Sector Multipliers")
        sector_mults = sig.get('sector_multipliers', {})
        if sector_mults:
            df_sector = pd.DataFrame([
                {"Agent": k.title(), "Multiplier": f"{v:.1f}x", "Impact": "↑ Boosted" if v > 1 else "↓ Reduced" if v < 1 else "→ Normal"}
                for k, v in sector_mults.items()
            ])
            st.dataframe(df_sector, hide_index=True, use_container_width=True)
    
    # Effective Weights
    st.markdown("#### Final Effective Agent Weights")
    effective = sig.get('effective_agent_weights', {})
    if effective:
        fig = go.Figure(data=[go.Pie(
            labels=[k.title() for k in effective.keys()],
            values=[v*100 for v in effective.values()],
            marker_colors=['#3b82f6', '#8b5cf6', '#f59e0b', '#10b981', '#ec4899'],
            hole=0.4
        )])
        fig.update_layout(
            title="Effective Agent Weights After Regime & Sector Adjustments",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            title_font_color='#f8fafc'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    
    # Risk Management
    st.markdown("### 💰 Risk Management")
    
    cols = st.columns(5)
    cols[0].metric("Stop Loss", f"${sig['stop_loss']:.2f}")
    cols[1].metric("Take Profit 1", f"${sig['take_profit_1']:.2f}")
    cols[2].metric("Take Profit 2", f"${sig['take_profit_2']:.2f}")
    cols[3].metric("Position Size", f"{sig['position_size_pct']*100:.1f}%")
    cols[4].metric("Risk/Reward", f"{sig['risk_reward_ratio']:.2f}")
    
    # Summary for all tickers
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### 📋 All Advanced Signals Summary")
    
    summary_data = []
    for s in signals:
        summary_data.append({
            "Ticker": s['ticker'],
            "Signal": s['signal'],
            "Confidence": f"{s['confidence']*100:.0f}%",
            "Strength": f"{s['strength']:.1f}",
            "Regime": s['regime'].replace('_', ' ').title(),
            "Bullish": s['bullish_indicators'],
            "Bearish": s['bearish_indicators'],
            "1W Return": f"{s['pred_return_1w']*100:+.2f}%"
        })
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, hide_index=True, use_container_width=True)


def show_all_signals():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">📊 All Signals</h1>
        <p class="main-subtitle">Complete list of analyzed stocks with all details</p>
    </div>
    """, unsafe_allow_html=True)
    
    signals = load_short_term_signals()
    if signals is None:
        st.warning("No signals available.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_types = list(set(s.get('signal', 'NEUTRAL') for s in signals))
        selected_types = st.multiselect("Signal Type", signal_types, default=signal_types)
    
    with col2:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
    
    with col3:
        sort_by = st.selectbox("Sort By", ["strength", "confidence", "pred_return_1w", "ticker"])
    
    # Filter and sort
    filtered = [s for s in signals if s.get('signal', 'NEUTRAL') in selected_types and s.get('confidence', 0) >= min_conf]
    
    if sort_by == "ticker":
        filtered = sorted(filtered, key=lambda x: x.get('ticker', ''))
    else:
        filtered = sorted(filtered, key=lambda x: x.get(sort_by, 0), reverse=True)
    
    st.markdown(f"### Showing {len(filtered)} signals")
    
    # Display signals
    for sig in filtered:
        signal_class = get_signal_class(sig.get('signal', ''))
        border_color = '#10b981' if signal_class == 'buy' else '#ef4444' if signal_class == 'sell' else '#64748b'
        ret_1w = sig.get('pred_return_1w', 0)
        ret_class = 'positive' if ret_1w > 0 else 'negative'
        
        st.markdown(f"""
        <div class="all-signals-row" style="border-left: 4px solid {border_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-family: 'JetBrains Mono'; font-size: 1.4rem; font-weight: 700; color: #f8fafc;">{sig['ticker']}</span>
                    <span style="color: #94a3b8; margin-left: 1rem; font-size: 1.1rem;">${sig.get('current_price', 0):.2f}</span>
                </div>
                <span class="signal-badge-lg {signal_class}">{sig.get('signal', 'NEUTRAL')}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(8, 1fr); gap: 0.75rem; margin-top: 1rem;">
                <div class="metric-box">
                    <div class="metric-label">1D Return</div>
                    <div class="metric-value {'positive' if sig.get('pred_return_1d', 0) > 0 else 'negative'}">{sig.get('pred_return_1d', 0)*100:+.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">3D Return</div>
                    <div class="metric-value {'positive' if sig.get('pred_return_3d', 0) > 0 else 'negative'}">{sig.get('pred_return_3d', 0)*100:+.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">1W Return</div>
                    <div class="metric-value {ret_class}">{ret_1w*100:+.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{sig.get('confidence', 0)*100:.0f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Strength</div>
                    <div class="metric-value highlight">{sig.get('strength', 0):.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Stop Loss</div>
                    <div class="metric-value negative">${sig.get('stop_loss', 0):.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Take Profit</div>
                    <div class="metric-value positive">${sig.get('take_profit_1', 0):.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">R/R Ratio</div>
                    <div class="metric-value">{sig.get('risk_reward_ratio', 0):.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Download
    df = pd.DataFrame(filtered)
    st.download_button("📥 Download Signals (CSV)", df.to_csv(index=False), "short_term_signals.csv", "text/csv")

def show_understanding():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">📖 Understanding the System</h1>
        <p class="main-subtitle">Learn how to interpret and use the short-term predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Key Metrics Explained")
    
    st.markdown("""
    | Metric | Description | How to Use |
    |--------|-------------|------------|
    | **Signal** | BUY/SELL/NEUTRAL recommendation | Primary trading direction |
    | **Confidence** | Model certainty (0-100%) | Higher = more reliable signal |
    | **Strength** | Combined agent score (0-100) | Higher = stronger conviction |
    | **1D/3D/1W Return** | Expected price change | Predicted return over timeframe |
    | **Stop Loss** | Suggested exit if trade goes wrong | Place stop order here |
    | **Take Profit** | Suggested profit target | Consider taking profits here |
    | **R/R Ratio** | Risk/Reward ratio | >1 = potential reward > risk |
    | **Position Size** | Suggested portfolio allocation | Based on risk parameters |
    """)
    
    st.markdown("### 🧠 Advanced Features")
    
    st.markdown("""
    <div class="explanation-panel">
        <div class="explanation-title">Dynamic Weight System</div>
        <div class="explanation-text">
            The advanced agent uses 18 research-backed indicators across 5 categories:
            <ul style="margin-top: 0.5rem;">
                <li><b>Trend (23%)</b>: EMA, MACD, ADX, Supertrend</li>
                <li><b>Pattern (21%)</b>: Candlesticks, Support/Resistance, Breakouts</li>
                <li><b>Momentum (19%)</b>: RSI, Stochastic, ROC, Divergences</li>
                <li><b>Volatility (19%)</b>: Bollinger, Keltner, Squeeze</li>
                <li><b>Volume (17%)</b>: OBV, VWAP, CMF, Climax</li>
            </ul>
            Weights are dynamically adjusted based on:
            <ul>
                <li><b>Market Regime</b>: Trending markets boost Trend Agent, ranging markets boost Volatility</li>
                <li><b>Sector</b>: Tech/Growth favors momentum, Defensive favors mean-reversion</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 How to Trade These Signals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="explanation-panel" style="background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3);">
            <div class="explanation-title" style="color: #10b981;">🟢 For BUY Signals</div>
            <div class="explanation-text">
                <ol>
                    <li>Check confidence is >50%</li>
                    <li>Verify R/R ratio is >1.0</li>
                    <li>Use suggested position size</li>
                    <li>Set stop loss at indicated level</li>
                    <li>Consider taking partial profits at Take Profit 1</li>
                    <li>Trail stop for remaining position</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="explanation-panel" style="background: rgba(239, 68, 68, 0.1); border-color: rgba(239, 68, 68, 0.3);">
            <div class="explanation-title" style="color: #ef4444;">🔴 For SELL Signals</div>
            <div class="explanation-text">
                <ol>
                    <li>If holding long: consider reducing position</li>
                    <li>For shorting: ensure high confidence</li>
                    <li>Use tighter stops for shorts</li>
                    <li>Watch for reversal patterns</li>
                    <li>Be aware of short squeeze risk</li>
                    <li>Cover at Take Profit level</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### ⚠️ Important Disclaimers")
    
    st.warning("""
    **Risk Warning**: Trading stocks involves substantial risk of loss. These signals are generated by AI models 
    and should not be considered financial advice. Always:
    - Do your own research
    - Never risk more than you can afford to lose
    - Use proper position sizing
    - Set stop losses
    - Consider market conditions
    """)

def show_run_analysis():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">⚙️ Run Analysis</h1>
        <p class="main-subtitle">Generate fresh short-term predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🚀 Execute Analysis")
        
        st.markdown("#### Standard Multi-Agent")
        if st.button("🔄 Run Standard Analysis", use_container_width=True):
            with st.spinner("Running multi-agent analysis..."):
                success, output = run_short_term_analysis()
                st.session_state['analysis_log'] = output
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("#### 🧠 Advanced Weighted Agent")
        if st.button("🚀 Run Advanced Analysis", use_container_width=True, type="primary"):
            with st.spinner("Running advanced 18-indicator analysis with regime detection..."):
                success, output = run_advanced_analysis()
                st.session_state['adv_analysis_log'] = output
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("""
        **Standard Analysis:**
        - 5 specialized agents
        - Basic signal combination
        - ~2-3 minutes
        
        **Advanced Analysis:**
        - 18 research-backed indicators
        - Dynamic weight adjustments
        - Regime & sector detection
        - Trade explanations
        - ~3-5 minutes
        """)
    
    with col2:
        st.markdown("### 📋 Execution Log")
        
        tab1, tab2 = st.tabs(["Standard Log", "Advanced Log"])
        
        with tab1:
            if 'analysis_log' in st.session_state:
                st.code(st.session_state['analysis_log'])
            else:
                st.info("Run standard analysis to see output")
        
        with tab2:
            if 'adv_analysis_log' in st.session_state:
                st.code(st.session_state['adv_analysis_log'])
            else:
                st.info("Run advanced analysis to see output")

if __name__ == "__main__":
    main()
