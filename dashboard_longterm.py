"""
📊 QuantEdge Pro - Long-Term Trading Dashboard
30-Day Prediction Horizon | Daily Analysis

Run with: streamlit run dashboard_longterm.py --server.port 8501
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
    page_title="QuantEdge Pro | Long-Term Trading",
    page_icon="📈",
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
        --accent-cyan: #00d4ff;
        --accent-purple: #8b5cf6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
        --accent-blue: #3b82f6;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-color: #2d3748;
    }
    
    .stApp { background: linear-gradient(135deg, #0a0e17 0%, #1a1033 100%); }
    .main .block-container { padding: 1rem 2rem; max-width: 100%; }
    #MainMenu, footer, header { visibility: hidden; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 100%);
        border-right: 1px solid #2d3748;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #10b981 100%);
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
        background: linear-gradient(135deg, #3b82f6, #10b981);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    .best-picks-container {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 2px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .best-pick-card {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .best-pick-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
    }
    
    .best-pick-card.sell {
        background: rgba(239, 68, 68, 0.15);
        border-color: rgba(239, 68, 68, 0.3);
    }
    
    .best-pick-card.sell:hover {
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.2);
    }
    
    .ticker-large {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .price-large {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        color: #94a3b8;
    }
    
    .signal-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
    }
    
    .signal-badge.buy {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    
    .signal-badge.sell {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }
    
    .signal-badge.hold {
        background: linear-gradient(135deg, #64748b, #475569);
        color: white;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .metric-item {
        text-align: center;
        padding: 0.75rem;
        background: rgba(0,0,0,0.2);
        border-radius: 8px;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
    }
    
    .metric-value.positive { color: #10b981; }
    .metric-value.negative { color: #ef4444; }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }
    
    .explanation-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .explanation-title {
        color: #3b82f6;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .explanation-text {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
    }
    
    .info-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .risk-meter {
        height: 8px;
        background: #1a1f2e;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .risk-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .all-signals-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_signals():
    if os.path.exists('data/signals_latest.csv'):
        return pd.read_csv('data/signals_latest.csv')
    return None

def load_portfolio():
    if os.path.exists('data/portfolio_latest.csv'):
        return pd.read_csv('data/portfolio_latest.csv')
    return None

def load_uncertainty():
    if os.path.exists('models/uncertainty.json'):
        with open('models/uncertainty.json', 'r') as f:
            return json.load(f)
    return None

def get_signal_class(action):
    if 'BUY' in str(action).upper() or action == 'LONG':
        return 'buy'
    elif 'SELL' in str(action).upper() or action == 'SHORT':
        return 'sell'
    return 'hold'

def run_pipeline(cmd):
    try:
        result = subprocess.run([sys.executable, "-u"] + cmd, capture_output=True, text=True, timeout=600)
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
            <div style="font-size: 3rem;">📈</div>
            <div style="font-family: 'Inter'; font-size: 1.3rem; font-weight: 800; 
                        background: linear-gradient(135deg, #3b82f6, #10b981);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Long-Term Trading
            </div>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;">
                30-Day Predictions
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio("", ["🏠 Overview", "🏆 Best Picks", "📊 All Signals", "💼 Portfolio", "🔬 Model Details", "⚙️ Run Pipeline"])
        
        st.markdown("---")
        
        # Quick stats
        signals = load_signals()
        if signals is not None:
            st.markdown("### 📊 Quick Stats")
            st.metric("Total Signals", len(signals))
            if 'actionable' in signals.columns:
                st.metric("Actionable", int(signals['actionable'].sum()))
            if 'confidence' in signals.columns:
                st.metric("Avg Confidence", f"{signals['confidence'].mean()*100:.1f}%")
        
        st.markdown("---")
        st.markdown(f"<div style='color: #64748b; font-size: 0.75rem; text-align: center;'>"
                   f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)
    
    # Main content
    if page == "🏠 Overview":
        show_overview()
    elif page == "🏆 Best Picks":
        show_best_picks()
    elif page == "📊 All Signals":
        show_all_signals()
    elif page == "💼 Portfolio":
        show_portfolio()
    elif page == "🔬 Model Details":
        show_model_details()
    elif page == "⚙️ Run Pipeline":
        show_pipeline()

def show_overview():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">📈 Long-Term Trading Dashboard</h1>
        <p class="main-subtitle">AI-Powered 30-Day Stock Predictions with Uncertainty Quantification</p>
        <div class="horizon-badge">🎯 30-Day Prediction Horizon</div>
    </div>
    """, unsafe_allow_html=True)
    
    signals = load_signals()
    uncertainty = load_uncertainty()
    
    if signals is None:
        st.warning("No signals available. Run the pipeline first from the sidebar.")
        return
    
    # Summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    action_col = 'action' if 'action' in signals.columns else 'soft_decision' if 'soft_decision' in signals.columns else 'decision'
    
    with col1:
        st.metric("📊 Total Signals", len(signals))
    with col2:
        actionable = signals['actionable'].sum() if 'actionable' in signals.columns else 0
        st.metric("✅ Actionable", int(actionable))
    with col3:
        buys = len(signals[signals[action_col].str.contains('BUY|LONG', case=False, na=False)])
        st.metric("🟢 Buy Signals", buys)
    with col4:
        sells = len(signals[signals[action_col].str.contains('SELL|SHORT', case=False, na=False)])
        st.metric("🔴 Sell Signals", sells)
    with col5:
        st.metric("📈 Avg Confidence", f"{signals['confidence'].mean()*100:.1f}%")
    with col6:
        if 'exp_30d_return' in signals.columns:
            st.metric("💰 Avg Exp. Return", f"{signals['exp_30d_return'].mean()*100:+.2f}%")
    
    st.markdown("---")
    
    # Top picks preview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Top Buy Signal")
        buys = signals[signals[action_col].str.contains('BUY|LONG', case=False, na=False)]
        if len(buys) > 0:
            if 'signal_strength' in buys.columns:
                top_buy = buys.sort_values('signal_strength', ascending=False).iloc[0]
            else:
                top_buy = buys.sort_values('confidence', ascending=False).iloc[0]
            
            st.markdown(f"""
            <div class="best-pick-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="ticker-large">{top_buy['ticker']}</span>
                    <span class="signal-badge buy">BUY</span>
                </div>
                <div class="price-large">${top_buy.get('price', 0):.2f}</div>
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-label">Expected Return</div>
                        <div class="metric-value positive">{top_buy.get('exp_30d_return', 0)*100:+.2f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{top_buy.get('confidence', 0)*100:.0f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Signal Strength</div>
                        <div class="metric-value">{top_buy.get('signal_strength', 0):.0f}/100</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">ADX</div>
                        <div class="metric-value">{top_buy.get('adx14', 0):.1f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No buy signals available")
    
    with col2:
        st.markdown("### 🔴 Top Sell Signal")
        sells = signals[signals[action_col].str.contains('SELL|SHORT', case=False, na=False)]
        if len(sells) > 0:
            if 'signal_strength' in sells.columns:
                top_sell = sells.sort_values('signal_strength', ascending=False).iloc[0]
            else:
                top_sell = sells.sort_values('confidence', ascending=False).iloc[0]
            
            st.markdown(f"""
            <div class="best-pick-card sell">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="ticker-large">{top_sell['ticker']}</span>
                    <span class="signal-badge sell">SELL</span>
                </div>
                <div class="price-large">${top_sell.get('price', 0):.2f}</div>
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-label">Expected Return</div>
                        <div class="metric-value negative">{top_sell.get('exp_30d_return', 0)*100:+.2f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{top_sell.get('confidence', 0)*100:.0f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Signal Strength</div>
                        <div class="metric-value">{top_sell.get('signal_strength', 0):.0f}/100</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">ADX</div>
                        <div class="metric-value">{top_sell.get('adx14', 0):.1f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No sell signals available")
    
    # Signal distribution chart
    st.markdown("### 📊 Signal Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        action_counts = signals[action_col].value_counts()
        colors = ['#10b981' if 'BUY' in str(l) or l == 'LONG' else '#ef4444' if 'SELL' in str(l) or l == 'SHORT' else '#64748b' for l in action_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=action_counts.index,
            values=action_counts.values,
            hole=0.6,
            marker=dict(colors=colors, line=dict(color='#1a1f2e', width=2)),
            textinfo='percent+label'
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'confidence' in signals.columns:
            fig = px.histogram(signals, x='confidence', nbins=20, 
                              color_discrete_sequence=['#3b82f6'])
            fig.update_layout(
                title="Confidence Distribution",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc'),
                xaxis=dict(gridcolor='#2d3748'),
                yaxis=dict(gridcolor='#2d3748'),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

def show_best_picks():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🏆 Best Predictions</h1>
        <p class="main-subtitle">Top-ranked signals based on confidence, strength, and expected returns</p>
    </div>
    """, unsafe_allow_html=True)
    
    signals = load_signals()
    if signals is None:
        st.warning("No signals available.")
        return
    
    action_col = 'action' if 'action' in signals.columns else 'soft_decision' if 'soft_decision' in signals.columns else 'decision'
    
    # Calculate composite score
    if 'signal_strength' in signals.columns and 'confidence' in signals.columns:
        signals['composite_score'] = signals['signal_strength'] * signals['confidence']
    else:
        signals['composite_score'] = signals['confidence']
    
    # Top Buys
    st.markdown("## 🟢 TOP BUY RECOMMENDATIONS")
    st.markdown("""
    <div class="explanation-box">
        <div class="explanation-title">💡 What makes a strong BUY signal?</div>
        <div class="explanation-text">
            These stocks show the highest probability of significant upward movement (5%+ in 30 days).
            They combine high model confidence, strong technical indicators (high ADX, positive momentum),
            and favorable prediction intervals.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    buys = signals[signals[action_col].str.contains('BUY|LONG', case=False, na=False)].sort_values('composite_score', ascending=False)
    
    for i, (_, row) in enumerate(buys.head(5).iterrows(), 1):
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="font-size: 2rem; font-weight: 800; color: #10b981;">#{i}</div>
                <div class="ticker-large">{row['ticker']}</div>
                <div class="price-large">${row.get('price', 0):.2f}</div>
                <div class="signal-badge buy" style="margin-top: 0.5rem;">STRONG BUY</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**📈 Key Metrics**")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Exp. Return", f"{row.get('exp_30d_return', 0)*100:+.2f}%")
            mcol2.metric("Confidence", f"{row.get('confidence', 0)*100:.0f}%")
            mcol3.metric("Strength", f"{row.get('signal_strength', 0):.0f}")
            mcol4.metric("ADX", f"{row.get('adx14', 0):.1f}")
            
            st.markdown("**📊 Technical Analysis**")
            tcol1, tcol2, tcol3 = st.columns(3)
            tcol1.metric("Momentum 60d", f"{row.get('mom_60', 0)*100:+.1f}%")
            if 'prob_5pct_30d' in row:
                tcol2.metric("P(5%+ Move)", f"{row['prob_5pct_30d']*100:.0f}%")
            if 'atr_pct' in row:
                tcol3.metric("Volatility", f"{row['atr_pct']*100:.1f}%")
        
        with col3:
            st.markdown("**🎯 Prediction Interval**")
            if 'pi_low' in row and 'pi_high' in row:
                st.markdown(f"""
                - **Low**: {row['pi_low']*100:+.2f}%
                - **Expected**: {row.get('exp_30d_return', 0)*100:+.2f}%
                - **High**: {row['pi_high']*100:+.2f}%
                """)
            
            if 'stop' in row and 'target1' in row:
                st.markdown("**💰 Risk Management**")
                st.markdown(f"""
                - **Stop Loss**: ${row['stop']:.2f}
                - **Target**: ${row['target1']:.2f}
                """)
        
        st.markdown("---")
    
    # Top Sells
    st.markdown("## 🔴 TOP SELL RECOMMENDATIONS")
    st.markdown("""
    <div class="explanation-box" style="border-color: rgba(239, 68, 68, 0.3); background: rgba(239, 68, 68, 0.1);">
        <div class="explanation-title" style="color: #ef4444;">💡 What makes a strong SELL signal?</div>
        <div class="explanation-text">
            These stocks show the highest probability of significant downward movement.
            Consider reducing positions or taking profits if you hold these stocks.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    sells = signals[signals[action_col].str.contains('SELL|SHORT', case=False, na=False)].sort_values('composite_score', ascending=False)
    
    for i, (_, row) in enumerate(sells.head(5).iterrows(), 1):
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.1); border-radius: 12px; padding: 1rem; text-align: center;">
                <div style="font-size: 2rem; font-weight: 800; color: #ef4444;">#{i}</div>
                <div class="ticker-large">{row['ticker']}</div>
                <div class="price-large">${row.get('price', 0):.2f}</div>
                <div class="signal-badge sell" style="margin-top: 0.5rem;">STRONG SELL</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**📉 Key Metrics**")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Exp. Return", f"{row.get('exp_30d_return', 0)*100:+.2f}%")
            mcol2.metric("Confidence", f"{row.get('confidence', 0)*100:.0f}%")
            mcol3.metric("Strength", f"{row.get('signal_strength', 0):.0f}")
            mcol4.metric("ADX", f"{row.get('adx14', 0):.1f}")
        
        with col3:
            if 'pi_low' in row and 'pi_high' in row:
                st.markdown("**🎯 Prediction Interval**")
                st.markdown(f"""
                - **Low**: {row['pi_low']*100:+.2f}%
                - **Expected**: {row.get('exp_30d_return', 0)*100:+.2f}%
                - **High**: {row['pi_high']*100:+.2f}%
                """)
        
        st.markdown("---")

def show_all_signals():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">📊 All Trading Signals</h1>
        <p class="main-subtitle">Complete list of all analyzed stocks with detailed metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    signals = load_signals()
    if signals is None:
        st.warning("No signals available.")
        return
    
    # Filters
    st.markdown("### 🔍 Filter Signals")
    
    col1, col2, col3, col4 = st.columns(4)
    
    action_col = 'action' if 'action' in signals.columns else 'soft_decision' if 'soft_decision' in signals.columns else 'decision'
    
    with col1:
        actions = st.multiselect("Signal Type", signals[action_col].unique().tolist(), default=signals[action_col].unique().tolist())
    with col2:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
    with col3:
        if 'signal_strength' in signals.columns:
            min_strength = st.slider("Min Strength", 0, 100, 0)
        else:
            min_strength = 0
    with col4:
        sort_by = st.selectbox("Sort By", ['signal_strength', 'confidence', 'exp_30d_return', 'ticker'])
    
    # Apply filters
    filtered = signals[signals[action_col].isin(actions)]
    filtered = filtered[filtered['confidence'] >= min_conf]
    if 'signal_strength' in filtered.columns:
        filtered = filtered[filtered['signal_strength'] >= min_strength]
    
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=(sort_by == 'ticker'))
    
    st.markdown(f"### Showing {len(filtered)} signals")
    
    # Display as cards
    for _, row in filtered.iterrows():
        signal_class = get_signal_class(row[action_col])
        exp_ret = row.get('exp_30d_return', 0)
        ret_class = 'positive' if exp_ret > 0 else 'negative'
        
        st.markdown(f"""
        <div class="all-signals-card" style="border-left: 4px solid {'#10b981' if signal_class == 'buy' else '#ef4444' if signal_class == 'sell' else '#64748b'};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-family: 'JetBrains Mono'; font-size: 1.3rem; font-weight: 700; color: #f8fafc;">{row['ticker']}</span>
                    <span style="color: #94a3b8; margin-left: 1rem;">${row.get('price', 0):.2f}</span>
                </div>
                <span class="signal-badge {signal_class}">{row[action_col]}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 1rem; margin-top: 1rem;">
                <div class="metric-item">
                    <div class="metric-label">Exp. Return</div>
                    <div class="metric-value {ret_class}">{exp_ret*100:+.2f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{row.get('confidence', 0)*100:.0f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Strength</div>
                    <div class="metric-value">{row.get('signal_strength', 0):.0f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">ADX</div>
                    <div class="metric-value">{row.get('adx14', 0):.1f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Momentum</div>
                    <div class="metric-value">{row.get('mom_60', 0)*100:+.1f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Gates</div>
                    <div class="metric-value">{'✅' if row.get('gates_pass', False) else '❌'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Download
    st.download_button("📥 Download All Signals (CSV)", filtered.to_csv(index=False), "long_term_signals.csv", "text/csv")

def show_portfolio():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">💼 Optimized Portfolio</h1>
        <p class="main-subtitle">Mean-variance optimized portfolio allocation based on signals</p>
    </div>
    """, unsafe_allow_html=True)
    
    portfolio = load_portfolio()
    if portfolio is None or 'weight' not in portfolio.columns:
        st.warning("No portfolio data available. Run the pipeline first.")
        return
    
    positions = portfolio[portfolio['weight'] > 0].sort_values('weight', ascending=False)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Positions", len(positions))
    with col2:
        st.metric("Total Allocation", f"{positions['weight'].sum():.1%}")
    with col3:
        st.metric("Max Position", f"{positions['weight'].max():.1%}")
    with col4:
        if 'exp_30d_return' in positions.columns:
            exp_ret = (positions['weight'] * positions['exp_30d_return']).sum()
            st.metric("Portfolio Exp. Return", f"{exp_ret*100:+.2f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(positions, values='weight', names='ticker', hole=0.5,
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(
            title="Portfolio Allocation",
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(positions.sort_values('weight'), x='weight', y='ticker', orientation='h',
                    color='weight', color_continuous_scale=['#8b5cf6', '#3b82f6', '#10b981'])
        fig.update_layout(
            title="Position Weights",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc'),
            xaxis=dict(gridcolor='#2d3748'),
            yaxis=dict(gridcolor='#2d3748'),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Position Details")
    st.dataframe(positions, use_container_width=True)

def show_model_details():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🔬 Model Details</h1>
        <p class="main-subtitle">Understanding the quantitative models and their predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    uncertainty = load_uncertainty()
    
    # Model architecture explanation
    st.markdown("### 🧠 Model Architecture")
    
    st.markdown("""
    <div class="explanation-box">
        <div class="explanation-title">Multi-Model Ensemble</div>
        <div class="explanation-text">
            Our prediction system uses a stacked ensemble of 4 machine learning models:
            <ul>
                <li><b>LightGBM</b> - Gradient boosting for fast, accurate predictions</li>
                <li><b>XGBoost</b> - Extreme gradient boosting for complex patterns</li>
                <li><b>Random Forest</b> - Ensemble of decision trees for robustness</li>
                <li><b>CatBoost</b> - Handles categorical features effectively</li>
            </ul>
            A meta-learner (Logistic Regression) combines their predictions for the final signal.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if uncertainty:
        st.markdown("### 📊 Uncertainty Calibration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Confidence Threshold (τ)", f"{uncertainty.get('selective_threshold', 0):.3f}")
            st.caption("Minimum confidence to issue signal")
        
        with col2:
            coverage = uncertainty.get('selective_meta', {}).get('coverage', 0)
            st.metric("Signal Coverage", f"{coverage*100:.1f}%")
            st.caption("% of stocks with actionable signals")
        
        with col3:
            st.metric("Prediction Interval (q̂)", f"{uncertainty.get('q_hat', 0):.4f}")
            st.caption("Conformal prediction width")
        
        with col4:
            st.metric("Target Error Rate (ε)", f"{uncertainty.get('epsilon', 0)*100:.0f}%")
            st.caption("Maximum allowed error rate")
        
        # Key metric explanations
        st.markdown("### 📖 Understanding the Metrics")
        
        st.markdown("""
        | Metric | Description | Interpretation |
        |--------|-------------|----------------|
        | **Confidence** | Model's certainty in the prediction | Higher = more reliable signal |
        | **Signal Strength** | Combined score from all models | 0-100 scale, higher = stronger |
        | **Expected Return** | Predicted price change over 30 days | Positive = buy, Negative = sell |
        | **ADX** | Average Directional Index | >25 = strong trend |
        | **Momentum 60d** | 60-day price momentum | Positive = uptrend |
        | **Prediction Interval** | Confidence range for returns | Narrow = more precise |
        | **Gates Pass** | Technical requirements met | ✅ = all conditions satisfied |
        """)

def show_pipeline():
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">⚙️ Run Pipeline</h1>
        <p class="main-subtitle">Train models and generate fresh predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Pipeline Steps")
        
        if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary"):
            with st.spinner("Running full pipeline... This may take several minutes."):
                steps = [
                    ("Training Models", ["-m", "src.modelling.train"]),
                    ("Generating Signals", ["-m", "src.live.run_signals"]),
                    ("Creating Portfolio", ["-m", "src.live.portfolio_from_signals"]),
                ]
                log = []
                for name, cmd in steps:
                    log.append(f"⏳ {name}...")
                    success, output = run_pipeline(cmd)
                    if success:
                        log.append(f"✅ {name} completed")
                    else:
                        log.append(f"❌ {name} failed")
                        break
                st.session_state['pipeline_log'] = "\n".join(log)
                st.rerun()
        
        st.markdown("---")
        
        if st.button("1️⃣ Train Models Only", use_container_width=True):
            with st.spinner("Training models..."):
                success, output = run_pipeline(["-m", "src.modelling.train"])
                st.session_state['pipeline_log'] = output
                st.rerun()
        
        if st.button("2️⃣ Generate Signals Only", use_container_width=True):
            with st.spinner("Generating signals..."):
                success, output = run_pipeline(["-m", "src.live.run_signals"])
                st.session_state['pipeline_log'] = output
                st.rerun()
    
    with col2:
        st.markdown("### Execution Log")
        if 'pipeline_log' in st.session_state:
            st.code(st.session_state['pipeline_log'])
        else:
            st.info("Run a pipeline step to see output here")

if __name__ == "__main__":
    main()
