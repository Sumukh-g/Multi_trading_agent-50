"""
🚀 QuantEdge Pro - Advanced Quantitative Trading Platform
A world-class trading dashboard with professional UI/UX

Features:
- Real-time signal analysis
- Multi-agent short-term trading system
- Advanced analytics and filtering
- Portfolio optimization

Run with: streamlit run dashboard.py
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
from datetime import datetime, timedelta
import json
import time

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="QuantEdge Pro | Trading Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PROFESSIONAL CSS STYLING
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a1f2e;
        --bg-hover: #252d3d;
        --accent-cyan: #00d4ff;
        --accent-purple: #8b5cf6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
        --accent-blue: #3b82f6;
        --accent-pink: #ec4899;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --border-color: #2d3748;
        --gradient-1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-2: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 100%);
        --gradient-3: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        --gradient-4: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
    }
    
    .stApp { background: var(--bg-primary); }
    .main .block-container { padding: 1rem 2rem 2rem 2rem; max-width: 100%; }
    #MainMenu, footer, header { visibility: hidden; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        border-right: 1px solid var(--border-color);
    }
    
    .platform-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .platform-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: var(--gradient-2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .platform-subtitle {
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        color: var(--accent-green);
        font-weight: 600;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-green);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: var(--accent-cyan);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.15);
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .metric-value.positive { color: var(--accent-green); }
    .metric-value.negative { color: var(--accent-red); }
    .metric-value.cyan { color: var(--accent-cyan); }
    .metric-value.purple { color: var(--accent-purple); }
    
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--border-color);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .signal-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .signal-card:hover {
        border-color: var(--accent-purple);
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.1);
    }
    
    .signal-card.buy { border-left: 4px solid var(--accent-green); }
    .signal-card.sell { border-left: 4px solid var(--accent-red); }
    .signal-card.strong-buy { border-left: 4px solid #22c55e; background: rgba(16, 185, 129, 0.05); }
    .signal-card.strong-sell { border-left: 4px solid #dc2626; background: rgba(239, 68, 68, 0.05); }
    
    .signal-ticker {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-buy { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-sell { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-hold { background: rgba(148, 163, 184, 0.15); color: var(--text-secondary); border: 1px solid rgba(148, 163, 184, 0.3); }
    .badge-strong-buy { background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-strong-sell { background: rgba(220, 38, 38, 0.2); color: #dc2626; border: 1px solid rgba(220, 38, 38, 0.4); }
    
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.25);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--text-secondary);
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-2);
        color: white;
    }
    
    .glass-panel {
        background: rgba(26, 31, 46, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
    }
    
    .agent-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .agent-name {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--accent-cyan);
        font-size: 0.9rem;
    }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-purple); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_files_exist():
    return {
        'models': os.path.exists('models/lgb_clf.pkl'),
        'signals': os.path.exists('data/signals_latest.csv'),
        'portfolio': os.path.exists('data/portfolio_latest.csv'),
        'summary': os.path.exists('data/signals_summary.json'),
        'short_term': os.path.exists('data/short_term_signals.json'),
        'data': len([f for f in os.listdir('raw') if f.endswith('.parquet')]) if os.path.exists('raw') else 0
    }

def load_signals():
    if os.path.exists('data/signals_latest.csv'):
        return pd.read_csv('data/signals_latest.csv')
    return None

def load_portfolio():
    if os.path.exists('data/portfolio_latest.csv'):
        return pd.read_csv('data/portfolio_latest.csv')
    return None

def load_summary():
    if os.path.exists('data/signals_summary.json'):
        with open('data/signals_summary.json', 'r') as f:
            return json.load(f)
    return None

def load_short_term_signals():
    if os.path.exists('data/short_term_signals.json'):
        with open('data/short_term_signals.json', 'r') as f:
            return json.load(f)
    return None

def load_uncertainty():
    if os.path.exists('models/uncertainty.json'):
        with open('models/uncertainty.json', 'r') as f:
            return json.load(f)
    return None

def create_gauge_chart(value, title, max_val=100, color='#8b5cf6'):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': '#94a3b8'}},
        number={'font': {'size': 28, 'color': '#f8fafc'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#64748b'},
            'bar': {'color': color},
            'bgcolor': '#1a1f2e',
            'borderwidth': 0,
            'steps': [
                {'range': [0, max_val*0.33], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [max_val*0.33, max_val*0.66], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [max_val*0.66, max_val], 'color': 'rgba(16, 185, 129, 0.2)'}
            ],
            'threshold': {'line': {'color': '#00d4ff', 'width': 3}, 'thickness': 0.8, 'value': value}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#f8fafc'}, height=200, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_donut_chart(labels, values, colors, title=""):
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.65,
        marker=dict(colors=colors, line=dict(color='#1a1f2e', width=2)),
        textinfo='percent', textfont=dict(size=12, color='#f8fafc')
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#f8fafc'), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(color='#94a3b8', size=11)),
        height=300, margin=dict(l=20, r=20, t=50, b=50)
    )
    return fig

def create_heatmap(data, x_labels, y_labels, title=""):
    fig = go.Figure(data=go.Heatmap(
        z=data, x=x_labels, y=y_labels,
        colorscale=[[0, '#ef4444'], [0.5, '#1a1f2e'], [1, '#10b981']],
        showscale=True
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#f8fafc')),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'), height=400, margin=dict(l=60, r=20, t=50, b=60)
    )
    return fig

def run_pipeline_step(command):
    python_exe = sys.executable
    try:
        result = subprocess.run([python_exe, "-u"] + command, capture_output=True, text=True, timeout=600)
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    files = check_files_exist()
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚡</div>
            <div style="font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 800; 
                        background: linear-gradient(135deg, #00d4ff 0%, #8b5cf6 100%);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                QuantEdge Pro
            </div>
            <div style="color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;">
                Quantitative Trading Platform
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🚀 Pipeline", "📈 Signals", "💼 Portfolio", 
             "📉 Analytics", "⏱️ Short-Term", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🔋 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"{'🟢' if files['data'] > 0 else '🔴'} Data: {files['data']}")
            st.markdown(f"{'🟢' if files['models'] else '🔴'} Models")
        with col2:
            st.markdown(f"{'🟢' if files['signals'] else '🔴'} Signals")
            st.markdown(f"{'🟢' if files['short_term'] else '🔴'} Short-Term")
        
        st.markdown("---")
        st.markdown(f"<div style='color: #64748b; font-size: 0.75rem; text-align: center;'>"
                   f"Updated: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
    
    if page == "📊 Dashboard":
        show_dashboard(files)
    elif page == "🚀 Pipeline":
        show_pipeline()
    elif page == "📈 Signals":
        show_signals_enhanced()
    elif page == "💼 Portfolio":
        show_portfolio()
    elif page == "📉 Analytics":
        show_analytics_enhanced()
    elif page == "⏱️ Short-Term":
        show_short_term()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard(files):
    st.markdown("""
    <div class="platform-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="platform-title">Trading Dashboard</h1>
                <p class="platform-subtitle">Real-time quantitative analysis and signal generation</p>
            </div>
            <div class="live-indicator"><div class="live-dot"></div>LIVE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    signals = load_signals()
    summary = load_summary()
    portfolio = load_portfolio()
    
    # Quick metrics
    cols = st.columns(5)
    metrics = [
        ("Total Signals", len(signals) if signals is not None else 0, "cyan"),
        ("Actionable", summary.get('actionable', 0) if summary else 0, "purple"),
        ("Buy Signals", summary.get('buy_signals', 0) if summary else 0, "positive"),
        ("Sell Signals", summary.get('sell_signals', 0) if summary else 0, "negative"),
        ("Avg Confidence", f"{summary.get('avg_confidence', 0)*100:.1f}%" if summary else "N/A", "")
    ]
    
    for col, (label, value, color_class) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {color_class}">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">🏆 Top Signals</div>', unsafe_allow_html=True)
        if signals is not None and len(signals) > 0:
            for _, row in signals.head(8).iterrows():
                action = row.get('action', row.get('soft_decision', 'HOLD'))
                is_buy = 'BUY' in str(action) or action == 'LONG'
                is_sell = 'SELL' in str(action) or action == 'SHORT'
                card_class = 'buy' if is_buy else 'sell' if is_sell else ''
                exp_ret = row.get('exp_30d_return', 0) * 100
                conf = row.get('confidence', 0) * 100
                strength = row.get('signal_strength', 0)
                badge_class = 'badge-buy' if is_buy else 'badge-sell' if is_sell else 'badge-hold'
                badge_text = 'LONG' if is_buy else 'SHORT' if is_sell else 'HOLD'
                
                st.markdown(f"""
                <div class="signal-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span class="signal-ticker">{row['ticker']}</span>
                            <span style="color: #94a3b8; margin-left: 1rem;">${row.get('price', 0):.2f}</span>
                        </div>
                        <span class="badge {badge_class}">{badge_text}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;">
                        <div><div style="color: #64748b; font-size: 0.75rem;">Expected Return</div>
                            <div style="color: {'#10b981' if exp_ret > 0 else '#ef4444'}; font-weight: 600;">{exp_ret:+.2f}%</div></div>
                        <div><div style="color: #64748b; font-size: 0.75rem;">Confidence</div>
                            <div style="color: #f8fafc; font-weight: 600;">{conf:.1f}%</div></div>
                        <div><div style="color: #64748b; font-size: 0.75rem;">Signal Strength</div>
                            <div style="color: #8b5cf6; font-weight: 600;">{strength:.0f}/100</div></div>
                        <div><div style="color: #64748b; font-size: 0.75rem;">ADX</div>
                            <div style="color: #f8fafc; font-weight: 600;">{row.get('adx14', 0):.1f}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No signals available. Run the pipeline to generate signals.")
    
    with col2:
        st.markdown('<div class="section-header">📊 Signal Distribution</div>', unsafe_allow_html=True)
        if signals is not None:
            action_col = 'action' if 'action' in signals.columns else 'soft_decision' if 'soft_decision' in signals.columns else 'decision'
            action_counts = signals[action_col].value_counts()
            colors = ['#10b981' if 'BUY' in str(l) or l == 'LONG' else '#ef4444' if 'SELL' in str(l) or l == 'SHORT' else '#64748b' for l in action_counts.index]
            fig = create_donut_chart(action_counts.index.tolist(), action_counts.values.tolist(), colors, "Actions Breakdown")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="section-header">🎯 Avg Confidence</div>', unsafe_allow_html=True)
        if summary:
            fig = create_gauge_chart(summary.get('avg_confidence', 0) * 100, "Model Confidence", 100)
            st.plotly_chart(fig, use_container_width=True)
    
    if portfolio is not None and 'weight' in portfolio.columns:
        st.markdown('<div class="section-header">💼 Portfolio Allocation</div>', unsafe_allow_html=True)
        positions = portfolio[portfolio['weight'] > 0].sort_values('weight', ascending=True).tail(10)
        if len(positions) > 0:
            fig = px.bar(positions, x='weight', y='ticker', orientation='h', color='weight',
                        color_continuous_scale=['#8b5cf6', '#00d4ff'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f8fafc'), xaxis=dict(gridcolor='#2d3748'),
                            yaxis=dict(gridcolor='#2d3748'), height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_pipeline():
    st.markdown("""
    <div class="platform-header">
        <h1 class="platform-title">Pipeline Control</h1>
        <p class="platform-subtitle">Execute and monitor the quantitative analysis pipeline</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="section-header">⚡ Quick Actions</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary"):
            with st.spinner("Running full pipeline..."):
                steps = [
                    ("Train Models", ["-m", "src.modelling.train"]),
                    ("Generate Signals", ["-m", "src.live.run_signals"]),
                    ("Create Portfolio", ["-m", "src.live.portfolio_from_signals"]),
                ]
                log = []
                for step_name, cmd in steps:
                    log.append(f"\n{'='*50}\n⏳ {step_name}\n{'='*50}")
                    success, output = run_pipeline_step(cmd)
                    log.append(output)
                    if not success:
                        break
                st.session_state['pipeline_log'] = "\n".join(log)
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Individual Steps:**")
        
        if st.button("1️⃣ Train Models", use_container_width=True):
            with st.spinner("Training..."):
                success, output = run_pipeline_step(["-m", "src.modelling.train"])
                st.session_state['pipeline_log'] = output
                st.rerun()
        
        if st.button("2️⃣ Generate Signals", use_container_width=True):
            with st.spinner("Generating signals..."):
                success, output = run_pipeline_step(["-m", "src.live.run_signals"])
                st.session_state['pipeline_log'] = output
                st.rerun()
        
        if st.button("3️⃣ Create Portfolio", use_container_width=True):
            with st.spinner("Creating portfolio..."):
                success, output = run_pipeline_step(["-m", "src.live.portfolio_from_signals"])
                st.session_state['pipeline_log'] = output
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("⏱️ Run Short-Term Analysis", use_container_width=True):
            with st.spinner("Running short-term multi-agent analysis..."):
                success, output = run_pipeline_step(["-m", "src.agents.short_term.orchestrator"])
                st.session_state['pipeline_log'] = output
                st.rerun()
    
    with col2:
        st.markdown('<div class="section-header">📋 Execution Log</div>', unsafe_allow_html=True)
        if 'pipeline_log' in st.session_state:
            st.code(st.session_state['pipeline_log'], language='bash')
        else:
            st.markdown("""
            <div class="glass-panel" style="height: 400px; display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center; color: #64748b;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📋</div>
                    <div>Run a pipeline step to see output</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def show_signals_enhanced():
    st.markdown("""
    <div class="platform-header">
        <h1 class="platform-title">Signal Analysis</h1>
        <p class="platform-subtitle">Comprehensive trading signal breakdown with advanced filtering</p>
    </div>
    """, unsafe_allow_html=True)
    
    signals = load_signals()
    if signals is None:
        st.warning("No signals data available. Run the pipeline first.")
        return
    
    # Enhanced filters
    st.markdown("### 🔍 Advanced Filters")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        show_actionable = st.checkbox("Actionable Only", value=False)
    
    with col2:
        action_col = 'action' if 'action' in signals.columns else 'soft_decision' if 'soft_decision' in signals.columns else 'decision'
        actions = st.multiselect("Filter Actions", signals[action_col].unique().tolist(), default=signals[action_col].unique().tolist())
    
    with col3:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
    
    with col4:
        min_strength = st.slider("Min Signal Strength", 0, 100, 0, 5) if 'signal_strength' in signals.columns else 0
    
    with col5:
        sort_options = ["signal_strength", "confidence", "exp_30d_return", "ticker", "adx14", "mom_60"]
        sort_options = [s for s in sort_options if s in signals.columns]
        sort_by = st.selectbox("Sort By", sort_options)
    
    # Additional filters row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'exp_30d_return' in signals.columns:
            min_return = st.slider("Min Expected Return", -0.2, 0.2, -0.2, 0.01, format="%.0f%%")
        else:
            min_return = -1
    
    with col2:
        if 'adx14' in signals.columns:
            min_adx = st.slider("Min ADX (Trend Strength)", 0, 50, 0, 5)
        else:
            min_adx = 0
    
    with col3:
        if 'prob_5pct_30d' in signals.columns:
            min_prob = st.slider("Min Move Probability", 0.0, 1.0, 0.0, 0.05)
        else:
            min_prob = 0
    
    with col4:
        if 'rsi14' in signals.columns:
            rsi_range = st.slider("RSI Range", 0, 100, (0, 100))
        else:
            rsi_range = (0, 100)
    
    # Apply filters
    filtered = signals.copy()
    
    if show_actionable and 'actionable' in filtered.columns:
        filtered = filtered[filtered['actionable'] == True]
    
    filtered = filtered[filtered[action_col].isin(actions)]
    
    if 'confidence' in filtered.columns:
        filtered = filtered[filtered['confidence'] >= min_confidence]
    
    if 'signal_strength' in filtered.columns:
        filtered = filtered[filtered['signal_strength'] >= min_strength]
    
    if 'exp_30d_return' in filtered.columns:
        filtered = filtered[filtered['exp_30d_return'] >= min_return]
    
    if 'adx14' in filtered.columns:
        filtered = filtered[filtered['adx14'] >= min_adx]
    
    if 'prob_5pct_30d' in filtered.columns:
        filtered = filtered[filtered['prob_5pct_30d'] >= min_prob]
    
    if 'rsi14' in filtered.columns:
        filtered = filtered[(filtered['rsi14'] >= rsi_range[0]) & (filtered['rsi14'] <= rsi_range[1])]
    
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=(sort_by == 'ticker'))
    
    # Display metrics
    st.markdown("### 📊 Filtered Results")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Filtered Signals", len(filtered))
    with col2:
        st.metric("Avg Confidence", f"{filtered['confidence'].mean()*100:.1f}%" if 'confidence' in filtered.columns else "N/A")
    with col3:
        st.metric("Avg Strength", f"{filtered['signal_strength'].mean():.0f}" if 'signal_strength' in filtered.columns else "N/A")
    with col4:
        st.metric("Avg Exp. Return", f"{filtered['exp_30d_return'].mean()*100:+.2f}%" if 'exp_30d_return' in filtered.columns else "N/A")
    with col5:
        bullish = len(filtered[filtered[action_col].str.contains('BUY|LONG', case=False, na=False)])
        st.metric("Bull/Bear Ratio", f"{bullish}/{len(filtered)-bullish}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display table with more columns
    display_cols = ['ticker', 'price', action_col, 'exp_30d_return', 'confidence', 
                   'signal_strength', 'prob_5pct_30d', 'prob_2pct', 'prob_10pct',
                   'adx14', 'mom_60', 'rsi14', 'atr_pct', 'gates_pass']
    display_cols = [c for c in display_cols if c in filtered.columns]
    
    st.dataframe(filtered[display_cols], use_container_width=True, height=500)
    
    # Download
    csv = filtered.to_csv(index=False)
    st.download_button("📥 Download Filtered CSV", csv, "filtered_signals.csv", "text/csv", use_container_width=True)

def show_analytics_enhanced():
    st.markdown("""
    <div class="platform-header">
        <h1 class="platform-title">Advanced Analytics</h1>
        <p class="platform-subtitle">Deep dive into model performance, signal characteristics, and market insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    signals = load_signals()
    uncertainty = load_uncertainty()
    
    if signals is None:
        st.warning("No data available for analysis.")
        return
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Distributions", "🎯 Correlations", "📈 Performance", "🔬 Model Metrics", "🌡️ Market Heat"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'confidence' in signals.columns:
                fig = px.histogram(signals, x='confidence', nbins=30, title='Confidence Distribution',
                                  color_discrete_sequence=['#8b5cf6'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#f8fafc'), xaxis=dict(gridcolor='#2d3748'), yaxis=dict(gridcolor='#2d3748'))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'exp_30d_return' in signals.columns:
                fig = px.histogram(signals, x='exp_30d_return', nbins=30, title='Expected Return Distribution',
                                  color_discrete_sequence=['#00d4ff'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#f8fafc'), xaxis=dict(gridcolor='#2d3748'), yaxis=dict(gridcolor='#2d3748'))
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if 'signal_strength' in signals.columns:
                fig = px.histogram(signals, x='signal_strength', nbins=30, title='Signal Strength Distribution',
                                  color_discrete_sequence=['#10b981'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#f8fafc'), xaxis=dict(gridcolor='#2d3748'), yaxis=dict(gridcolor='#2d3748'))
                st.plotly_chart(fig, use_container_width=True)
        
        # Additional distributions
        col1, col2 = st.columns(2)
        
        with col1:
            if 'adx14' in signals.columns:
                fig = px.box(signals, y='adx14', title='ADX Distribution', color_discrete_sequence=['#f59e0b'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#f8fafc'), yaxis=dict(gridcolor='#2d3748'))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'atr_pct' in signals.columns:
                fig = px.box(signals, y='atr_pct', title='ATR% Distribution (Volatility)', color_discrete_sequence=['#ec4899'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#f8fafc'), yaxis=dict(gridcolor='#2d3748'))
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'confidence' in signals.columns and 'exp_30d_return' in signals.columns:
            action_col = 'soft_decision' if 'soft_decision' in signals.columns else 'decision'
            fig = px.scatter(signals, x='confidence', y='exp_30d_return', color=action_col,
                           size='signal_strength' if 'signal_strength' in signals.columns else None,
                           title='Confidence vs Expected Return',
                           color_discrete_map={'LONG': '#10b981', 'SHORT': '#ef4444', 'HOLD': '#64748b', 'UNSURE': '#64748b'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f8fafc'), xaxis=dict(gridcolor='#2d3748'), yaxis=dict(gridcolor='#2d3748'))
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        numeric_cols = ['confidence', 'exp_30d_return', 'signal_strength', 'adx14', 'mom_60', 'atr_pct', 'prob_5pct_30d']
        numeric_cols = [c for c in numeric_cols if c in signals.columns]
        
        if len(numeric_cols) > 1:
            corr_matrix = signals[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto='.2f', title='Feature Correlation Matrix',
                           color_continuous_scale=['#ef4444', '#1a1f2e', '#10b981'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f8fafc'))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 Signal Performance Analysis")
        
        # Signal breakdown by action
        action_col = 'soft_decision' if 'soft_decision' in signals.columns else 'decision'
        
        col1, col2 = st.columns(2)
        
        with col1:
            grouped = signals.groupby(action_col).agg({
                'confidence': 'mean',
                'exp_30d_return': 'mean',
                'signal_strength': 'mean' if 'signal_strength' in signals.columns else 'count'
            }).reset_index()
            
            fig = px.bar(grouped, x=action_col, y='exp_30d_return', color=action_col,
                        title='Average Expected Return by Signal Type',
                        color_discrete_map={'LONG': '#10b981', 'SHORT': '#ef4444', 'HOLD': '#64748b', 'UNSURE': '#64748b'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f8fafc'), xaxis=dict(gridcolor='#2d3748'), yaxis=dict(gridcolor='#2d3748'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(grouped, x=action_col, y='confidence', color=action_col,
                        title='Average Confidence by Signal Type',
                        color_discrete_map={'LONG': '#10b981', 'SHORT': '#ef4444', 'HOLD': '#64748b', 'UNSURE': '#64748b'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#f8fafc'), xaxis=dict(gridcolor='#2d3748'), yaxis=dict(gridcolor='#2d3748'))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        if uncertainty:
            st.markdown("### 🔬 Model Calibration Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Confidence Threshold (τ)", f"{uncertainty.get('selective_threshold', 0):.3f}")
            with col2:
                coverage = uncertainty.get('selective_meta', {}).get('coverage', 0)
                st.metric("Coverage", f"{coverage*100:.1f}%")
            with col3:
                st.metric("Prediction Interval (q̂)", f"{uncertainty.get('q_hat', 0):.4f}")
            with col4:
                st.metric("Target Error Rate (ε)", f"{uncertainty.get('epsilon', 0)*100:.0f}%")
            
            st.markdown("### Model Parameters")
            st.json(uncertainty)
    
    with tab5:
        st.markdown("### 🌡️ Market Heat Map")
        
        if 'ticker' in signals.columns and 'exp_30d_return' in signals.columns:
            # Create a treemap-style visualization
            fig = px.treemap(signals, path=['ticker'], values='signal_strength' if 'signal_strength' in signals.columns else 'confidence',
                            color='exp_30d_return', color_continuous_scale=['#ef4444', '#1a1f2e', '#10b981'],
                            title='Signal Strength by Ticker (colored by Expected Return)')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#f8fafc'))
            st.plotly_chart(fig, use_container_width=True)

def show_short_term():
    st.markdown("""
    <div class="platform-header">
        <h1 class="platform-title">⏱️ Short-Term Trading</h1>
        <p class="platform-subtitle">Multi-Agent System | 1-Week Predictions | Hourly Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Run analysis button
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Run Short-Term Analysis", use_container_width=True, type="primary"):
            with st.spinner("Running multi-agent analysis on hourly data..."):
                success, output = run_pipeline_step(["-m", "src.agents.short_term.orchestrator"])
                st.session_state['short_term_log'] = output
                st.rerun()
    
    with col2:
        st.info("⏱️ Horizon: 1 Week (168h)")
    
    # Load short-term signals
    short_term = load_short_term_signals()
    
    if short_term is None:
        st.warning("No short-term signals available. Run the analysis first.")
        if 'short_term_log' in st.session_state:
            st.code(st.session_state['short_term_log'], language='bash')
        return
    
    signals_df = pd.DataFrame(short_term)
    
    # Summary metrics
    st.markdown("### 📊 Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    bullish = len([s for s in short_term if 'BUY' in s.get('signal', '')])
    bearish = len([s for s in short_term if 'SELL' in s.get('signal', '')])
    
    with col1:
        st.metric("Total Analyzed", len(short_term))
    with col2:
        st.metric("Bullish Signals", bullish)
    with col3:
        st.metric("Bearish Signals", bearish)
    with col4:
        avg_conf = np.mean([s.get('confidence', 0) for s in short_term])
        st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
    with col5:
        avg_rr = np.mean([s.get('risk_reward_ratio', 0) for s in short_term])
        st.metric("Avg Risk/Reward", f"{avg_rr:.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Agent analysis section
    st.markdown("### 🤖 Multi-Agent Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top signals
        st.markdown("#### 🏆 Top Signals (Ranked by Strength × Confidence)")
        
        # Sort by strength * confidence
        sorted_signals = sorted(short_term, key=lambda x: x.get('strength', 0) * x.get('confidence', 0), reverse=True)
        
        for sig in sorted_signals[:6]:
            signal_type = sig.get('signal', 'NEUTRAL')
            is_buy = 'BUY' in signal_type
            is_sell = 'SELL' in signal_type
            is_strong = 'STRONG' in signal_type
            
            card_class = 'strong-buy' if is_strong and is_buy else 'strong-sell' if is_strong and is_sell else 'buy' if is_buy else 'sell' if is_sell else ''
            badge_class = 'badge-strong-buy' if is_strong and is_buy else 'badge-strong-sell' if is_strong and is_sell else 'badge-buy' if is_buy else 'badge-sell' if is_sell else 'badge-hold'
            
            st.markdown(f"""
            <div class="signal-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="signal-ticker">{sig.get('ticker', 'N/A')}</span>
                        <span style="color: #94a3b8; margin-left: 1rem;">${sig.get('current_price', 0):.2f}</span>
                    </div>
                    <span class="badge {badge_class}">{signal_type}</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-top: 1rem;">
                    <div><div style="color: #64748b; font-size: 0.7rem;">1W Return</div>
                        <div style="color: {'#10b981' if sig.get('pred_return_1w', 0) > 0 else '#ef4444'}; font-weight: 600; font-size: 0.9rem;">
                            {sig.get('pred_return_1w', 0)*100:+.2f}%</div></div>
                    <div><div style="color: #64748b; font-size: 0.7rem;">Confidence</div>
                        <div style="color: #f8fafc; font-weight: 600; font-size: 0.9rem;">{sig.get('confidence', 0)*100:.0f}%</div></div>
                    <div><div style="color: #64748b; font-size: 0.7rem;">Strength</div>
                        <div style="color: #8b5cf6; font-weight: 600; font-size: 0.9rem;">{sig.get('strength', 0):.0f}</div></div>
                    <div><div style="color: #64748b; font-size: 0.7rem;">Stop Loss</div>
                        <div style="color: #ef4444; font-weight: 600; font-size: 0.9rem;">${sig.get('stop_loss', 0):.2f}</div></div>
                    <div><div style="color: #64748b; font-size: 0.7rem;">Take Profit</div>
                        <div style="color: #10b981; font-weight: 600; font-size: 0.9rem;">${sig.get('take_profit_1', 0):.2f}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Agent breakdown
        st.markdown("#### 🤖 Agent System")
        
        agents = [
            ("📈 Trend Agent", "Trend detection using EMA, MACD, ADX, Supertrend", "#00d4ff"),
            ("🚀 Momentum Agent", "RSI, Stochastic, ROC, divergence detection", "#8b5cf6"),
            ("📊 Volatility Agent", "Bollinger, Keltner, squeeze detection", "#f59e0b"),
            ("📦 Volume Agent", "OBV, VWAP, CMF, accumulation/distribution", "#10b981"),
            ("🎯 Pattern Agent", "Candlestick patterns, S/R levels, breakouts", "#ec4899")
        ]
        
        for name, desc, color in agents:
            st.markdown(f"""
            <div class="agent-card">
                <div class="agent-name" style="color: {color};">{name}</div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 0.25rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed data table
    st.markdown("### 📋 All Signals")
    
    display_cols = ['ticker', 'current_price', 'signal', 'confidence', 'strength', 
                   'pred_return_1d', 'pred_return_3d', 'pred_return_1w',
                   'stop_loss', 'take_profit_1', 'risk_reward_ratio', 'position_size_pct']
    display_cols = [c for c in display_cols if c in signals_df.columns]
    
    st.dataframe(signals_df[display_cols], use_container_width=True, height=400)

def show_portfolio():
    st.markdown("""
    <div class="platform-header">
        <h1 class="platform-title">Portfolio Manager</h1>
        <p class="platform-subtitle">Optimized portfolio allocation and risk analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    portfolio = load_portfolio()
    
    if portfolio is None or 'weight' not in portfolio.columns:
        st.warning("No portfolio data available. Run the pipeline first.")
        return
    
    positions = portfolio[portfolio['weight'] > 0].copy()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Positions", len(positions))
    with col2:
        st.metric("Total Allocation", f"{positions['weight'].sum():.1%}")
    with col3:
        st.metric("Max Position", f"{positions['weight'].max():.1%}")
    with col4:
        if 'exp_30d_return' in positions.columns:
            expected = (positions['weight'] * positions['exp_30d_return']).sum()
            st.metric("Expected Return", f"{expected*100:+.2f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_donut_chart(positions['ticker'].tolist(), positions['weight'].tolist(),
                                px.colors.qualitative.Set3[:len(positions)], "Portfolio Allocation")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        sorted_pos = positions.sort_values('weight', ascending=True)
        fig = px.bar(sorted_pos, x='weight', y='ticker', orientation='h', color='weight',
                    color_continuous_scale=['#8b5cf6', '#00d4ff'], title="Position Weights")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#f8fafc'), xaxis=dict(gridcolor='#2d3748'), yaxis=dict(gridcolor='#2d3748'), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-header">📋 Position Details</div>', unsafe_allow_html=True)
    
    display_cols = ['ticker', 'weight', 'price', 'exp_30d_return', 'confidence', 'stop', 'target1']
    display_cols = [c for c in display_cols if c in positions.columns]
    
    st.dataframe(positions[display_cols], use_container_width=True)

def show_settings():
    st.markdown("""
    <div class="platform-header">
        <h1 class="platform-title">Settings</h1>
        <p class="platform-subtitle">Platform configuration and system information</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📁 Configuration", "ℹ️ System Info", "🤖 Agent Config"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Universe Configuration")
            if os.path.exists("config/universe.yaml"):
                with open("config/universe.yaml", "r") as f:
                    st.code(f.read(), language="yaml")
        with col2:
            st.markdown("#### Trading Rules")
            if os.path.exists("docs/rulebook.yaml"):
                with open("docs/rulebook.yaml", "r") as f:
                    st.code(f.read(), language="yaml")
    
    with tab2:
        st.info(f"**Python:** {sys.version}")
        st.info(f"**Working Directory:** {os.getcwd()}")
        st.info(f"**Streamlit Version:** {st.__version__}")
    
    with tab3:
        st.markdown("#### Short-Term Multi-Agent System Configuration")
        st.markdown("""
        The short-term trading system uses 5 specialized agents:
        
        | Agent | Weight | Focus |
        |-------|--------|-------|
        | **Trend Agent** | 1.2 | EMA crossovers, MACD, ADX, Supertrend |
        | **Momentum Agent** | 1.0 | RSI, Stochastic, ROC, divergences |
        | **Volatility Agent** | 1.0 | Bollinger Bands, Keltner, squeeze |
        | **Volume Agent** | 0.9 | OBV, VWAP, CMF, climax detection |
        | **Pattern Agent** | 1.1 | Candlesticks, S/R, breakouts |
        
        Signals are combined using confidence-weighted voting with dynamic weight adjustment.
        """)

if __name__ == "__main__":
    main()
