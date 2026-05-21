"""
Advanced Multi-Agent Orchestrator

Coordinates the advanced weighted agent system with:
- 18 research-backed indicators
- 5 agent categories with dynamic weights
- Regime detection and adaptation
- Sector-specific adjustments
- Comprehensive trade explanations
- Meta-labeling for signal filtering
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import json
import os

from .base_agent import AgentConfig, SignalType
from .advanced_weighted_agent import (
    AdvancedWeightedAgent, AdvancedSignal, 
    MarketRegime, SectorType, AgentContribution, IndicatorSignal
)


@dataclass
class AdvancedPrediction:
    """Complete advanced prediction for a ticker."""
    ticker: str
    current_price: float
    
    # Core signal
    signal_type: SignalType
    signal_name: str  # e.g., "STRONG_BUY"
    confidence: float
    strength: float
    expected_return: float
    
    # Multi-horizon predictions
    pred_return_1d: float
    pred_return_3d: float
    pred_return_1w: float
    
    # Market context
    regime: str
    regime_explanation: str
    sector: str
    sector_impact: str
    
    # Agent breakdown
    agent_scores: Dict[str, Dict]  # agent -> {score, signal, confidence, explanation}
    
    # Indicator breakdown
    indicator_details: List[Dict]  # List of indicator signals
    bullish_indicators: int
    bearish_indicators: int
    
    # Trade thesis and explanation
    trade_thesis: str
    why_this_trade: str
    key_factors: List[str]
    risk_factors: List[str]
    
    # Risk management
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size_pct: float
    risk_reward_ratio: float
    
    # Weights used
    effective_agent_weights: Dict[str, float]
    regime_multipliers: Dict[str, float]
    sector_multipliers: Dict[str, float]
    
    timestamp: datetime


class AdvancedOrchestrator:
    """
    Advanced orchestrator for the multi-agent trading system.
    
    Features:
    - 18 indicators across 5 agent categories
    - Research-backed weighting (sum to 1.0)
    - Dynamic regime-based weight adjustment
    - Sector-specific multipliers
    - Comprehensive trade explanations
    - Multi-horizon predictions (1D, 3D, 1W)
    """
    
    # Regime explanations
    REGIME_EXPLANATIONS = {
        MarketRegime.STRONG_TREND: "Market is in a strong trending phase. Trend-following indicators are given higher weight (1.4x). Mean reversion strategies are de-emphasized.",
        MarketRegime.MEAN_REVERTING: "Market is range-bound with oscillating prices. Volatility and momentum oscillators are prioritized (1.3x, 1.2x). Trend indicators are de-weighted (0.6x).",
        MarketRegime.VOLATILITY_BREAKOUT: "Volatility squeeze has released - expect a directional move. Volatility and pattern agents weighted higher (1.4x, 1.3x). Volume confirmation is critical.",
        MarketRegime.ILLIQUID_NOISY: "Low liquidity conditions detected. Volume signals are unreliable (0.6x). Pattern-based signals emphasized (1.2x). Use wider stops.",
        MarketRegime.NORMAL: "Normal market conditions. Standard indicator weights applied across all agents."
    }
    
    # Sector impact descriptions
    SECTOR_IMPACTS = {
        SectorType.TECH_GROWTH: "Tech/Growth sector favors momentum and trend signals (Trend 1.25x, Momentum 1.2x). Watch for high volatility events.",
        SectorType.DEFENSIVE: "Defensive sector exhibits mean-reversion characteristics (Volatility 1.3x, Momentum 1.2x). Slower-moving trends.",
        SectorType.COMMODITY_ENERGY: "Energy/Commodity sector shows cyclical patterns (Trend 1.3x, Volume 1.2x). Macro factors dominate.",
        SectorType.SMALL_CAP_BIOTECH: "Small-cap/Biotech requires volume confirmation (Volume 1.4x, Pattern 1.3x). Event-driven moves common.",
        SectorType.GENERAL: "General sector classification. Standard weights applied."
    }
    
    def __init__(self, horizon_hours: int = 168):
        """
        Initialize the advanced orchestrator.
        
        Args:
            horizon_hours: Primary prediction horizon (default 168 = 1 week)
        """
        self.horizon_hours = horizon_hours
        
        # Initialize the advanced weighted agent
        self.agent = AdvancedWeightedAgent(
            AgentConfig(name="AdvancedWeightedAgent", weight=1.0)
        )
        
        # Data cache
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_expiry = timedelta(hours=1)
        self.cache_timestamps: Dict[str, datetime] = {}
    
    def fetch_hourly_data(self, ticker: str, lookback_days: int = 60) -> Optional[pd.DataFrame]:
        """Fetch hourly OHLCV data for a ticker."""
        now = datetime.now()
        if ticker in self.data_cache:
            if now - self.cache_timestamps.get(ticker, datetime.min) < self.cache_expiry:
                return self.data_cache[ticker]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                df = yf.download(
                    ticker,
                    period=f"{min(lookback_days, 59)}d",
                    interval="1h",
                    auto_adjust=True,
                    progress=False
                )
                
                if df.empty or len(df) < 100:
                    return None
                
                # Clean column names
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                df.columns = [str(c).title() for c in df.columns]
                
                # Verify columns
                required = ['Open', 'High', 'Low', 'Close', 'Volume']
                for c in required:
                    if c not in df.columns:
                        return None
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                
                df = df.dropna(subset=['Close'])
                if len(df) < 100:
                    return None
                
                self.data_cache[ticker] = df
                self.cache_timestamps[ticker] = now
                return df
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def analyze_ticker(self, ticker: str) -> Optional[AdvancedPrediction]:
        """Run advanced analysis for a single ticker."""
        df = self.fetch_hourly_data(ticker)
        if df is None:
            return None
        
        try:
            # Get current price
            close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
            current_price = float(close.iloc[-1])
            
            # Run advanced analysis
            adv_signal = self.agent.get_advanced_signal(df, ticker)
            
            # Multi-horizon predictions
            pred_1d, _ = self._estimate_horizon_return(adv_signal, 24)
            pred_3d, _ = self._estimate_horizon_return(adv_signal, 72)
            pred_1w, _ = self._estimate_horizon_return(adv_signal, 168)
            
            # Risk management
            atr = self._calculate_atr(df)
            direction = 1 if adv_signal.signal_type.value > 0 else -1
            
            stop_loss = current_price - (2.0 * atr * direction)
            take_profit_1 = current_price + (1.5 * atr * direction)
            take_profit_2 = current_price + (3.0 * atr * direction)
            
            # Position sizing
            risk_per_trade = 0.02
            position_size = risk_per_trade / (atr / current_price + 0.001)
            position_size = min(0.25, max(0.02, position_size))
            
            # Risk/reward
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit_1 - current_price)
            rr_ratio = reward / (risk + 0.001)
            
            # Format agent scores (convert numpy types to Python types)
            agent_scores = {}
            for name, contrib in adv_signal.agent_contributions.items():
                agent_scores[name] = {
                    "name": contrib.name,
                    "weight": float(contrib.weight),
                    "score": float(contrib.adjusted_score),
                    "signal": contrib.signal,
                    "confidence": float(contrib.confidence),
                    "explanation": contrib.explanation
                }
            
            # Format indicator details (convert numpy types to Python types)
            indicator_details = []
            for ind in adv_signal.indicator_signals:
                indicator_details.append({
                    "name": ind.name,
                    "agent": ind.agent_category,
                    "value": float(ind.raw_value) if not pd.isna(ind.raw_value) else 0.0,
                    "score": float(ind.normalized_score) if not pd.isna(ind.normalized_score) else 0.0,
                    "weight": float(ind.weight),
                    "contribution": float(ind.contribution) if not pd.isna(ind.contribution) else 0.0,
                    "is_bullish": bool(ind.is_bullish),  # Convert numpy bool to Python bool
                    "confidence": float(ind.confidence) if not pd.isna(ind.confidence) else 0.0,
                    "explanation": str(ind.explanation)
                })
            
            return AdvancedPrediction(
                ticker=ticker,
                current_price=current_price,
                signal_type=adv_signal.signal_type,
                signal_name=adv_signal.signal_type.name,
                confidence=adv_signal.confidence,
                strength=adv_signal.strength,
                expected_return=adv_signal.expected_return,
                pred_return_1d=pred_1d,
                pred_return_3d=pred_3d,
                pred_return_1w=pred_1w,
                regime=adv_signal.regime.value,
                regime_explanation=self.REGIME_EXPLANATIONS.get(adv_signal.regime, ""),
                sector=adv_signal.sector.value,
                sector_impact=self.SECTOR_IMPACTS.get(adv_signal.sector, ""),
                agent_scores=agent_scores,
                indicator_details=indicator_details,
                bullish_indicators=sum(1 for i in adv_signal.indicator_signals if i.is_bullish),
                bearish_indicators=sum(1 for i in adv_signal.indicator_signals if not i.is_bullish),
                trade_thesis=adv_signal.trade_thesis,
                why_this_trade=adv_signal.why_this_trade,
                key_factors=adv_signal.key_factors,
                risk_factors=adv_signal.risk_factors,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                position_size_pct=position_size,
                risk_reward_ratio=rr_ratio,
                effective_agent_weights=adv_signal.effective_weights,
                regime_multipliers=adv_signal.regime_multipliers,
                sector_multipliers=adv_signal.sector_multipliers,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            import traceback
            print(f"Error analyzing {ticker}: {e}")
            traceback.print_exc()
            return None
    
    def analyze_multiple(self, tickers: List[str]) -> List[AdvancedPrediction]:
        """Analyze multiple tickers."""
        predictions = []
        
        for ticker in tickers:
            print(f"  [ADV] Analyzing {ticker}...", end=" ", flush=True)
            try:
                pred = self.analyze_ticker(ticker)
                if pred:
                    predictions.append(pred)
                    print(f"✓ {pred.signal_name} ({pred.bullish_indicators}B/{pred.bearish_indicators}S)", flush=True)
                else:
                    print("NO DATA", flush=True)
            except Exception as e:
                print(f"ERROR: {str(e)[:40]}", flush=True)
        
        # Sort by signal strength
        predictions.sort(key=lambda p: p.strength * p.confidence, reverse=True)
        return predictions
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj
    
    def save_predictions(self, predictions: List[AdvancedPrediction], 
                        path: str = "data/advanced_signals.json"):
        """Save advanced predictions to JSON with atomic write."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = []
        for p in predictions:
            try:
                record = {
                    "ticker": str(p.ticker),
                    "current_price": float(p.current_price),
                    "signal": str(p.signal_name),
                    "confidence": float(p.confidence),
                    "strength": float(p.strength),
                    "expected_return": float(p.expected_return),
                    "pred_return_1d": float(p.pred_return_1d),
                    "pred_return_3d": float(p.pred_return_3d),
                    "pred_return_1w": float(p.pred_return_1w),
                    "regime": str(p.regime),
                    "regime_explanation": str(p.regime_explanation),
                    "sector": str(p.sector),
                    "sector_impact": str(p.sector_impact),
                    "agent_scores": self._convert_to_serializable(p.agent_scores),
                    "indicator_details": self._convert_to_serializable(p.indicator_details),
                    "bullish_indicators": int(p.bullish_indicators),
                    "bearish_indicators": int(p.bearish_indicators),
                    "trade_thesis": str(p.trade_thesis),
                    "why_this_trade": str(p.why_this_trade),
                    "key_factors": [str(f) for f in p.key_factors],
                    "risk_factors": [str(f) for f in p.risk_factors],
                    "stop_loss": float(p.stop_loss),
                    "take_profit_1": float(p.take_profit_1),
                    "take_profit_2": float(p.take_profit_2),
                    "position_size_pct": float(p.position_size_pct),
                    "risk_reward_ratio": float(p.risk_reward_ratio),
                    "effective_agent_weights": self._convert_to_serializable(p.effective_agent_weights),
                    "regime_multipliers": self._convert_to_serializable(p.regime_multipliers),
                    "sector_multipliers": self._convert_to_serializable(p.sector_multipliers),
                    "timestamp": str(p.timestamp.isoformat())
                }
                data.append(record)
            except Exception as e:
                print(f"Warning: Failed to serialize prediction for {p.ticker}: {e}")
                continue
        
        # Atomic write: write to temp file first, then rename
        temp_path = path + ".tmp"
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Rename temp file to final file (atomic on most systems)
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise e
        
        return path
    
    def generate_report(self, predictions: List[AdvancedPrediction]) -> Dict:
        """Generate summary report."""
        if not predictions:
            return {"error": "No predictions available"}
        
        bullish = [p for p in predictions if p.signal_type.value > 0]
        bearish = [p for p in predictions if p.signal_type.value < 0]
        neutral = [p for p in predictions if p.signal_type.value == 0]
        
        # Regime distribution
        regimes = {}
        for p in predictions:
            regimes[p.regime] = regimes.get(p.regime, 0) + 1
        
        # Top picks
        top_longs = sorted(bullish, key=lambda p: p.strength * p.confidence, reverse=True)[:5]
        top_shorts = sorted(bearish, key=lambda p: p.strength * p.confidence, reverse=True)[:5]
        
        return {
            "generated_at": datetime.now().isoformat(),
            "total_analyzed": len(predictions),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "neutral_count": len(neutral),
            "avg_confidence": np.mean([p.confidence for p in predictions]),
            "avg_strength": np.mean([p.strength for p in predictions]),
            "regime_distribution": regimes,
            "market_sentiment": "BULLISH" if len(bullish) > len(bearish) * 1.5 else 
                               "BEARISH" if len(bearish) > len(bullish) * 1.5 else "MIXED",
            "top_longs": [
                {
                    "ticker": p.ticker,
                    "price": p.current_price,
                    "signal": p.signal_name,
                    "confidence": p.confidence,
                    "strength": p.strength,
                    "regime": p.regime,
                    "bullish_count": p.bullish_indicators,
                    "bearish_count": p.bearish_indicators,
                    "thesis": p.trade_thesis[:200] + "..." if len(p.trade_thesis) > 200 else p.trade_thesis
                }
                for p in top_longs
            ],
            "top_shorts": [
                {
                    "ticker": p.ticker,
                    "price": p.current_price,
                    "signal": p.signal_name,
                    "confidence": p.confidence,
                    "strength": p.strength,
                    "regime": p.regime,
                    "bullish_count": p.bullish_indicators,
                    "bearish_count": p.bearish_indicators,
                    "thesis": p.trade_thesis[:200] + "..." if len(p.trade_thesis) > 200 else p.trade_thesis
                }
                for p in top_shorts
            ]
        }
    
    def _estimate_horizon_return(self, signal: AdvancedSignal, hours: int) -> Tuple[float, float]:
        """Estimate return for specific horizon."""
        base_return = signal.expected_return
        horizon_factor = np.sqrt(hours / self.horizon_hours)
        adjusted_return = base_return * horizon_factor
        
        conf_decay = 0.95 ** (hours / 24)
        adjusted_conf = signal.confidence * conf_decay
        
        return float(adjusted_return), float(adjusted_conf)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        high = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
        low = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        atr = tr['tr'].ewm(span=period, adjust=False).mean().iloc[-1]
        return float(atr) if not pd.isna(atr) else float(close.iloc[-1] * 0.02)


def main():
    """Run the advanced multi-agent system."""
    import yaml
    import sys
    import io
    
    # Fix Windows encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("=" * 70, flush=True)
    print("[*] ADVANCED MULTI-AGENT TRADING SYSTEM", flush=True)
    print("    18 Indicators | 5 Agent Categories | Dynamic Weights", flush=True)
    print("    Regime Detection | Sector Adjustments | Trade Explanations", flush=True)
    print("=" * 70, flush=True)
    
    # Load universe
    try:
        uni = yaml.safe_load(open("config/universe.yaml"))
        tickers = uni.get('tickers', [])[:20]
    except:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'MA']
    
    print(f"\n[+] Analyzing {len(tickers)} tickers...\n", flush=True)
    
    # Run analysis
    orchestrator = AdvancedOrchestrator(horizon_hours=168)
    predictions = orchestrator.analyze_multiple(tickers)
    
    if not predictions:
        print("\n[X] No predictions generated!", flush=True)
        return
    
    # Generate report
    report = orchestrator.generate_report(predictions)
    
    # Save predictions
    save_path = orchestrator.save_predictions(predictions)
    
    # Print summary
    print(f"\n{'='*70}", flush=True)
    print("[+] ADVANCED ANALYSIS COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"    Total Analyzed: {report['total_analyzed']}", flush=True)
    print(f"    Bullish: {report['bullish_count']}", flush=True)
    print(f"    Bearish: {report['bearish_count']}", flush=True)
    print(f"    Neutral: {report['neutral_count']}", flush=True)
    print(f"    Sentiment: {report['market_sentiment']}", flush=True)
    print(f"    Avg Confidence: {report['avg_confidence']*100:.1f}%", flush=True)
    
    print(f"\n[REGIMES] Distribution:", flush=True)
    for regime, count in report['regime_distribution'].items():
        print(f"    {regime}: {count} stocks", flush=True)
    
    print(f"\n[LONG] Top Long Recommendations:", flush=True)
    for sig in report['top_longs']:
        print(f"    {sig['ticker']}: {sig['signal']} | "
              f"Conf: {sig['confidence']*100:.0f}% | "
              f"Regime: {sig['regime']} | "
              f"Indicators: {sig['bullish_count']}B/{sig['bearish_count']}S", flush=True)
    
    print(f"\n[SHORT] Top Short Recommendations:", flush=True)
    for sig in report['top_shorts']:
        print(f"    {sig['ticker']}: {sig['signal']} | "
              f"Conf: {sig['confidence']*100:.0f}% | "
              f"Regime: {sig['regime']} | "
              f"Indicators: {sig['bullish_count']}B/{sig['bearish_count']}S", flush=True)
    
    print(f"\n[SAVED] Advanced signals: {save_path}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

