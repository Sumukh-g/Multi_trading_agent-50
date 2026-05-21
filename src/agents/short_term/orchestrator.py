"""
Multi-Agent Orchestrator for Short-Term Trading

Coordinates all agents and provides a unified interface for:
- Downloading hourly data
- Running all agents
- Generating combined predictions
- 1-week (168 hours) forward predictions
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

from .base_agent import AgentConfig
from .trend_agent import TrendAgent
from .momentum_agent import MomentumAgent
from .volatility_agent import VolatilityAgent
from .volume_agent import VolumeAgent
from .pattern_agent import PatternAgent
from .meta_agent import MetaAgent, MetaSignal


@dataclass
class ShortTermPrediction:
    """Complete prediction for a ticker."""
    ticker: str
    current_price: float
    meta_signal: MetaSignal
    predicted_return_1d: float
    predicted_return_3d: float
    predicted_return_1w: float
    confidence_1d: float
    confidence_3d: float
    confidence_1w: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size_pct: float  # Suggested position size as % of portfolio
    risk_reward_ratio: float
    timestamp: datetime


class ShortTermOrchestrator:
    """
    Main orchestrator for the short-term multi-agent trading system.
    
    Coordinates:
    - Data fetching (hourly bars)
    - Agent execution
    - Signal combination
    - Risk management recommendations
    """
    
    def __init__(self, horizon_hours: int = 168):
        """
        Initialize the orchestrator with all agents.
        
        Args:
            horizon_hours: Primary prediction horizon (default 168 = 1 week)
        """
        self.horizon_hours = horizon_hours
        
        # Initialize all specialized agents
        self.agents = [
            TrendAgent(AgentConfig(name="TrendAgent", weight=1.2)),
            MomentumAgent(AgentConfig(name="MomentumAgent", weight=1.0)),
            VolatilityAgent(AgentConfig(name="VolatilityAgent", weight=1.0)),
            VolumeAgent(AgentConfig(name="VolumeAgent", weight=0.9)),
            PatternAgent(AgentConfig(name="PatternAgent", weight=1.1))
        ]
        
        # Initialize meta-agent for signal combination
        self.meta_agent = MetaAgent(self.agents, horizon_hours)
        
        # Cache for downloaded data
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.cache_expiry = timedelta(hours=1)
        self.cache_timestamps: Dict[str, datetime] = {}
    
    def fetch_hourly_data(self, ticker: str, lookback_days: int = 60) -> Optional[pd.DataFrame]:
        """
        Fetch hourly OHLCV data for a ticker.
        
        Args:
            ticker: Stock symbol
            lookback_days: Number of days of history to fetch
            
        Returns:
            DataFrame with hourly OHLCV data or None if failed
        """
        # Check cache
        now = datetime.now()
        if ticker in self.data_cache:
            if now - self.cache_timestamps.get(ticker, datetime.min) < self.cache_expiry:
                return self.data_cache[ticker]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Download hourly data (max 730 days for hourly)
                df = yf.download(
                    ticker,
                    period=f"{min(lookback_days, 59)}d",  # Yahoo limits hourly to ~60 days
                    interval="1h",
                    auto_adjust=True,
                    progress=False
                )
                
                if df.empty or len(df) < 100:
                    return None
                
                # Clean column names - handle multi-index
                if isinstance(df.columns, pd.MultiIndex):
                    # Get the first level (price type)
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Ensure proper column names
                df.columns = [str(c).title() for c in df.columns]
                
                # Verify required columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [c for c in required_cols if c not in df.columns]
                if missing_cols:
                    print(f"Missing columns for {ticker}: {missing_cols}")
                    return None
                
                # Ensure numeric types
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop any rows with NaN in Close
                df = df.dropna(subset=['Close'])
                
                if len(df) < 100:
                    return None
                
                # Cache the data
                self.data_cache[ticker] = df
                self.cache_timestamps[ticker] = now
                
                return df
                
        except Exception as e:
            print(f"Error fetching hourly data for {ticker}: {e}")
            return None
    
    def analyze_ticker(self, ticker: str) -> Optional[ShortTermPrediction]:
        """
        Run complete analysis for a single ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            ShortTermPrediction or None if analysis failed
        """
        # Fetch data
        df = self.fetch_hourly_data(ticker)
        if df is None:
            return None
        
        try:
            # Get current price - handle potential DataFrame/Series
            close_col = df['Close']
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            current_price = float(close_col.iloc[-1])
            
            # Run meta-agent analysis
            meta_signal = self.meta_agent.analyze(df, ticker)
            
            # Calculate multi-horizon predictions
            pred_1d, conf_1d = self._estimate_horizon_return(meta_signal, 24, df)
            pred_3d, conf_3d = self._estimate_horizon_return(meta_signal, 72, df)
            pred_1w, conf_1w = self._estimate_horizon_return(meta_signal, 168, df)
            
            # Calculate risk management levels
            atr = self._calculate_atr(df)
            direction = 1 if meta_signal.signal_type.value > 0 else -1
            
            stop_loss = current_price - (2.0 * atr * direction)
            take_profit_1 = current_price + (1.5 * atr * direction)
            take_profit_2 = current_price + (3.0 * atr * direction)
            
            # Position sizing based on risk
            risk_per_trade = 0.02  # 2% risk per trade
            position_size = risk_per_trade / (atr / current_price + 0.001)
            position_size = min(0.25, max(0.02, position_size))  # 2-25% of portfolio
            
            # Risk/reward ratio
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit_1 - current_price)
            rr_ratio = reward / (risk + 0.001)
            
            return ShortTermPrediction(
                ticker=ticker,
                current_price=current_price,
                meta_signal=meta_signal,
                predicted_return_1d=pred_1d,
                predicted_return_3d=pred_3d,
                predicted_return_1w=pred_1w,
                confidence_1d=conf_1d,
                confidence_3d=conf_3d,
                confidence_1w=conf_1w,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                position_size_pct=position_size,
                risk_reward_ratio=rr_ratio,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            import traceback
            print(f"Error analyzing {ticker}: {e}", flush=True)
            traceback.print_exc()
            return None
    
    def analyze_multiple(self, tickers: List[str]) -> List[ShortTermPrediction]:
        """
        Analyze multiple tickers.
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            List of ShortTermPrediction objects
        """
        predictions = []
        
        for ticker in tickers:
            print(f"Analyzing {ticker}...", end=" ", flush=True)
            try:
                pred = self.analyze_ticker(ticker)
                if pred:
                    predictions.append(pred)
                    signal_type = pred.meta_signal.signal_type.name
                    print(f"OK {signal_type} (conf: {pred.meta_signal.confidence*100:.0f}%)", flush=True)
                else:
                    print("NO DATA", flush=True)
            except Exception as e:
                print(f"ERROR: {str(e)[:50]}", flush=True)
        
        # Sort by signal strength
        predictions.sort(key=lambda p: p.meta_signal.strength, reverse=True)
        
        return predictions
    
    def get_top_signals(self, tickers: List[str], top_n: int = 5) -> Tuple[List[ShortTermPrediction], List[ShortTermPrediction]]:
        """
        Get top long and short signals.
        
        Args:
            tickers: List of tickers to analyze
            top_n: Number of top signals to return
            
        Returns:
            Tuple of (top_longs, top_shorts)
        """
        predictions = self.analyze_multiple(tickers)
        
        longs = [p for p in predictions if p.meta_signal.signal_type.value > 0]
        shorts = [p for p in predictions if p.meta_signal.signal_type.value < 0]
        
        # Sort by strength * confidence
        longs.sort(key=lambda p: p.meta_signal.strength * p.meta_signal.confidence, reverse=True)
        shorts.sort(key=lambda p: p.meta_signal.strength * p.meta_signal.confidence, reverse=True)
        
        return longs[:top_n], shorts[:top_n]
    
    def generate_report(self, predictions: List[ShortTermPrediction]) -> Dict:
        """
        Generate a summary report of all predictions.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Dictionary with report data
        """
        if not predictions:
            return {"error": "No predictions available"}
        
        bullish = [p for p in predictions if p.meta_signal.signal_type.value > 0]
        bearish = [p for p in predictions if p.meta_signal.signal_type.value < 0]
        neutral = [p for p in predictions if p.meta_signal.signal_type.value == 0]
        
        avg_confidence = np.mean([p.meta_signal.confidence for p in predictions])
        avg_strength = np.mean([p.meta_signal.strength for p in predictions])
        
        # Top picks
        top_longs = sorted(bullish, 
                          key=lambda p: p.meta_signal.strength * p.meta_signal.confidence,
                          reverse=True)[:5]
        top_shorts = sorted(bearish,
                           key=lambda p: p.meta_signal.strength * p.meta_signal.confidence,
                           reverse=True)[:5]
        
        return {
            "generated_at": datetime.now().isoformat(),
            "horizon_hours": self.horizon_hours,
            "total_analyzed": len(predictions),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "neutral_count": len(neutral),
            "avg_confidence": avg_confidence,
            "avg_strength": avg_strength,
            "market_sentiment": "BULLISH" if len(bullish) > len(bearish) * 1.5 else 
                               "BEARISH" if len(bearish) > len(bullish) * 1.5 else "MIXED",
            "top_longs": [
                {
                    "ticker": p.ticker,
                    "price": p.current_price,
                    "signal": p.meta_signal.signal_type.name,
                    "confidence": p.meta_signal.confidence,
                    "strength": p.meta_signal.strength,
                    "exp_return_1w": p.predicted_return_1w,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit_1
                }
                for p in top_longs
            ],
            "top_shorts": [
                {
                    "ticker": p.ticker,
                    "price": p.current_price,
                    "signal": p.meta_signal.signal_type.name,
                    "confidence": p.meta_signal.confidence,
                    "strength": p.meta_signal.strength,
                    "exp_return_1w": p.predicted_return_1w,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit_1
                }
                for p in top_shorts
            ]
        }
    
    def save_predictions(self, predictions: List[ShortTermPrediction], path: str = "data/short_term_signals.json"):
        """Save predictions to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = []
        for p in predictions:
            data.append({
                "ticker": p.ticker,
                "current_price": p.current_price,
                "signal": p.meta_signal.signal_type.name,
                "confidence": p.meta_signal.confidence,
                "strength": p.meta_signal.strength,
                "expected_return": p.meta_signal.expected_return,
                "risk_score": p.meta_signal.risk_score,
                "pred_return_1d": p.predicted_return_1d,
                "pred_return_3d": p.predicted_return_3d,
                "pred_return_1w": p.predicted_return_1w,
                "confidence_1d": p.confidence_1d,
                "confidence_3d": p.confidence_3d,
                "confidence_1w": p.confidence_1w,
                "stop_loss": p.stop_loss,
                "take_profit_1": p.take_profit_1,
                "take_profit_2": p.take_profit_2,
                "position_size_pct": p.position_size_pct,
                "risk_reward_ratio": p.risk_reward_ratio,
                "reasoning": p.meta_signal.reasoning,
                "timestamp": p.timestamp.isoformat()
            })
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return path
    
    def _estimate_horizon_return(self, meta_signal: MetaSignal, 
                                 hours: int, df: pd.DataFrame) -> Tuple[float, float]:
        """Estimate return and confidence for a specific horizon."""
        # Base on meta signal
        base_return = meta_signal.expected_return
        base_conf = meta_signal.confidence
        
        # Adjust for horizon
        horizon_factor = np.sqrt(hours / self.horizon_hours)
        adjusted_return = base_return * horizon_factor
        
        # Confidence decreases with longer horizons
        conf_decay = 0.95 ** (hours / 24)
        adjusted_conf = base_conf * conf_decay
        
        return float(adjusted_return), float(adjusted_conf)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        # Handle potential DataFrame/Series
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
    """Run the short-term trading system."""
    import yaml
    import sys
    import io
    
    # Fix Windows console encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("=" * 70, flush=True)
    print("[*] SHORT-TERM MULTI-AGENT TRADING SYSTEM", flush=True)
    print("    Prediction Horizon: 1 Week (168 hours)", flush=True)
    print("    Data: Hourly OHLCV", flush=True)
    print("=" * 70, flush=True)
    
    # Load universe
    try:
        uni = yaml.safe_load(open("config/universe.yaml"))
        tickers = uni.get('short_term_tickers', uni.get('tickers', []))[:20]
    except:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'MA']
    
    print(f"\nAnalyzing {len(tickers)} tickers...\n", flush=True)
    
    # Initialize and run
    orchestrator = ShortTermOrchestrator(horizon_hours=168)
    predictions = orchestrator.analyze_multiple(tickers)
    
    if not predictions:
        print("\n❌ No predictions generated!")
        return
    
    # Generate report
    report = orchestrator.generate_report(predictions)
    
    # Save predictions
    save_path = orchestrator.save_predictions(predictions)
    
    # Print summary
    print(f"\n{'='*70}", flush=True)
    print("[+] ANALYSIS COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"    Total Analyzed: {report['total_analyzed']}", flush=True)
    print(f"    Bullish: {report['bullish_count']}", flush=True)
    print(f"    Bearish: {report['bearish_count']}", flush=True)
    print(f"    Neutral: {report['neutral_count']}", flush=True)
    print(f"    Market Sentiment: {report['market_sentiment']}", flush=True)
    print(f"    Avg Confidence: {report['avg_confidence']*100:.1f}%", flush=True)
    
    print(f"\n[LONG] TOP LONG SIGNALS:", flush=True)
    for sig in report['top_longs']:
        print(f"    {sig['ticker']}: {sig['signal']} | "
              f"Conf: {sig['confidence']*100:.0f}% | "
              f"Exp 1W: {sig['exp_return_1w']*100:+.1f}%", flush=True)
    
    print(f"\n[SHORT] TOP SHORT SIGNALS:", flush=True)
    for sig in report['top_shorts']:
        print(f"    {sig['ticker']}: {sig['signal']} | "
              f"Conf: {sig['confidence']*100:.0f}% | "
              f"Exp 1W: {sig['exp_return_1w']*100:+.1f}%", flush=True)
    
    print(f"\n[SAVED] Saved to: {save_path}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()

