"""
Meta Agent (ML-Based Ensemble)

Uses machine learning to combine signals from all agents:
- LightGBM for signal combination
- Confidence-weighted voting
- Dynamic weight adjustment
- Regime-aware predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import joblib
import os

from .base_agent import BaseAgent, AgentSignal, AgentConfig, SignalType


@dataclass
class MetaSignal:
    """Combined signal from all agents."""
    signal_type: SignalType
    confidence: float
    strength: float  # 0-100
    expected_return: float  # Expected return over horizon
    risk_score: float  # 0-1, higher = riskier
    horizon_hours: int
    reasoning: str
    agent_signals: Dict[str, AgentSignal]
    agent_weights: Dict[str, float]
    timestamp: pd.Timestamp


class MetaAgent:
    """
    Meta-learning agent that combines signals from specialized agents.
    
    Uses multiple combination methods:
    1. Confidence-weighted voting
    2. ML-based stacking (if trained)
    3. Dynamic weight adjustment based on recent performance
    """
    
    def __init__(self, agents: List[BaseAgent], horizon_hours: int = 168):
        self.agents = {agent.name: agent for agent in agents}
        self.horizon_hours = horizon_hours  # Default 1 week (168 hours)
        
        # Default weights (can be adjusted dynamically)
        self.weights = {
            "TrendAgent": 1.2,
            "MomentumAgent": 1.0,
            "VolatilityAgent": 1.0,
            "VolumeAgent": 0.9,
            "PatternAgent": 1.1
        }
        
        # Load ML model if available
        self.ml_model = None
        self._load_ml_model()
        
        # Performance tracking for dynamic weights
        self.performance_history: List[Dict] = []
    
    def _load_ml_model(self):
        """Load the trained ML stacking model if available."""
        model_path = "models/short_term_meta.pkl"
        if os.path.exists(model_path):
            try:
                self.ml_model = joblib.load(model_path)
            except:
                pass
    
    def analyze(self, df: pd.DataFrame, ticker: str) -> MetaSignal:
        """
        Run all agents and combine their signals.
        
        Args:
            df: OHLCV DataFrame with hourly data
            ticker: Stock symbol
            
        Returns:
            MetaSignal with combined recommendation
        """
        # Collect signals from all agents
        agent_signals = {}
        for name, agent in self.agents.items():
            try:
                signal = agent.analyze(df, ticker)
                agent_signals[name] = signal
            except Exception as e:
                print(f"Agent {name} failed: {e}")
                continue
        
        if not agent_signals:
            return self._empty_signal(ticker, df.index[-1])
        
        # Combine signals
        if self.ml_model is not None:
            combined = self._ml_combine(agent_signals, df)
        else:
            combined = self._weighted_vote(agent_signals)
        
        # Calculate expected return
        expected_return = self._estimate_return(agent_signals, combined, df)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(agent_signals, df)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(agent_signals, combined)
        
        return MetaSignal(
            signal_type=combined['signal_type'],
            confidence=combined['confidence'],
            strength=combined['strength'],
            expected_return=expected_return,
            risk_score=risk_score,
            horizon_hours=self.horizon_hours,
            reasoning=reasoning,
            agent_signals=agent_signals,
            agent_weights=self.weights.copy(),
            timestamp=df.index[-1]
        )
    
    def _weighted_vote(self, signals: Dict[str, AgentSignal]) -> Dict:
        """Combine signals using confidence-weighted voting."""
        total_weight = 0
        weighted_score = 0
        weighted_confidence = 0
        weighted_strength = 0
        
        for name, signal in signals.items():
            weight = self.weights.get(name, 1.0) * signal.confidence
            total_weight += weight
            
            # Convert signal type to numeric (-2 to +2)
            score = signal.signal_type.value
            
            weighted_score += score * weight
            weighted_confidence += signal.confidence * weight
            weighted_strength += signal.strength * weight
        
        if total_weight == 0:
            return {
                'signal_type': SignalType.NEUTRAL,
                'confidence': 0.0,
                'strength': 0.0
            }
        
        avg_score = weighted_score / total_weight
        avg_confidence = weighted_confidence / total_weight
        avg_strength = weighted_strength / total_weight
        
        # Determine signal type from average score
        if avg_score > 1.2:
            signal_type = SignalType.STRONG_BUY
        elif avg_score > 0.4:
            signal_type = SignalType.BUY
        elif avg_score < -1.2:
            signal_type = SignalType.STRONG_SELL
        elif avg_score < -0.4:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Boost confidence if agents agree
        agreement = self._calculate_agreement(signals)
        final_confidence = min(1.0, avg_confidence * (0.7 + 0.3 * agreement))
        
        return {
            'signal_type': signal_type,
            'confidence': final_confidence,
            'strength': avg_strength,
            'raw_score': avg_score
        }
    
    def _ml_combine(self, signals: Dict[str, AgentSignal], df: pd.DataFrame) -> Dict:
        """Use ML model to combine signals."""
        # Build feature vector from agent signals
        features = []
        
        for name in ['TrendAgent', 'MomentumAgent', 'VolatilityAgent', 
                     'VolumeAgent', 'PatternAgent']:
            if name in signals:
                signal = signals[name]
                features.extend([
                    signal.signal_type.value,
                    signal.confidence,
                    signal.strength
                ])
                # Add key metrics
                for key in ['adx', 'rsi', 'bb_pct', 'vol_ratio', 'trend_structure']:
                    features.append(signal.metrics.get(key, 0))
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0, 0])  # Padding
        
        # Predict
        try:
            X = np.array(features).reshape(1, -1)
            pred = self.ml_model.predict_proba(X)[0]
            
            # Map to signal type (assuming 5-class classification)
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            
            signal_map = {
                0: SignalType.STRONG_SELL,
                1: SignalType.SELL,
                2: SignalType.NEUTRAL,
                3: SignalType.BUY,
                4: SignalType.STRONG_BUY
            }
            
            return {
                'signal_type': signal_map.get(class_idx, SignalType.NEUTRAL),
                'confidence': confidence,
                'strength': confidence * 100
            }
        except:
            # Fall back to weighted voting
            return self._weighted_vote(signals)
    
    def _calculate_agreement(self, signals: Dict[str, AgentSignal]) -> float:
        """Calculate how much agents agree (0-1)."""
        if len(signals) < 2:
            return 0.5
        
        directions = [s.signal_type.value for s in signals.values()]
        
        # Count bullish vs bearish
        bullish = sum(1 for d in directions if d > 0)
        bearish = sum(1 for d in directions if d < 0)
        neutral = sum(1 for d in directions if d == 0)
        
        total = len(directions)
        max_agreement = max(bullish, bearish, neutral)
        
        return max_agreement / total
    
    def _estimate_return(self, signals: Dict[str, AgentSignal], 
                        combined: Dict, df: pd.DataFrame) -> float:
        """Estimate expected return over the horizon."""
        # Base return from signal strength
        signal_value = combined['signal_type'].value
        base_return = signal_value * 0.01 * combined['strength'] / 100
        
        # Adjust for volatility - handle potential DataFrame/Series
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        vol = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(self.horizon_hours)
        
        # Scale return by volatility
        if not pd.isna(vol):
            base_return *= min(2.0, 0.02 / (vol + 0.01))
        
        return np.clip(base_return, -0.15, 0.15)
    
    def _calculate_risk_score(self, signals: Dict[str, AgentSignal], 
                             df: pd.DataFrame) -> float:
        """Calculate risk score (0-1, higher = riskier)."""
        risk = 0.5  # Base risk
        
        # Volatility contribution - handle potential DataFrame/Series
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        current_vol = close.pct_change().rolling(10).std().iloc[-1]
        avg_vol = close.pct_change().rolling(50).std().iloc[-1]
        
        if not pd.isna(current_vol) and not pd.isna(avg_vol) and avg_vol > 0:
            vol_ratio = current_vol / avg_vol
            if vol_ratio > 1.5:
                risk += 0.2
            elif vol_ratio < 0.7:
                risk -= 0.1
        
        # Disagreement contribution
        agreement = self._calculate_agreement(signals)
        if agreement < 0.5:
            risk += 0.15
        
        # Low confidence contribution
        avg_conf = np.mean([s.confidence for s in signals.values()])
        if avg_conf < 0.4:
            risk += 0.15
        
        return np.clip(risk, 0, 1)
    
    def _generate_reasoning(self, signals: Dict[str, AgentSignal], 
                           combined: Dict) -> str:
        """Generate comprehensive reasoning from all agents."""
        parts = []
        
        # Overall direction
        signal_type = combined['signal_type']
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            parts.append(f"[+] BULLISH signal (confidence: {combined['confidence']*100:.0f}%)")
        elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            parts.append(f"[-] BEARISH signal (confidence: {combined['confidence']*100:.0f}%)")
        else:
            parts.append(f"[=] NEUTRAL signal (confidence: {combined['confidence']*100:.0f}%)")
        
        # Agent breakdown
        parts.append("\n[AGENTS] Agent Analysis:")
        for name, signal in signals.items():
            indicator = "[+]" if signal.signal_type.value > 0 else "[-]" if signal.signal_type.value < 0 else "[=]"
            parts.append(f"  {indicator} {name}: {signal.signal_type.name} ({signal.confidence*100:.0f}%)")
        
        # Key insights from agents
        parts.append("\n[INFO] Key Insights:")
        for name, signal in signals.items():
            if signal.reasoning:
                parts.append(f"  * {name}: {signal.reasoning[:100]}")
        
        return "\n".join(parts)
    
    def _empty_signal(self, ticker: str, timestamp) -> MetaSignal:
        """Return empty signal when no agents succeed."""
        return MetaSignal(
            signal_type=SignalType.NEUTRAL,
            confidence=0.0,
            strength=0.0,
            expected_return=0.0,
            risk_score=1.0,
            horizon_hours=self.horizon_hours,
            reasoning="No agent signals available",
            agent_signals={},
            agent_weights=self.weights.copy(),
            timestamp=timestamp
        )
    
    def update_weights(self, performance: Dict[str, float]):
        """Update agent weights based on recent performance."""
        for name, perf in performance.items():
            if name in self.weights:
                # Adjust weight based on performance (e.g., Sharpe ratio)
                adjustment = np.clip(perf / 2, -0.3, 0.3)
                self.weights[name] = max(0.3, min(2.0, self.weights[name] + adjustment))

