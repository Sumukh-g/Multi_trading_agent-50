"""
Volatility Agent

Specializes in volatility-based trading using:
- Bollinger Bands
- Keltner Channels  
- ATR analysis
- Historical volatility
- Volatility regime detection
- Squeeze detection (BB inside Keltner)
"""

import pandas as pd
import numpy as np
from typing import Dict
from .base_agent import BaseAgent, AgentSignal, AgentConfig, SignalType


class VolatilityAgent(BaseAgent):
    """
    Advanced volatility analysis agent.
    
    Identifies volatility regimes, breakout conditions,
    and mean reversion opportunities.
    """
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(name="VolatilityAgent", weight=1.0)
        super().__init__(config)
        
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std = self.params.get('bb_std', 2.0)
        self.atr_period = self.params.get('atr_period', 14)
        self.keltner_period = self.params.get('keltner_period', 20)
        self.keltner_mult = self.params.get('keltner_mult', 1.5)
    
    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility-related features."""
        # Handle potential Series vs DataFrame issues
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        high = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
        low = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
        
        features = pd.DataFrame(index=df.index)
        
        # Bollinger Bands
        bb_mid = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        features['bb_upper'] = bb_mid + (self.bb_std * bb_std)
        features['bb_lower'] = bb_mid - (self.bb_std * bb_std)
        features['bb_mid'] = bb_mid
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_mid
        features['bb_pct'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
        
        # ATR
        features['atr'] = self._calculate_atr(high, low, close, self.atr_period)
        features['atr_pct'] = features['atr'] / close
        
        # Keltner Channels
        kc_mid = close.ewm(span=self.keltner_period, adjust=False).mean()
        features['kc_upper'] = kc_mid + (self.keltner_mult * features['atr'])
        features['kc_lower'] = kc_mid - (self.keltner_mult * features['atr'])
        features['kc_mid'] = kc_mid
        
        # Squeeze detection (BB inside Keltner)
        features['squeeze'] = ((features['bb_lower'] > features['kc_lower']) & 
                               (features['bb_upper'] < features['kc_upper'])).astype(int)
        
        # Historical volatility
        returns = close.pct_change()
        features['hvol_10'] = returns.rolling(10).std() * np.sqrt(252 * 6.5)  # Hourly annualized
        features['hvol_20'] = returns.rolling(20).std() * np.sqrt(252 * 6.5)
        features['hvol_50'] = returns.rolling(50).std() * np.sqrt(252 * 6.5)
        
        # Volatility ratio
        features['vol_ratio'] = features['hvol_10'] / (features['hvol_50'] + 1e-10)
        
        # Volatility regime (low, normal, high)
        vol_percentile = features['hvol_20'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        features['vol_regime'] = vol_percentile.apply(
            lambda x: -1 if x < 0.25 else (1 if x > 0.75 else 0) if not pd.isna(x) else 0
        )
        
        # Bollinger Band breakout detection
        features['bb_breakout_up'] = (close > features['bb_upper']).astype(int)
        features['bb_breakout_down'] = (close < features['bb_lower']).astype(int)
        
        # Price distance from mean (for mean reversion)
        features['zscore'] = (close - bb_mid) / (bb_std + 1e-10)
        
        return features
    
    def analyze(self, df: pd.DataFrame, ticker: str) -> AgentSignal:
        """Analyze volatility conditions and generate signal."""
        features = self.get_features(df)
        close_series = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        close = float(close_series.iloc[-1])
        
        # Get latest values
        bb_pct = self._safe_value(features['bb_pct'], 0.5)
        bb_width = self._safe_value(features['bb_width'])
        squeeze = self._safe_value(features['squeeze'])
        vol_ratio = self._safe_value(features['vol_ratio'], 1.0)
        vol_regime = self._safe_value(features['vol_regime'])
        zscore = self._safe_value(features['zscore'])
        atr_pct = self._safe_value(features['atr_pct'])
        bb_breakout_up = self._safe_value(features['bb_breakout_up'])
        bb_breakout_down = self._safe_value(features['bb_breakout_down'])
        
        # Calculate signal
        signal_score = 0
        
        # Bollinger Band position (mean reversion)
        if bb_pct < 0.1:
            signal_score += 20  # Near lower band = potential buy
        elif bb_pct > 0.9:
            signal_score -= 20  # Near upper band = potential sell
        
        # Z-score mean reversion
        if zscore < -2:
            signal_score += 25  # Extreme oversold
        elif zscore > 2:
            signal_score -= 25  # Extreme overbought
        elif abs(zscore) < 0.5:
            # Near mean, look for breakout
            if squeeze:
                signal_score += 5 * np.sign(features['bb_pct'].diff().iloc[-5:].mean())
        
        # Squeeze breakout
        squeeze_ending = (features['squeeze'].iloc[-1] == 0 and 
                         features['squeeze'].iloc[-2] == 1)
        if squeeze_ending:
            # Direction of breakout
            if close > features['bb_mid'].iloc[-1]:
                signal_score += 30
            else:
                signal_score -= 30
        
        # Volatility expansion/contraction
        if vol_ratio > 1.5:
            # Volatility expanding - favor momentum
            signal_score *= 1.2
        elif vol_ratio < 0.7:
            # Volatility contracting - expect breakout soon
            pass
        
        # Regime-based adjustment
        if vol_regime == 1:  # High vol regime
            signal_score *= 0.8  # Be more cautious
        elif vol_regime == -1:  # Low vol regime
            signal_score *= 1.1  # More confident signals
        
        # Determine signal type
        if signal_score > 35:
            signal_type = SignalType.STRONG_BUY
        elif signal_score > 15:
            signal_type = SignalType.BUY
        elif signal_score < -35:
            signal_type = SignalType.STRONG_SELL
        elif signal_score < -15:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Confidence based on volatility conditions
        confidence = 0.5
        if abs(zscore) > 1.5:
            confidence += 0.2
        if squeeze_ending:
            confidence += 0.2
        if vol_regime == -1:
            confidence += 0.1
        confidence = min(1.0, confidence)
        
        reasoning = self._generate_reasoning(bb_pct, zscore, squeeze, 
                                            vol_ratio, vol_regime, squeeze_ending)
        
        signal = AgentSignal(
            signal_type=signal_type,
            confidence=confidence,
            strength=abs(signal_score),
            reasoning=reasoning,
            metrics={
                'bb_pct': bb_pct,
                'bb_width': bb_width,
                'zscore': zscore,
                'squeeze': squeeze,
                'vol_ratio': vol_ratio,
                'vol_regime': vol_regime,
                'atr_pct': atr_pct,
                'signal_score': signal_score
            },
            timestamp=df.index[-1]
        )
        
        self.update_signal(signal)
        return signal
    
    def _calculate_atr(self, high, low, close, period):
        """Calculate Average True Range."""
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return tr['tr'].ewm(span=period, adjust=False).mean()
    
    def _generate_reasoning(self, bb_pct, zscore, squeeze, vol_ratio, vol_regime, squeeze_ending):
        """Generate human-readable reasoning."""
        parts = []
        
        if zscore < -2:
            parts.append(f"Extreme oversold (Z={zscore:.2f})")
        elif zscore > 2:
            parts.append(f"Extreme overbought (Z={zscore:.2f})")
        elif bb_pct < 0.2:
            parts.append("Near lower Bollinger Band")
        elif bb_pct > 0.8:
            parts.append("Near upper Bollinger Band")
        
        if squeeze:
            parts.append("SQUEEZE active - consolidation")
        if squeeze_ending:
            parts.append("SQUEEZE BREAKOUT imminent")
        
        if vol_ratio > 1.3:
            parts.append("Volatility expanding")
        elif vol_ratio < 0.7:
            parts.append("Volatility contracting")
        
        regime_names = {-1: "low", 0: "normal", 1: "high"}
        parts.append(f"Volatility regime: {regime_names.get(vol_regime, 'unknown')}")
        
        return ". ".join(parts) + "."

