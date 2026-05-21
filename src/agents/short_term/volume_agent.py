"""
Volume Agent

Specializes in volume-based analysis using:
- Volume profile analysis
- VWAP deviation
- On-Balance Volume (OBV)
- Volume Rate of Change
- Accumulation/Distribution
- Chaikin Money Flow
"""

import pandas as pd
import numpy as np
from typing import Dict
from .base_agent import BaseAgent, AgentSignal, AgentConfig, SignalType


class VolumeAgent(BaseAgent):
    """
    Advanced volume analysis agent.
    
    Identifies accumulation/distribution patterns,
    volume-price divergences, and institutional activity.
    """
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(name="VolumeAgent", weight=0.9)
        super().__init__(config)
        
        self.obv_signal_period = self.params.get('obv_signal_period', 20)
        self.cmf_period = self.params.get('cmf_period', 20)
        self.vwap_period = self.params.get('vwap_period', 40)
    
    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volume-related features."""
        # Handle potential Series vs DataFrame issues
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        high = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
        low = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
        
        if 'Volume' in df.columns:
            volume = df['Volume'] if isinstance(df['Volume'], pd.Series) else df['Volume'].iloc[:, 0]
        else:
            volume = pd.Series(1000000, index=df.index)
        
        features = pd.DataFrame(index=df.index)
        
        # Basic volume metrics
        features['volume'] = volume
        features['vol_sma'] = volume.rolling(20).mean()
        features['vol_ratio'] = volume / (features['vol_sma'] + 1e-10)
        
        # On-Balance Volume
        features['obv'] = self._calculate_obv(close, volume)
        features['obv_sma'] = features['obv'].rolling(self.obv_signal_period).mean()
        features['obv_trend'] = (features['obv'] - features['obv_sma']) / (features['obv_sma'].abs() + 1e-10)
        
        # VWAP
        typical_price = (high + low + close) / 3
        features['vwap'] = (typical_price * volume).rolling(self.vwap_period).sum() / volume.rolling(self.vwap_period).sum()
        features['vwap_dev'] = (close - features['vwap']) / features['vwap']
        
        # Accumulation/Distribution Line
        features['ad_line'] = self._calculate_ad_line(high, low, close, volume)
        features['ad_sma'] = features['ad_line'].rolling(20).mean()
        
        # Chaikin Money Flow
        features['cmf'] = self._calculate_cmf(high, low, close, volume, self.cmf_period)
        
        # Volume Rate of Change
        features['vroc'] = volume.pct_change(10) * 100
        
        # Price-Volume Trend
        features['pvt'] = self._calculate_pvt(close, volume)
        
        # Volume-weighted price momentum
        features['vw_momentum'] = self._calculate_vw_momentum(close, volume, 10)
        
        # Volume climax detection
        features['vol_climax_up'] = ((volume > features['vol_sma'] * 2) & 
                                      (close > close.shift(1))).astype(int)
        features['vol_climax_down'] = ((volume > features['vol_sma'] * 2) & 
                                        (close < close.shift(1))).astype(int)
        
        # Volume divergence (price up, volume down or vice versa)
        price_trend = close.diff(5) > 0
        vol_trend = volume.rolling(5).mean() > volume.rolling(20).mean()
        features['vol_divergence'] = (price_trend != vol_trend).astype(int)
        
        return features
    
    def analyze(self, df: pd.DataFrame, ticker: str) -> AgentSignal:
        """Analyze volume patterns and generate signal."""
        features = self.get_features(df)
        
        # Get latest values
        vol_ratio = self._safe_value(features['vol_ratio'], 1.0)
        obv_trend = self._safe_value(features['obv_trend'])
        vwap_dev = self._safe_value(features['vwap_dev'])
        cmf = self._safe_value(features['cmf'])
        vroc = self._safe_value(features['vroc'])
        vw_momentum = self._safe_value(features['vw_momentum'])
        vol_climax_up = self._safe_value(features['vol_climax_up'])
        vol_climax_down = self._safe_value(features['vol_climax_down'])
        vol_divergence = self._safe_value(features['vol_divergence'])
        
        # Calculate signal
        signal_score = 0
        
        # OBV trend contribution
        signal_score += np.clip(obv_trend * 30, -20, 20)
        
        # VWAP deviation (mean reversion)
        if vwap_dev < -0.02:
            signal_score += 15  # Below VWAP = potential buy
        elif vwap_dev > 0.02:
            signal_score -= 15  # Above VWAP = potential sell
        
        # CMF contribution
        signal_score += np.clip(cmf * 40, -20, 20)
        
        # Volume momentum
        signal_score += np.clip(vw_momentum * 50, -15, 15)
        
        # Volume climax (potential reversal)
        if vol_climax_up:
            signal_score -= 10  # Exhaustion after climax up
        if vol_climax_down:
            signal_score += 10  # Exhaustion after climax down
        
        # Volume divergence (warning signal)
        if vol_divergence:
            signal_score *= 0.7  # Reduce confidence
        
        # High volume confirmation
        if vol_ratio > 1.5:
            signal_score *= 1.2  # Increase confidence on high volume
        
        # Determine signal type
        if signal_score > 30:
            signal_type = SignalType.STRONG_BUY
        elif signal_score > 12:
            signal_type = SignalType.BUY
        elif signal_score < -30:
            signal_type = SignalType.STRONG_SELL
        elif signal_score < -12:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Calculate confidence
        confidence = 0.4
        if vol_ratio > 1.2:
            confidence += 0.2
        if abs(cmf) > 0.2:
            confidence += 0.15
        if not vol_divergence:
            confidence += 0.15
        confidence = min(1.0, confidence)
        
        reasoning = self._generate_reasoning(vol_ratio, obv_trend, vwap_dev, 
                                            cmf, vol_climax_up, vol_climax_down, vol_divergence)
        
        signal = AgentSignal(
            signal_type=signal_type,
            confidence=confidence,
            strength=abs(signal_score),
            reasoning=reasoning,
            metrics={
                'vol_ratio': vol_ratio,
                'obv_trend': obv_trend,
                'vwap_dev': vwap_dev,
                'cmf': cmf,
                'vroc': vroc,
                'vw_momentum': vw_momentum,
                'signal_score': signal_score
            },
            timestamp=df.index[-1]
        )
        
        self.update_signal(signal)
        return signal
    
    def _calculate_obv(self, close, volume):
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_ad_line(self, high, low, close, volume):
        """Calculate Accumulation/Distribution Line."""
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume
        return mfv.cumsum()
    
    def _calculate_cmf(self, high, low, close, volume, period):
        """Calculate Chaikin Money Flow."""
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume
        return mfv.rolling(period).sum() / volume.rolling(period).sum()
    
    def _calculate_pvt(self, close, volume):
        """Calculate Price Volume Trend."""
        pct_change = close.pct_change()
        return (pct_change * volume).cumsum()
    
    def _calculate_vw_momentum(self, close, volume, period):
        """Calculate volume-weighted momentum."""
        returns = close.pct_change()
        vw_returns = returns * volume
        vw_sum = vw_returns.rolling(period).sum()
        vol_sum = volume.rolling(period).sum()
        return vw_sum / (vol_sum + 1e-10)
    
    def _generate_reasoning(self, vol_ratio, obv_trend, vwap_dev, cmf, 
                           climax_up, climax_down, divergence):
        """Generate human-readable reasoning."""
        parts = []
        
        if vol_ratio > 1.5:
            parts.append(f"High volume ({vol_ratio:.1f}x average)")
        elif vol_ratio < 0.5:
            parts.append("Low volume - weak conviction")
        
        if obv_trend > 0.1:
            parts.append("OBV trending up (accumulation)")
        elif obv_trend < -0.1:
            parts.append("OBV trending down (distribution)")
        
        if vwap_dev < -0.02:
            parts.append("Trading below VWAP")
        elif vwap_dev > 0.02:
            parts.append("Trading above VWAP")
        
        if cmf > 0.2:
            parts.append("Strong buying pressure (CMF)")
        elif cmf < -0.2:
            parts.append("Strong selling pressure (CMF)")
        
        if climax_up:
            parts.append("Volume climax on up move")
        if climax_down:
            parts.append("Volume climax on down move")
        
        if divergence:
            parts.append("WARNING: Volume-price divergence")
        
        return ". ".join(parts) + "." if parts else "Normal volume conditions."

