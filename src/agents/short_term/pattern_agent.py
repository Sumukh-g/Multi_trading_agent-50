"""
Pattern Recognition Agent

Specializes in detecting chart patterns using:
- Candlestick patterns
- Support/Resistance levels
- Price action patterns
- Chart formations (head & shoulders, triangles, etc.)
- Breakout detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .base_agent import BaseAgent, AgentSignal, AgentConfig, SignalType


class PatternAgent(BaseAgent):
    """
    Advanced pattern recognition agent.
    
    Identifies candlestick patterns, chart formations,
    and key support/resistance levels.
    """
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(name="PatternAgent", weight=1.1)
        super().__init__(config)
        
        self.sr_lookback = self.params.get('sr_lookback', 50)
        self.sr_tolerance = self.params.get('sr_tolerance', 0.02)
    
    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract pattern-related features."""
        # Handle potential Series vs DataFrame issues
        open_p = df['Open'] if isinstance(df['Open'], pd.Series) else df['Open'].iloc[:, 0]
        high = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
        low = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        
        features = pd.DataFrame(index=df.index)
        
        # Candlestick body and shadow ratios
        body = abs(close - open_p)
        upper_shadow = high - pd.concat([close, open_p], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_p], axis=1).min(axis=1) - low
        total_range = high - low
        
        features['body_ratio'] = body / (total_range + 1e-10)
        features['upper_shadow_ratio'] = upper_shadow / (total_range + 1e-10)
        features['lower_shadow_ratio'] = lower_shadow / (total_range + 1e-10)
        features['is_bullish'] = (close > open_p).astype(int)
        
        # Candlestick patterns
        features['doji'] = self._detect_doji(open_p, high, low, close)
        features['hammer'] = self._detect_hammer(open_p, high, low, close)
        features['engulfing'] = self._detect_engulfing(open_p, high, low, close)
        features['morning_star'] = self._detect_morning_star(open_p, high, low, close)
        features['evening_star'] = self._detect_evening_star(open_p, high, low, close)
        
        # Support/Resistance levels
        features['near_support'], features['near_resistance'] = self._detect_sr_proximity(
            high, low, close, self.sr_lookback, self.sr_tolerance
        )
        
        # Higher highs/Lower lows
        features['higher_high'] = (high > high.rolling(10).max().shift(1)).astype(int)
        features['lower_low'] = (low < low.rolling(10).min().shift(1)).astype(int)
        features['higher_low'] = (low > low.shift(1)).astype(int).rolling(3).sum()
        features['lower_high'] = (high < high.shift(1)).astype(int).rolling(3).sum()
        
        # Breakout detection
        features['breakout_up'] = (close > high.rolling(20).max().shift(1)).astype(int)
        features['breakout_down'] = (close < low.rolling(20).min().shift(1)).astype(int)
        
        # Inside bar pattern
        features['inside_bar'] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
        
        # Pin bar detection
        features['bullish_pin'] = self._detect_bullish_pin(open_p, high, low, close)
        features['bearish_pin'] = self._detect_bearish_pin(open_p, high, low, close)
        
        # Trend structure score
        features['trend_structure'] = self._calculate_trend_structure(high, low, close)
        
        return features
    
    def analyze(self, df: pd.DataFrame, ticker: str) -> AgentSignal:
        """Analyze patterns and generate signal."""
        features = self.get_features(df)
        
        # Get latest and recent values
        doji = self._safe_value(features['doji'])
        hammer = self._safe_value(features['hammer'])
        engulfing = self._safe_value(features['engulfing'])
        morning_star = self._safe_value(features['morning_star'])
        evening_star = self._safe_value(features['evening_star'])
        near_support = self._safe_value(features['near_support'])
        near_resistance = self._safe_value(features['near_resistance'])
        breakout_up = self._safe_value(features['breakout_up'])
        breakout_down = self._safe_value(features['breakout_down'])
        bullish_pin = self._safe_value(features['bullish_pin'])
        bearish_pin = self._safe_value(features['bearish_pin'])
        trend_structure = self._safe_value(features['trend_structure'])
        inside_bar = self._safe_value(features['inside_bar'])
        
        # Calculate signal score
        signal_score = 0
        patterns_found = []
        
        # Candlestick patterns
        if hammer > 0:
            signal_score += 15
            patterns_found.append("Hammer")
        
        if engulfing > 0:
            signal_score += engulfing * 20
            patterns_found.append("Bullish Engulfing" if engulfing > 0 else "Bearish Engulfing")
        
        if morning_star > 0:
            signal_score += 25
            patterns_found.append("Morning Star")
        
        if evening_star > 0:
            signal_score -= 25
            patterns_found.append("Evening Star")
        
        if bullish_pin > 0:
            signal_score += 15
            patterns_found.append("Bullish Pin Bar")
        
        if bearish_pin > 0:
            signal_score -= 15
            patterns_found.append("Bearish Pin Bar")
        
        # Support/Resistance
        if near_support > 0:
            signal_score += 10  # Near support = potential bounce
            patterns_found.append("Near Support")
        
        if near_resistance > 0:
            signal_score -= 10  # Near resistance = potential rejection
            patterns_found.append("Near Resistance")
        
        # Breakouts
        if breakout_up > 0:
            signal_score += 30
            patterns_found.append("Breakout UP")
        
        if breakout_down > 0:
            signal_score -= 30
            patterns_found.append("Breakout DOWN")
        
        # Trend structure
        signal_score += trend_structure * 15
        
        # Inside bar (consolidation)
        if inside_bar:
            patterns_found.append("Inside Bar (consolidation)")
        
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
        
        # Calculate confidence
        pattern_count = len(patterns_found)
        confidence = min(1.0, 0.3 + pattern_count * 0.15)
        
        # Boost confidence for breakouts
        if breakout_up or breakout_down:
            confidence = min(1.0, confidence + 0.2)
        
        reasoning = self._generate_reasoning(patterns_found, trend_structure)
        
        signal = AgentSignal(
            signal_type=signal_type,
            confidence=confidence,
            strength=abs(signal_score),
            reasoning=reasoning,
            metrics={
                'patterns_found': len(patterns_found),
                'near_support': near_support,
                'near_resistance': near_resistance,
                'breakout_up': breakout_up,
                'breakout_down': breakout_down,
                'trend_structure': trend_structure,
                'signal_score': signal_score
            },
            timestamp=df.index[-1]
        )
        
        self.update_signal(signal)
        return signal
    
    def _detect_doji(self, open_p, high, low, close):
        """Detect Doji patterns."""
        body = abs(close - open_p)
        total_range = high - low
        return (body / (total_range + 1e-10) < 0.1).astype(int)
    
    def _detect_hammer(self, open_p, high, low, close):
        """Detect Hammer/Hanging Man patterns."""
        body = abs(close - open_p)
        lower_shadow = pd.concat([close, open_p], axis=1).min(axis=1) - low
        upper_shadow = high - pd.concat([close, open_p], axis=1).max(axis=1)
        
        hammer = ((lower_shadow > 2 * body) & 
                  (upper_shadow < body * 0.5) &
                  (body > 0)).astype(int)
        return hammer
    
    def _detect_engulfing(self, open_p, high, low, close):
        """Detect Engulfing patterns. Returns 1 for bullish, -1 for bearish."""
        result = pd.Series(0, index=close.index)
        
        for i in range(1, len(close)):
            prev_body = close.iloc[i-1] - open_p.iloc[i-1]
            curr_body = close.iloc[i] - open_p.iloc[i]
            
            # Bullish engulfing
            if (prev_body < 0 and curr_body > 0 and
                open_p.iloc[i] < close.iloc[i-1] and
                close.iloc[i] > open_p.iloc[i-1]):
                result.iloc[i] = 1
            
            # Bearish engulfing
            elif (prev_body > 0 and curr_body < 0 and
                  open_p.iloc[i] > close.iloc[i-1] and
                  close.iloc[i] < open_p.iloc[i-1]):
                result.iloc[i] = -1
        
        return result
    
    def _detect_morning_star(self, open_p, high, low, close):
        """Detect Morning Star pattern."""
        result = pd.Series(0, index=close.index)
        
        for i in range(2, len(close)):
            # First candle: bearish
            first_bearish = close.iloc[i-2] < open_p.iloc[i-2]
            # Second candle: small body (doji-like)
            second_small = abs(close.iloc[i-1] - open_p.iloc[i-1]) < (high.iloc[i-1] - low.iloc[i-1]) * 0.3
            # Third candle: bullish, closes above midpoint of first
            third_bullish = close.iloc[i] > open_p.iloc[i]
            closes_above = close.iloc[i] > (open_p.iloc[i-2] + close.iloc[i-2]) / 2
            
            if first_bearish and second_small and third_bullish and closes_above:
                result.iloc[i] = 1
        
        return result
    
    def _detect_evening_star(self, open_p, high, low, close):
        """Detect Evening Star pattern."""
        result = pd.Series(0, index=close.index)
        
        for i in range(2, len(close)):
            # First candle: bullish
            first_bullish = close.iloc[i-2] > open_p.iloc[i-2]
            # Second candle: small body
            second_small = abs(close.iloc[i-1] - open_p.iloc[i-1]) < (high.iloc[i-1] - low.iloc[i-1]) * 0.3
            # Third candle: bearish, closes below midpoint of first
            third_bearish = close.iloc[i] < open_p.iloc[i]
            closes_below = close.iloc[i] < (open_p.iloc[i-2] + close.iloc[i-2]) / 2
            
            if first_bullish and second_small and third_bearish and closes_below:
                result.iloc[i] = 1
        
        return result
    
    def _detect_bullish_pin(self, open_p, high, low, close):
        """Detect bullish pin bar."""
        body = abs(close - open_p)
        lower_shadow = pd.concat([close, open_p], axis=1).min(axis=1) - low
        upper_shadow = high - pd.concat([close, open_p], axis=1).max(axis=1)
        
        return ((lower_shadow > 2.5 * body) & 
                (upper_shadow < body) &
                (close > open_p)).astype(int)
    
    def _detect_bearish_pin(self, open_p, high, low, close):
        """Detect bearish pin bar."""
        body = abs(close - open_p)
        lower_shadow = pd.concat([close, open_p], axis=1).min(axis=1) - low
        upper_shadow = high - pd.concat([close, open_p], axis=1).max(axis=1)
        
        return ((upper_shadow > 2.5 * body) & 
                (lower_shadow < body) &
                (close < open_p)).astype(int)
    
    def _detect_sr_proximity(self, high, low, close, lookback, tolerance):
        """Detect proximity to support/resistance levels."""
        near_support = pd.Series(0, index=close.index)
        near_resistance = pd.Series(0, index=close.index)
        
        for i in range(lookback, len(close)):
            window_low = low.iloc[i-lookback:i].min()
            window_high = high.iloc[i-lookback:i].max()
            
            # Near support
            if close.iloc[i] < window_low * (1 + tolerance):
                near_support.iloc[i] = 1
            
            # Near resistance  
            if close.iloc[i] > window_high * (1 - tolerance):
                near_resistance.iloc[i] = 1
        
        return near_support, near_resistance
    
    def _calculate_trend_structure(self, high, low, close):
        """Calculate trend structure score based on higher highs/lows."""
        score = pd.Series(0.0, index=close.index)
        
        for i in range(10, len(close)):
            hh = (high.iloc[i] > high.iloc[i-10:i].max())
            hl = (low.iloc[i] > low.iloc[i-10:i-5].min())
            ll = (low.iloc[i] < low.iloc[i-10:i].min())
            lh = (high.iloc[i] < high.iloc[i-10:i-5].max())
            
            if hh and hl:
                score.iloc[i] = 1  # Uptrend
            elif ll and lh:
                score.iloc[i] = -1  # Downtrend
        
        return score
    
    def _generate_reasoning(self, patterns_found, trend_structure):
        """Generate human-readable reasoning."""
        if not patterns_found:
            return "No significant patterns detected."
        
        parts = [f"Patterns detected: {', '.join(patterns_found)}"]
        
        if trend_structure > 0.5:
            parts.append("Uptrend structure intact")
        elif trend_structure < -0.5:
            parts.append("Downtrend structure intact")
        else:
            parts.append("Neutral/ranging structure")
        
        return ". ".join(parts) + "."

