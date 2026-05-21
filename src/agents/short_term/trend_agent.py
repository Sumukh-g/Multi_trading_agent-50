"""
Trend Following Agent

Specializes in detecting and following price trends using:
- Multiple moving averages (EMA, SMA, WMA)
- MACD analysis
- ADX trend strength
- Supertrend indicator
- Linear regression channels
"""

import pandas as pd
import numpy as np
from typing import Dict
from .base_agent import BaseAgent, AgentSignal, AgentConfig, SignalType


class TrendAgent(BaseAgent):
    """
    Advanced trend detection and following agent.
    
    Uses a combination of moving averages, trend indicators,
    and regression analysis to identify strong trends.
    """
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(name="TrendAgent", weight=1.2)
        super().__init__(config)
        
        # Default parameters
        self.ema_fast = self.params.get('ema_fast', 8)
        self.ema_medium = self.params.get('ema_medium', 21)
        self.ema_slow = self.params.get('ema_slow', 55)
        self.adx_period = self.params.get('adx_period', 14)
        self.supertrend_period = self.params.get('supertrend_period', 10)
        self.supertrend_mult = self.params.get('supertrend_mult', 3.0)
    
    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract trend-related features."""
        # Handle potential Series vs DataFrame issues
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        high = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
        low = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
        
        features = pd.DataFrame(index=df.index)
        
        # Multiple EMAs
        features['ema_fast'] = close.ewm(span=self.ema_fast, adjust=False).mean()
        features['ema_medium'] = close.ewm(span=self.ema_medium, adjust=False).mean()
        features['ema_slow'] = close.ewm(span=self.ema_slow, adjust=False).mean()
        
        # EMA crossover signals
        features['ema_cross_fast_med'] = features['ema_fast'] - features['ema_medium']
        features['ema_cross_med_slow'] = features['ema_medium'] - features['ema_slow']
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # ADX
        features['adx'] = self._calculate_adx(high, low, close, self.adx_period)
        features['+di'], features['-di'] = self._calculate_di(high, low, close, self.adx_period)
        
        # Supertrend
        features['supertrend'], features['st_direction'] = self._calculate_supertrend(
            high, low, close, self.supertrend_period, self.supertrend_mult
        )
        
        # Linear regression slope
        features['lr_slope'] = self._calculate_lr_slope(close, 20)
        
        # Price position relative to EMAs
        features['price_vs_ema_fast'] = (close - features['ema_fast']) / features['ema_fast']
        features['price_vs_ema_slow'] = (close - features['ema_slow']) / features['ema_slow']
        
        # Trend alignment score
        features['trend_alignment'] = self._calculate_trend_alignment(features)
        
        return features
    
    def analyze(self, df: pd.DataFrame, ticker: str) -> AgentSignal:
        """Analyze trend and generate signal."""
        features = self.get_features(df)
        
        # Get latest values
        adx = self._safe_value(features['adx'])
        plus_di = self._safe_value(features['+di'])
        minus_di = self._safe_value(features['-di'])
        macd_hist = self._safe_value(features['macd_hist'])
        st_direction = self._safe_value(features['st_direction'])
        lr_slope = self._safe_value(features['lr_slope'])
        trend_alignment = self._safe_value(features['trend_alignment'])
        ema_cross = self._safe_value(features['ema_cross_fast_med'])
        
        # Calculate trend score (-100 to +100)
        trend_score = 0
        
        # ADX and DI contribution (strong trend detection)
        if adx > 25:
            if plus_di > minus_di:
                trend_score += min(30, adx - 25 + 10)
            else:
                trend_score -= min(30, adx - 25 + 10)
        
        # MACD contribution
        macd_normalized = np.clip(macd_hist * 1000, -20, 20)
        trend_score += macd_normalized
        
        # Supertrend contribution
        trend_score += st_direction * 15
        
        # Linear regression slope contribution
        lr_contribution = np.clip(lr_slope * 5000, -15, 15)
        trend_score += lr_contribution
        
        # EMA alignment contribution
        trend_score += trend_alignment * 20
        
        # Determine signal type
        if trend_score > 50:
            signal_type = SignalType.STRONG_BUY
        elif trend_score > 20:
            signal_type = SignalType.BUY
        elif trend_score < -50:
            signal_type = SignalType.STRONG_SELL
        elif trend_score < -20:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Calculate confidence based on ADX and alignment
        confidence = min(1.0, (adx / 50) * abs(trend_alignment))
        
        # Generate reasoning
        reasoning = self._generate_reasoning(adx, plus_di, minus_di, 
                                            macd_hist, st_direction, trend_alignment)
        
        signal = AgentSignal(
            signal_type=signal_type,
            confidence=confidence,
            strength=abs(trend_score),
            reasoning=reasoning,
            metrics={
                'adx': adx,
                '+di': plus_di,
                '-di': minus_di,
                'macd_hist': macd_hist,
                'supertrend': st_direction,
                'trend_alignment': trend_alignment,
                'trend_score': trend_score
            },
            timestamp=df.index[-1]
        )
        
        self.update_signal(signal)
        return signal
    
    def _calculate_adx(self, high, low, close, period):
        """Calculate ADX indicator."""
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        atr = tr['tr'].ewm(span=period, adjust=False).mean()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx
    
    def _calculate_di(self, high, low, close, period):
        """Calculate +DI and -DI."""
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        
        atr = tr['tr'].ewm(span=period, adjust=False).mean()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        return plus_di, minus_di
    
    def _calculate_supertrend(self, high, low, close, period, multiplier):
        """Calculate Supertrend indicator."""
        hl2 = (high + low) / 2
        atr = self._calculate_atr(high, low, close, period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)
        
        for i in range(period, len(close)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                if i > 0 and not pd.isna(supertrend.iloc[i-1]):
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    direction.iloc[i] = direction.iloc[i-1]
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
        
        return supertrend, direction
    
    def _calculate_atr(self, high, low, close, period):
        """Calculate Average True Range."""
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return tr['tr'].ewm(span=period, adjust=False).mean()
    
    def _calculate_lr_slope(self, series, period):
        """Calculate linear regression slope."""
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(period, len(series)):
            y = series.iloc[i-period:i].values
            x = np.arange(period)
            slope = np.polyfit(x, y, 1)[0]
            slopes.iloc[i] = slope
        return slopes
    
    def _calculate_trend_alignment(self, features):
        """Calculate how aligned all trend indicators are (-1 to 1)."""
        alignment = pd.Series(index=features.index, dtype=float)
        
        # Count bullish/bearish signals
        for i in range(len(features)):
            bullish = 0
            bearish = 0
            
            if features['ema_cross_fast_med'].iloc[i] > 0:
                bullish += 1
            else:
                bearish += 1
                
            if features['ema_cross_med_slow'].iloc[i] > 0:
                bullish += 1
            else:
                bearish += 1
                
            if features['macd_hist'].iloc[i] > 0:
                bullish += 1
            else:
                bearish += 1
                
            if features['st_direction'].iloc[i] > 0:
                bullish += 1
            else:
                bearish += 1
            
            total = bullish + bearish
            if total > 0:
                alignment.iloc[i] = (bullish - bearish) / total
            else:
                alignment.iloc[i] = 0
        
        return alignment
    
    def _generate_reasoning(self, adx, plus_di, minus_di, macd_hist, st_dir, alignment):
        """Generate human-readable reasoning."""
        parts = []
        
        if adx > 25:
            parts.append(f"Strong trend detected (ADX={adx:.1f})")
            if plus_di > minus_di:
                parts.append("with bullish directional pressure")
            else:
                parts.append("with bearish directional pressure")
        else:
            parts.append(f"Weak/ranging market (ADX={adx:.1f})")
        
        if macd_hist > 0:
            parts.append("MACD positive and expanding")
        else:
            parts.append("MACD negative and contracting")
        
        if st_dir > 0:
            parts.append("Supertrend bullish")
        else:
            parts.append("Supertrend bearish")
        
        if abs(alignment) > 0.75:
            parts.append("Strong indicator alignment")
        elif abs(alignment) < 0.25:
            parts.append("Mixed indicator signals")
        
        return ". ".join(parts) + "."

