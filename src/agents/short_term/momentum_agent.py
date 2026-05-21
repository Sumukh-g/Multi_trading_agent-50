"""
Momentum Agent

Specializes in momentum-based trading using:
- RSI with divergence detection
- Stochastic oscillator
- Rate of Change (ROC)
- Williams %R
- Commodity Channel Index (CCI)
- Money Flow Index (MFI)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .base_agent import BaseAgent, AgentSignal, AgentConfig, SignalType


class MomentumAgent(BaseAgent):
    """
    Advanced momentum detection agent.
    
    Combines multiple momentum oscillators to identify
    overbought/oversold conditions and momentum divergences.
    """
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(name="MomentumAgent", weight=1.0)
        super().__init__(config)
        
        self.rsi_period = self.params.get('rsi_period', 14)
        self.stoch_period = self.params.get('stoch_period', 14)
        self.cci_period = self.params.get('cci_period', 20)
        self.mfi_period = self.params.get('mfi_period', 14)
    
    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract momentum-related features."""
        # Handle potential Series vs DataFrame issues
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        high = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
        low = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
        
        if 'Volume' in df.columns:
            volume = df['Volume'] if isinstance(df['Volume'], pd.Series) else df['Volume'].iloc[:, 0]
        else:
            volume = pd.Series(1, index=df.index)
        
        features = pd.DataFrame(index=df.index)
        
        # RSI
        features['rsi'] = self._calculate_rsi(close, self.rsi_period)
        features['rsi_ma'] = features['rsi'].rolling(5).mean()
        
        # Stochastic
        features['stoch_k'], features['stoch_d'] = self._calculate_stochastic(
            high, low, close, self.stoch_period
        )
        
        # Rate of Change
        features['roc_5'] = close.pct_change(5) * 100
        features['roc_10'] = close.pct_change(10) * 100
        features['roc_20'] = close.pct_change(20) * 100
        
        # Williams %R
        features['williams_r'] = self._calculate_williams_r(high, low, close, 14)
        
        # CCI
        features['cci'] = self._calculate_cci(high, low, close, self.cci_period)
        
        # MFI
        features['mfi'] = self._calculate_mfi(high, low, close, volume, self.mfi_period)
        
        # RSI Divergence detection
        features['rsi_bullish_div'], features['rsi_bearish_div'] = self._detect_divergence(
            close, features['rsi'], 20
        )
        
        # Momentum composite score
        features['momentum_score'] = self._calculate_momentum_score(features)
        
        # Overbought/Oversold composite
        features['ob_os_score'] = self._calculate_ob_os_score(features)
        
        return features
    
    def analyze(self, df: pd.DataFrame, ticker: str) -> AgentSignal:
        """Analyze momentum and generate signal."""
        features = self.get_features(df)
        
        # Get latest values
        rsi = self._safe_value(features['rsi'], 50)
        stoch_k = self._safe_value(features['stoch_k'], 50)
        stoch_d = self._safe_value(features['stoch_d'], 50)
        roc_10 = self._safe_value(features['roc_10'])
        williams_r = self._safe_value(features['williams_r'], -50)
        cci = self._safe_value(features['cci'])
        mfi = self._safe_value(features['mfi'], 50)
        momentum_score = self._safe_value(features['momentum_score'])
        ob_os_score = self._safe_value(features['ob_os_score'])
        rsi_bull_div = self._safe_value(features['rsi_bullish_div'])
        rsi_bear_div = self._safe_value(features['rsi_bearish_div'])
        
        # Calculate composite signal
        signal_score = 0
        
        # RSI contribution
        if rsi < 30:
            signal_score += 20  # Oversold = potential buy
        elif rsi > 70:
            signal_score -= 20  # Overbought = potential sell
        else:
            signal_score += (rsi - 50) * 0.3  # Trending with momentum
        
        # Stochastic contribution
        if stoch_k < 20 and stoch_k > stoch_d:
            signal_score += 15  # Oversold with bullish cross
        elif stoch_k > 80 and stoch_k < stoch_d:
            signal_score -= 15  # Overbought with bearish cross
        
        # ROC contribution
        signal_score += np.clip(roc_10 * 2, -15, 15)
        
        # CCI contribution
        if cci < -100:
            signal_score += 10
        elif cci > 100:
            signal_score -= 10
        
        # MFI contribution
        if mfi < 20:
            signal_score += 10
        elif mfi > 80:
            signal_score -= 10
        
        # Divergence bonus
        if rsi_bull_div > 0:
            signal_score += 25
        if rsi_bear_div > 0:
            signal_score -= 25
        
        # Add composite scores
        signal_score += momentum_score * 15
        
        # Determine signal type
        if signal_score > 40:
            signal_type = SignalType.STRONG_BUY
        elif signal_score > 15:
            signal_type = SignalType.BUY
        elif signal_score < -40:
            signal_type = SignalType.STRONG_SELL
        elif signal_score < -15:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Calculate confidence
        # Higher confidence when multiple indicators agree
        indicator_agreement = self._calculate_indicator_agreement(features)
        confidence = min(1.0, indicator_agreement * 0.8 + abs(signal_score) / 100 * 0.2)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(rsi, stoch_k, roc_10, cci, mfi, 
                                            rsi_bull_div, rsi_bear_div)
        
        signal = AgentSignal(
            signal_type=signal_type,
            confidence=confidence,
            strength=abs(signal_score),
            reasoning=reasoning,
            metrics={
                'rsi': rsi,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'roc_10': roc_10,
                'williams_r': williams_r,
                'cci': cci,
                'mfi': mfi,
                'momentum_score': momentum_score,
                'signal_score': signal_score
            },
            timestamp=df.index[-1]
        )
        
        self.update_signal(signal)
        return signal
    
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_stochastic(self, high, low, close, period) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(3).mean()
        
        return stoch_k, stoch_d
    
    def _calculate_williams_r(self, high, low, close, period) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        return williams_r
    
    def _calculate_cci(self, high, low, close, period) -> pd.Series:
        """Calculate Commodity Channel Index."""
        tp = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        
        cci = (tp - sma) / (0.015 * mad + 1e-10)
        return cci
    
    def _calculate_mfi(self, high, low, close, volume, period) -> pd.Series:
        """Calculate Money Flow Index."""
        tp = (high + low + close) / 3
        mf = tp * volume
        
        pos_mf = pd.Series(0.0, index=close.index)
        neg_mf = pd.Series(0.0, index=close.index)
        
        for i in range(1, len(close)):
            if tp.iloc[i] > tp.iloc[i-1]:
                pos_mf.iloc[i] = mf.iloc[i]
            elif tp.iloc[i] < tp.iloc[i-1]:
                neg_mf.iloc[i] = mf.iloc[i]
        
        pos_mf_sum = pos_mf.rolling(period).sum()
        neg_mf_sum = neg_mf.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + pos_mf_sum / (neg_mf_sum + 1e-10)))
        return mfi
    
    def _detect_divergence(self, price, indicator, lookback) -> Tuple[pd.Series, pd.Series]:
        """Detect bullish and bearish divergences."""
        bullish_div = pd.Series(0, index=price.index)
        bearish_div = pd.Series(0, index=price.index)
        
        for i in range(lookback, len(price)):
            # Get window
            price_window = price.iloc[i-lookback:i+1]
            ind_window = indicator.iloc[i-lookback:i+1]
            
            # Find local minima/maxima
            price_min_idx = price_window.idxmin()
            price_max_idx = price_window.idxmax()
            
            # Bullish divergence: price makes lower low, indicator makes higher low
            if price.iloc[i] < price_window.min() * 1.01:  # Near new low
                if indicator.iloc[i] > ind_window.min():
                    bullish_div.iloc[i] = 1
            
            # Bearish divergence: price makes higher high, indicator makes lower high
            if price.iloc[i] > price_window.max() * 0.99:  # Near new high
                if indicator.iloc[i] < ind_window.max():
                    bearish_div.iloc[i] = 1
        
        return bullish_div, bearish_div
    
    def _calculate_momentum_score(self, features) -> pd.Series:
        """Calculate composite momentum score (-1 to 1)."""
        score = pd.Series(0.0, index=features.index)
        
        # Normalize and combine
        rsi_norm = (features['rsi'] - 50) / 50  # -1 to 1
        stoch_norm = (features['stoch_k'] - 50) / 50
        roc_norm = np.clip(features['roc_10'] / 10, -1, 1)
        
        score = (rsi_norm * 0.4 + stoch_norm * 0.3 + roc_norm * 0.3)
        
        return score
    
    def _calculate_ob_os_score(self, features) -> pd.Series:
        """Calculate overbought/oversold score (-1 to 1)."""
        score = pd.Series(0.0, index=features.index)
        
        # RSI contribution
        rsi = features['rsi']
        score += np.where(rsi > 70, -(rsi - 70) / 30, 0)
        score += np.where(rsi < 30, (30 - rsi) / 30, 0)
        
        # Stochastic contribution
        stoch = features['stoch_k']
        score += np.where(stoch > 80, -(stoch - 80) / 20, 0)
        score += np.where(stoch < 20, (20 - stoch) / 20, 0)
        
        return np.clip(score, -1, 1)
    
    def _calculate_indicator_agreement(self, features) -> float:
        """Calculate how many indicators agree on direction."""
        latest = features.iloc[-1]
        
        bullish = 0
        bearish = 0
        
        if latest['rsi'] > 50:
            bullish += 1
        else:
            bearish += 1
        
        if latest['stoch_k'] > 50:
            bullish += 1
        else:
            bearish += 1
        
        if latest['roc_10'] > 0:
            bullish += 1
        else:
            bearish += 1
        
        if latest['cci'] > 0:
            bullish += 1
        else:
            bearish += 1
        
        if latest['mfi'] > 50:
            bullish += 1
        else:
            bearish += 1
        
        return max(bullish, bearish) / (bullish + bearish)
    
    def _generate_reasoning(self, rsi, stoch_k, roc_10, cci, mfi, bull_div, bear_div):
        """Generate human-readable reasoning."""
        parts = []
        
        if rsi < 30:
            parts.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            parts.append(f"RSI overbought ({rsi:.1f})")
        else:
            parts.append(f"RSI neutral ({rsi:.1f})")
        
        if stoch_k < 20:
            parts.append("Stochastic oversold")
        elif stoch_k > 80:
            parts.append("Stochastic overbought")
        
        if roc_10 > 5:
            parts.append("Strong positive momentum")
        elif roc_10 < -5:
            parts.append("Strong negative momentum")
        
        if bull_div > 0:
            parts.append("BULLISH DIVERGENCE detected")
        if bear_div > 0:
            parts.append("BEARISH DIVERGENCE detected")
        
        return ". ".join(parts) + "."

