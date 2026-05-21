"""
Advanced Weighted Indicator Agent

Implements sophisticated indicator weighting system with:
- 5 Agent categories (Trend, Momentum, Volatility, Volume, Pattern)
- 18 individual indicators with research-backed weights
- Regime-based dynamic weight adjustment
- Sector-specific multipliers
- Meta-labeling for signal filtering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .base_agent import BaseAgent, AgentSignal, AgentConfig, SignalType


class MarketRegime(Enum):
    """Market regime types for dynamic weight adjustment."""
    STRONG_TREND = "strong_trend"
    MEAN_REVERTING = "mean_reverting"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    ILLIQUID_NOISY = "illiquid_noisy"
    NORMAL = "normal"


class SectorType(Enum):
    """Sector classifications for weight adjustments."""
    TECH_GROWTH = "tech_growth"
    DEFENSIVE = "defensive"
    COMMODITY_ENERGY = "commodity_energy"
    SMALL_CAP_BIOTECH = "small_cap_biotech"
    GENERAL = "general"


@dataclass
class IndicatorSignal:
    """Individual indicator signal with weight and explanation."""
    name: str
    agent_category: str
    raw_value: float
    normalized_score: float  # -1 to 1
    weight: float
    contribution: float  # normalized_score * weight
    explanation: str
    is_bullish: bool
    confidence: float


@dataclass
class AgentContribution:
    """Contribution from each agent category."""
    name: str
    weight: float
    raw_score: float
    adjusted_score: float
    indicators: List[IndicatorSignal]
    signal: str  # BUY/SELL/NEUTRAL
    confidence: float
    explanation: str


@dataclass
class AdvancedSignal:
    """Comprehensive signal from the advanced weighted system."""
    ticker: str
    signal_type: SignalType
    confidence: float
    strength: float
    expected_return: float
    
    # Market context
    regime: MarketRegime
    sector: SectorType
    
    # Agent breakdown
    agent_contributions: Dict[str, AgentContribution]
    indicator_signals: List[IndicatorSignal]
    
    # Trade explanation
    trade_thesis: str
    why_this_trade: str
    key_factors: List[str]
    risk_factors: List[str]
    
    # Weights used
    effective_weights: Dict[str, float]
    regime_multipliers: Dict[str, float]
    sector_multipliers: Dict[str, float]
    
    timestamp: datetime


class AdvancedWeightedAgent(BaseAgent):
    """
    Advanced multi-indicator agent with dynamic weighting.
    
    Implements research-backed indicator weights:
    - Trend Agent (23%): EMA, MACD, ADX, Supertrend
    - Momentum Agent (19%): RSI, Stochastic, ROC, Divergences
    - Volatility Agent (19%): Bollinger, Keltner, Squeeze
    - Volume Agent (17%): OBV, VWAP, CMF, Volume Climax
    - Pattern Agent (21%): Candlesticks, S/R, Breakouts
    """
    
    # Base agent weights (sum to 1.0)
    AGENT_WEIGHTS = {
        "trend": 0.23,
        "momentum": 0.19,
        "volatility": 0.19,
        "volume": 0.17,
        "pattern": 0.21
    }
    
    # Internal indicator weights (sum to 1.0 within each agent)
    TREND_WEIGHTS = {
        "ema": 0.30,
        "macd": 0.30,
        "adx": 0.25,
        "supertrend": 0.15
    }
    
    MOMENTUM_WEIGHTS = {
        "rsi": 0.35,
        "stochastic": 0.25,
        "roc": 0.20,
        "divergences": 0.20
    }
    
    VOLATILITY_WEIGHTS = {
        "bollinger": 0.35,
        "keltner": 0.25,
        "squeeze": 0.40
    }
    
    VOLUME_WEIGHTS = {
        "obv": 0.25,
        "vwap": 0.30,
        "cmf": 0.25,
        "volume_climax": 0.20
    }
    
    PATTERN_WEIGHTS = {
        "candlesticks": 0.30,
        "support_resistance": 0.35,
        "breakouts": 0.35
    }
    
    # Global indicator weights (pre-calculated)
    GLOBAL_WEIGHTS = {
        "ema": 0.069, "macd": 0.069, "adx": 0.058, "supertrend": 0.035,
        "rsi": 0.067, "stochastic": 0.048, "roc": 0.038, "divergences": 0.038,
        "bollinger": 0.067, "keltner": 0.048, "squeeze": 0.077,
        "obv": 0.043, "vwap": 0.052, "cmf": 0.043, "volume_climax": 0.035,
        "candlesticks": 0.063, "support_resistance": 0.074, "breakouts": 0.074
    }
    
    # Regime multipliers
    REGIME_MULTIPLIERS = {
        MarketRegime.STRONG_TREND: {
            "trend": 1.4, "momentum": 1.2, "pattern": 1.2,
            "volatility": 0.8, "volume": 0.9
        },
        MarketRegime.MEAN_REVERTING: {
            "volatility": 1.3, "momentum": 1.2, "pattern": 1.1,
            "trend": 0.6, "volume": 0.8
        },
        MarketRegime.VOLATILITY_BREAKOUT: {
            "volatility": 1.4, "pattern": 1.3, "volume": 1.2,
            "trend": 1.0, "momentum": 0.9
        },
        MarketRegime.ILLIQUID_NOISY: {
            "pattern": 1.2, "volatility": 1.1, "momentum": 1.1,
            "trend": 1.0, "volume": 0.6
        },
        MarketRegime.NORMAL: {
            "trend": 1.0, "momentum": 1.0, "volatility": 1.0,
            "volume": 1.0, "pattern": 1.0
        }
    }
    
    # Sector multipliers
    SECTOR_MULTIPLIERS = {
        SectorType.TECH_GROWTH: {
            "trend": 1.25, "momentum": 1.20, "pattern": 1.10,
            "volatility": 0.9, "volume": 1.05
        },
        SectorType.DEFENSIVE: {
            "volatility": 1.3, "momentum": 1.2, "pattern": 1.0,
            "trend": 0.8, "volume": 0.9
        },
        SectorType.COMMODITY_ENERGY: {
            "trend": 1.3, "volatility": 1.2, "volume": 1.2,
            "pattern": 1.0, "momentum": 0.9
        },
        SectorType.SMALL_CAP_BIOTECH: {
            "volume": 1.4, "pattern": 1.3, "momentum": 1.1,
            "trend": 0.8, "volatility": 1.0
        },
        SectorType.GENERAL: {
            "trend": 1.0, "momentum": 1.0, "volatility": 1.0,
            "volume": 1.0, "pattern": 1.0
        }
    }
    
    # Sector classification by ticker
    TICKER_SECTORS = {
        # Tech/Growth
        "AAPL": SectorType.TECH_GROWTH, "MSFT": SectorType.TECH_GROWTH,
        "GOOGL": SectorType.TECH_GROWTH, "AMZN": SectorType.TECH_GROWTH,
        "META": SectorType.TECH_GROWTH, "NVDA": SectorType.TECH_GROWTH,
        "TSLA": SectorType.TECH_GROWTH, "CSCO": SectorType.TECH_GROWTH,
        
        # Defensive
        "JNJ": SectorType.DEFENSIVE, "PG": SectorType.DEFENSIVE,
        "WMT": SectorType.DEFENSIVE, "UNH": SectorType.DEFENSIVE,
        
        # Financial
        "JPM": SectorType.GENERAL, "BAC": SectorType.GENERAL,
        "V": SectorType.GENERAL, "MA": SectorType.GENERAL,
        
        # Energy/Commodity
        "XOM": SectorType.COMMODITY_ENERGY, "CVX": SectorType.COMMODITY_ENERGY,
        
        # Consumer
        "HD": SectorType.GENERAL, "DIS": SectorType.GENERAL
    }
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(name="AdvancedWeightedAgent", weight=1.0)
        super().__init__(config)
        
        # Technical indicator parameters
        self.ema_fast = 8
        self.ema_medium = 21
        self.ema_slow = 55
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2.0
        self.atr_period = 14
    
    def analyze(self, df: pd.DataFrame, ticker: str) -> AgentSignal:
        """Run comprehensive analysis and generate signal."""
        # Get comprehensive signal
        adv_signal = self.get_advanced_signal(df, ticker)
        
        # Convert to standard AgentSignal
        return AgentSignal(
            signal_type=adv_signal.signal_type,
            confidence=adv_signal.confidence,
            strength=adv_signal.strength,
            reasoning=adv_signal.trade_thesis,
            metrics={
                "regime": adv_signal.regime.value,
                "sector": adv_signal.sector.value,
                "expected_return": adv_signal.expected_return,
                "num_bullish_indicators": sum(1 for i in adv_signal.indicator_signals if i.is_bullish),
                "num_bearish_indicators": sum(1 for i in adv_signal.indicator_signals if not i.is_bullish)
            },
            timestamp=df.index[-1]
        )
    
    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all indicator features."""
        close = self._get_series(df, 'Close')
        high = self._get_series(df, 'High')
        low = self._get_series(df, 'Low')
        open_p = self._get_series(df, 'Open')
        volume = self._get_series(df, 'Volume') if 'Volume' in df.columns else pd.Series(1e6, index=df.index)
        
        features = pd.DataFrame(index=df.index)
        
        # === TREND INDICATORS ===
        # EMA
        features['ema_fast'] = close.ewm(span=self.ema_fast, adjust=False).mean()
        features['ema_medium'] = close.ewm(span=self.ema_medium, adjust=False).mean()
        features['ema_slow'] = close.ewm(span=self.ema_slow, adjust=False).mean()
        features['ema_alignment'] = self._calc_ema_alignment(features, close)
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # ADX
        features['adx'], features['plus_di'], features['minus_di'] = self._calc_adx(high, low, close)
        
        # Supertrend
        features['supertrend'], features['st_direction'] = self._calc_supertrend(high, low, close)
        
        # === MOMENTUM INDICATORS ===
        # RSI
        features['rsi'] = self._calc_rsi(close, self.rsi_period)
        
        # Stochastic
        features['stoch_k'], features['stoch_d'] = self._calc_stochastic(high, low, close)
        
        # ROC
        features['roc'] = close.pct_change(10) * 100
        
        # Divergences
        features['rsi_bullish_div'], features['rsi_bearish_div'] = self._detect_divergences(close, features['rsi'])
        
        # === VOLATILITY INDICATORS ===
        # Bollinger Bands
        bb_mid = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        features['bb_upper'] = bb_mid + (self.bb_std * bb_std)
        features['bb_lower'] = bb_mid - (self.bb_std * bb_std)
        features['bb_pct'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_mid
        
        # Keltner Channels
        atr = self._calc_atr(high, low, close)
        kc_mid = close.ewm(span=20, adjust=False).mean()
        features['kc_upper'] = kc_mid + (1.5 * atr)
        features['kc_lower'] = kc_mid - (1.5 * atr)
        
        # Squeeze
        features['squeeze'] = ((features['bb_lower'] > features['kc_lower']) & 
                               (features['bb_upper'] < features['kc_upper'])).astype(int)
        features['squeeze_release'] = (features['squeeze'].diff() == -1).astype(int)
        
        # === VOLUME INDICATORS ===
        # OBV
        features['obv'] = self._calc_obv(close, volume)
        features['obv_trend'] = (features['obv'] - features['obv'].rolling(20).mean()) / (features['obv'].rolling(20).std() + 1e-10)
        
        # VWAP
        typical_price = (high + low + close) / 3
        features['vwap'] = (typical_price * volume).rolling(40).sum() / (volume.rolling(40).sum() + 1e-10)
        features['vwap_dev'] = (close - features['vwap']) / features['vwap']
        
        # CMF
        features['cmf'] = self._calc_cmf(high, low, close, volume)
        
        # Volume Climax
        vol_sma = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (vol_sma + 1e-10)
        features['vol_climax_up'] = ((volume > vol_sma * 2) & (close > close.shift(1))).astype(int)
        features['vol_climax_down'] = ((volume > vol_sma * 2) & (close < close.shift(1))).astype(int)
        
        # === PATTERN INDICATORS ===
        # Candlestick patterns
        features['engulfing'] = self._detect_engulfing(open_p, close)
        features['hammer'] = self._detect_hammer(open_p, high, low, close)
        features['doji'] = (abs(close - open_p) / (high - low + 1e-10) < 0.1).astype(int)
        
        # Support/Resistance
        features['near_support'], features['near_resistance'] = self._detect_sr_levels(high, low, close)
        
        # Breakouts
        features['breakout_up'] = (close > high.rolling(20).max().shift(1)).astype(int)
        features['breakout_down'] = (close < low.rolling(20).min().shift(1)).astype(int)
        
        # Higher highs/lows structure
        features['trend_structure'] = self._calc_trend_structure(high, low)
        
        return features
    
    def get_advanced_signal(self, df: pd.DataFrame, ticker: str) -> AdvancedSignal:
        """Generate comprehensive signal with full analysis."""
        features = self.get_features(df)
        close = self._get_series(df, 'Close')
        current_price = float(close.iloc[-1])
        
        # Detect regime
        regime = self._detect_regime(features)
        
        # Get sector
        sector = self.TICKER_SECTORS.get(ticker, SectorType.GENERAL)
        
        # Calculate effective weights
        effective_weights, regime_mults, sector_mults = self._calc_effective_weights(regime, sector)
        
        # Analyze each indicator and agent
        indicator_signals = self._analyze_all_indicators(features, effective_weights)
        agent_contributions = self._aggregate_agent_contributions(indicator_signals, effective_weights)
        
        # Calculate final signal
        total_score = sum(ac.adjusted_score for ac in agent_contributions.values())
        total_weight = sum(effective_weights.values())
        normalized_score = total_score / total_weight if total_weight > 0 else 0
        
        # Determine signal type
        if normalized_score > 0.4:
            signal_type = SignalType.STRONG_BUY
        elif normalized_score > 0.15:
            signal_type = SignalType.BUY
        elif normalized_score < -0.4:
            signal_type = SignalType.STRONG_SELL
        elif normalized_score < -0.15:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.NEUTRAL
        
        # Calculate confidence
        bullish_count = sum(1 for i in indicator_signals if i.is_bullish)
        bearish_count = len(indicator_signals) - bullish_count
        agreement = max(bullish_count, bearish_count) / len(indicator_signals) if indicator_signals else 0.5
        confidence = min(1.0, agreement * 0.7 + abs(normalized_score) * 0.3)
        
        # Expected return
        expected_return = normalized_score * 0.05 * confidence
        
        # Generate trade explanation
        trade_thesis, why_trade, key_factors, risk_factors = self._generate_trade_explanation(
            ticker, signal_type, regime, sector, agent_contributions, indicator_signals
        )
        
        return AdvancedSignal(
            ticker=ticker,
            signal_type=signal_type,
            confidence=confidence,
            strength=abs(normalized_score) * 100,
            expected_return=expected_return,
            regime=regime,
            sector=sector,
            agent_contributions=agent_contributions,
            indicator_signals=indicator_signals,
            trade_thesis=trade_thesis,
            why_this_trade=why_trade,
            key_factors=key_factors,
            risk_factors=risk_factors,
            effective_weights=effective_weights,
            regime_multipliers=regime_mults,
            sector_multipliers=sector_mults,
            timestamp=datetime.now()
        )
    
    def _detect_regime(self, features: pd.DataFrame) -> MarketRegime:
        """Detect current market regime."""
        adx = self._safe_value(features['adx'], 20)
        bb_width = self._safe_value(features['bb_width'], 0.05)
        squeeze = self._safe_value(features['squeeze'], 0)
        squeeze_release = self._safe_value(features['squeeze_release'], 0)
        volume_ratio = self._safe_value(features['volume_ratio'], 1)
        
        # Strong trend regime
        if adx > 25 and abs(self._safe_value(features['ema_alignment'], 0)) > 0.7:
            return MarketRegime.STRONG_TREND
        
        # Volatility breakout (squeeze release with volume)
        if squeeze_release > 0 and volume_ratio > 1.5:
            return MarketRegime.VOLATILITY_BREAKOUT
        
        # Mean reverting (low ADX, narrow bands, oscillating around VWAP)
        if adx < 20 and bb_width < 0.04:
            return MarketRegime.MEAN_REVERTING
        
        # Illiquid/noisy (very low volume)
        if volume_ratio < 0.5:
            return MarketRegime.ILLIQUID_NOISY
        
        return MarketRegime.NORMAL
    
    def _calc_effective_weights(self, regime: MarketRegime, sector: SectorType) -> Tuple[Dict, Dict, Dict]:
        """Calculate effective weights after regime and sector adjustments."""
        regime_mults = self.REGIME_MULTIPLIERS[regime]
        sector_mults = self.SECTOR_MULTIPLIERS[sector]
        
        effective = {}
        for agent, base_weight in self.AGENT_WEIGHTS.items():
            regime_mult = regime_mults.get(agent, 1.0)
            sector_mult = sector_mults.get(agent, 1.0)
            effective[agent] = base_weight * regime_mult * sector_mult
        
        # Normalize to sum to 1
        total = sum(effective.values())
        effective = {k: v / total for k, v in effective.items()}
        
        return effective, regime_mults, sector_mults
    
    def _analyze_all_indicators(self, features: pd.DataFrame, weights: Dict) -> List[IndicatorSignal]:
        """Analyze all 18 indicators and generate signals."""
        signals = []
        
        # === TREND AGENT ===
        trend_weight = weights.get('trend', 0.23)
        
        # EMA
        ema_align = self._safe_value(features['ema_alignment'], 0)
        signals.append(IndicatorSignal(
            name="EMA",
            agent_category="trend",
            raw_value=float(ema_align),
            normalized_score=float(np.clip(ema_align, -1, 1)),
            weight=float(self.TREND_WEIGHTS['ema'] * trend_weight),
            contribution=float(np.clip(ema_align, -1, 1) * self.TREND_WEIGHTS['ema'] * trend_weight),
            explanation=self._explain_ema(ema_align),
            is_bullish=bool(ema_align > 0),
            confidence=float(abs(ema_align))
        ))
        
        # MACD
        macd_hist = self._safe_value(features['macd_hist'], 0)
        macd_score = float(np.clip(macd_hist * 50, -1, 1))
        signals.append(IndicatorSignal(
            name="MACD",
            agent_category="trend",
            raw_value=float(macd_hist),
            normalized_score=macd_score,
            weight=float(self.TREND_WEIGHTS['macd'] * trend_weight),
            contribution=float(macd_score * self.TREND_WEIGHTS['macd'] * trend_weight),
            explanation=self._explain_macd(macd_hist, features),
            is_bullish=bool(macd_hist > 0),
            confidence=float(min(1.0, abs(macd_hist) * 20))
        ))
        
        # ADX
        adx = self._safe_value(features['adx'], 20)
        plus_di = self._safe_value(features['plus_di'], 25)
        minus_di = self._safe_value(features['minus_di'], 25)
        adx_direction = 1 if plus_di > minus_di else -1
        adx_score = float(adx_direction * min(1, (adx - 20) / 30) if adx > 20 else 0)
        signals.append(IndicatorSignal(
            name="ADX",
            agent_category="trend",
            raw_value=float(adx),
            normalized_score=adx_score,
            weight=float(self.TREND_WEIGHTS['adx'] * trend_weight),
            contribution=float(adx_score * self.TREND_WEIGHTS['adx'] * trend_weight),
            explanation=self._explain_adx(adx, plus_di, minus_di),
            is_bullish=bool(plus_di > minus_di),
            confidence=float(min(1.0, adx / 50))
        ))
        
        # Supertrend
        st_dir = self._safe_value(features['st_direction'], 0)
        signals.append(IndicatorSignal(
            name="Supertrend",
            agent_category="trend",
            raw_value=float(st_dir),
            normalized_score=float(st_dir),
            weight=float(self.TREND_WEIGHTS['supertrend'] * trend_weight),
            contribution=float(st_dir * self.TREND_WEIGHTS['supertrend'] * trend_weight),
            explanation="Supertrend " + ("BULLISH ↑" if st_dir > 0 else "BEARISH ↓" if st_dir < 0 else "NEUTRAL"),
            is_bullish=bool(st_dir > 0),
            confidence=float(abs(st_dir))
        ))
        
        # === MOMENTUM AGENT ===
        mom_weight = weights.get('momentum', 0.19)
        
        # RSI
        rsi = self._safe_value(features['rsi'], 50)
        rsi_score = float(self._score_rsi(rsi))
        signals.append(IndicatorSignal(
            name="RSI",
            agent_category="momentum",
            raw_value=float(rsi),
            normalized_score=rsi_score,
            weight=float(self.MOMENTUM_WEIGHTS['rsi'] * mom_weight),
            contribution=float(rsi_score * self.MOMENTUM_WEIGHTS['rsi'] * mom_weight),
            explanation=self._explain_rsi(rsi),
            is_bullish=bool(rsi < 50 or rsi < 30),  # Oversold is bullish opportunity
            confidence=float(abs(rsi - 50) / 50)
        ))
        
        # Stochastic
        stoch_k = self._safe_value(features['stoch_k'], 50)
        stoch_d = self._safe_value(features['stoch_d'], 50)
        stoch_score = float(self._score_stochastic(stoch_k, stoch_d))
        signals.append(IndicatorSignal(
            name="Stochastic",
            agent_category="momentum",
            raw_value=float(stoch_k),
            normalized_score=stoch_score,
            weight=float(self.MOMENTUM_WEIGHTS['stochastic'] * mom_weight),
            contribution=float(stoch_score * self.MOMENTUM_WEIGHTS['stochastic'] * mom_weight),
            explanation=self._explain_stochastic(stoch_k, stoch_d),
            is_bullish=bool(stoch_score > 0),
            confidence=float(abs(stoch_score))
        ))
        
        # ROC
        roc = self._safe_value(features['roc'], 0)
        roc_score = float(np.clip(roc / 10, -1, 1))
        signals.append(IndicatorSignal(
            name="ROC",
            agent_category="momentum",
            raw_value=float(roc),
            normalized_score=roc_score,
            weight=float(self.MOMENTUM_WEIGHTS['roc'] * mom_weight),
            contribution=float(roc_score * self.MOMENTUM_WEIGHTS['roc'] * mom_weight),
            explanation=f"Rate of Change: {roc:+.1f}% ({'accelerating' if roc > 3 else 'decelerating' if roc < -3 else 'stable'})",
            is_bullish=bool(roc > 0),
            confidence=float(min(1.0, abs(roc) / 10))
        ))
        
        # Divergences
        bull_div = self._safe_value(features['rsi_bullish_div'], 0)
        bear_div = self._safe_value(features['rsi_bearish_div'], 0)
        div_score = float(bull_div - bear_div)
        signals.append(IndicatorSignal(
            name="Divergences",
            agent_category="momentum",
            raw_value=div_score,
            normalized_score=div_score,
            weight=float(self.MOMENTUM_WEIGHTS['divergences'] * mom_weight),
            contribution=float(div_score * self.MOMENTUM_WEIGHTS['divergences'] * mom_weight),
            explanation=self._explain_divergences(bull_div, bear_div),
            is_bullish=bool(bull_div > bear_div),
            confidence=float(abs(div_score))
        ))
        
        # === VOLATILITY AGENT ===
        vol_weight = weights.get('volatility', 0.19)
        
        # Bollinger Bands
        bb_pct = self._safe_value(features['bb_pct'], 0.5)
        bb_score = float(self._score_bollinger(bb_pct))
        signals.append(IndicatorSignal(
            name="Bollinger Bands",
            agent_category="volatility",
            raw_value=float(bb_pct),
            normalized_score=bb_score,
            weight=float(self.VOLATILITY_WEIGHTS['bollinger'] * vol_weight),
            contribution=float(bb_score * self.VOLATILITY_WEIGHTS['bollinger'] * vol_weight),
            explanation=self._explain_bollinger(bb_pct),
            is_bullish=bool(bb_pct < 0.3),
            confidence=float(abs(bb_pct - 0.5) * 2)
        ))
        
        # Keltner
        # Use price position relative to Keltner similar to BB
        kc_upper = self._safe_value(features.get('kc_upper', features['bb_upper']), 0)
        kc_lower = self._safe_value(features.get('kc_lower', features['bb_lower']), 0)
        close_val = self._safe_value(features.get('ema_fast', kc_lower), 0)
        if kc_upper > kc_lower:
            kc_pct = float((close_val - kc_lower) / (kc_upper - kc_lower))
        else:
            kc_pct = 0.5
        kc_score = float(-1 * (kc_pct - 0.5) * 2)  # Inverted for mean reversion
        signals.append(IndicatorSignal(
            name="Keltner",
            agent_category="volatility",
            raw_value=float(kc_pct),
            normalized_score=float(np.clip(kc_score, -1, 1)),
            weight=float(self.VOLATILITY_WEIGHTS['keltner'] * vol_weight),
            contribution=float(np.clip(kc_score, -1, 1) * self.VOLATILITY_WEIGHTS['keltner'] * vol_weight),
            explanation=f"Keltner Channel: {'Near upper' if kc_pct > 0.8 else 'Near lower' if kc_pct < 0.2 else 'Mid-channel'}",
            is_bullish=bool(kc_pct < 0.3),
            confidence=float(abs(kc_pct - 0.5) * 2)
        ))
        
        # Squeeze
        squeeze = self._safe_value(features['squeeze'], 0)
        squeeze_release = self._safe_value(features['squeeze_release'], 0)
        ema_align = self._safe_value(features['ema_alignment'], 0)
        if squeeze_release > 0:
            squeeze_score = float(np.sign(ema_align))
        elif squeeze > 0:
            squeeze_score = 0.0  # Waiting for direction
        else:
            squeeze_score = 0.0
        signals.append(IndicatorSignal(
            name="Squeeze",
            agent_category="volatility",
            raw_value=float(squeeze + squeeze_release),
            normalized_score=squeeze_score,
            weight=float(self.VOLATILITY_WEIGHTS['squeeze'] * vol_weight),
            contribution=float(squeeze_score * self.VOLATILITY_WEIGHTS['squeeze'] * vol_weight),
            explanation=self._explain_squeeze(squeeze, squeeze_release, ema_align),
            is_bullish=bool(squeeze_score > 0),
            confidence=float(1.0 if squeeze_release > 0 else 0.3 if squeeze > 0 else 0.5)
        ))
        
        # === VOLUME AGENT ===
        volm_weight = weights.get('volume', 0.17)
        
        # OBV
        obv_trend = self._safe_value(features['obv_trend'], 0)
        obv_score = float(np.clip(obv_trend, -1, 1))
        signals.append(IndicatorSignal(
            name="OBV",
            agent_category="volume",
            raw_value=float(obv_trend),
            normalized_score=obv_score,
            weight=float(self.VOLUME_WEIGHTS['obv'] * volm_weight),
            contribution=float(obv_score * self.VOLUME_WEIGHTS['obv'] * volm_weight),
            explanation=self._explain_obv(obv_trend),
            is_bullish=bool(obv_trend > 0),
            confidence=float(min(1.0, abs(obv_trend)))
        ))
        
        # VWAP
        vwap_dev = self._safe_value(features['vwap_dev'], 0)
        vwap_score = float(-1 * np.clip(vwap_dev * 20, -1, 1))  # Below VWAP = bullish
        signals.append(IndicatorSignal(
            name="VWAP",
            agent_category="volume",
            raw_value=float(vwap_dev),
            normalized_score=vwap_score,
            weight=float(self.VOLUME_WEIGHTS['vwap'] * volm_weight),
            contribution=float(vwap_score * self.VOLUME_WEIGHTS['vwap'] * volm_weight),
            explanation=self._explain_vwap(vwap_dev),
            is_bullish=bool(vwap_dev < 0),
            confidence=float(min(1.0, abs(vwap_dev) * 20))
        ))
        
        # CMF
        cmf = self._safe_value(features['cmf'], 0)
        cmf_score = float(np.clip(cmf * 3, -1, 1))
        signals.append(IndicatorSignal(
            name="CMF",
            agent_category="volume",
            raw_value=float(cmf),
            normalized_score=cmf_score,
            weight=float(self.VOLUME_WEIGHTS['cmf'] * volm_weight),
            contribution=float(cmf_score * self.VOLUME_WEIGHTS['cmf'] * volm_weight),
            explanation=self._explain_cmf(cmf),
            is_bullish=bool(cmf > 0),
            confidence=float(min(1.0, abs(cmf) * 3))
        ))
        
        # Volume Climax
        vol_climax_up = self._safe_value(features['vol_climax_up'], 0)
        vol_climax_down = self._safe_value(features['vol_climax_down'], 0)
        # Climax often signals reversal
        climax_score = float(-0.5 if vol_climax_up else (0.5 if vol_climax_down else 0))
        signals.append(IndicatorSignal(
            name="Volume Climax",
            agent_category="volume",
            raw_value=float(vol_climax_up - vol_climax_down),
            normalized_score=climax_score,
            weight=float(self.VOLUME_WEIGHTS['volume_climax'] * volm_weight),
            contribution=float(climax_score * self.VOLUME_WEIGHTS['volume_climax'] * volm_weight),
            explanation=self._explain_volume_climax(vol_climax_up, vol_climax_down),
            is_bullish=bool(vol_climax_down > 0),  # Climax down often means capitulation
            confidence=float(max(vol_climax_up, vol_climax_down))
        ))
        
        # === PATTERN AGENT ===
        patt_weight = weights.get('pattern', 0.21)
        
        # Candlesticks
        engulfing = self._safe_value(features['engulfing'], 0)
        hammer = self._safe_value(features['hammer'], 0)
        candle_score = float(engulfing + hammer * 0.5)
        signals.append(IndicatorSignal(
            name="Candlesticks",
            agent_category="pattern",
            raw_value=candle_score,
            normalized_score=float(np.clip(candle_score, -1, 1)),
            weight=float(self.PATTERN_WEIGHTS['candlesticks'] * patt_weight),
            contribution=float(np.clip(candle_score, -1, 1) * self.PATTERN_WEIGHTS['candlesticks'] * patt_weight),
            explanation=self._explain_candlesticks(engulfing, hammer),
            is_bullish=bool(candle_score > 0),
            confidence=float(abs(candle_score))
        ))
        
        # Support/Resistance
        near_support = self._safe_value(features['near_support'], 0)
        near_resistance = self._safe_value(features['near_resistance'], 0)
        sr_score = float((near_support - near_resistance) * 0.5)
        signals.append(IndicatorSignal(
            name="Support/Resistance",
            agent_category="pattern",
            raw_value=float(near_support - near_resistance),
            normalized_score=sr_score,
            weight=float(self.PATTERN_WEIGHTS['support_resistance'] * patt_weight),
            contribution=float(sr_score * self.PATTERN_WEIGHTS['support_resistance'] * patt_weight),
            explanation=self._explain_sr(near_support, near_resistance),
            is_bullish=bool(near_support > 0),
            confidence=float(max(near_support, near_resistance))
        ))
        
        # Breakouts
        breakout_up = self._safe_value(features['breakout_up'], 0)
        breakout_down = self._safe_value(features['breakout_down'], 0)
        breakout_score = float(breakout_up - breakout_down)
        signals.append(IndicatorSignal(
            name="Breakouts",
            agent_category="pattern",
            raw_value=breakout_score,
            normalized_score=breakout_score,
            weight=float(self.PATTERN_WEIGHTS['breakouts'] * patt_weight),
            contribution=float(breakout_score * self.PATTERN_WEIGHTS['breakouts'] * patt_weight),
            explanation=self._explain_breakout(breakout_up, breakout_down),
            is_bullish=bool(breakout_up > 0),
            confidence=float(max(breakout_up, breakout_down))
        ))
        
        return signals
    
    def _aggregate_agent_contributions(self, signals: List[IndicatorSignal], 
                                       weights: Dict) -> Dict[str, AgentContribution]:
        """Aggregate indicator signals into agent contributions."""
        agents = {}
        
        for agent_name in ['trend', 'momentum', 'volatility', 'volume', 'pattern']:
            agent_signals = [s for s in signals if s.agent_category == agent_name]
            
            if not agent_signals:
                continue
            
            raw_score = float(sum(s.contribution for s in agent_signals))
            avg_conf = float(np.mean([s.confidence for s in agent_signals]))
            
            # Determine agent signal direction
            bullish = sum(1 for s in agent_signals if s.is_bullish)
            bearish = len(agent_signals) - bullish
            
            if bullish > bearish * 1.5:
                agent_signal = "BUY"
            elif bearish > bullish * 1.5:
                agent_signal = "SELL"
            else:
                agent_signal = "NEUTRAL"
            
            # Generate explanation
            top_signals = sorted(agent_signals, key=lambda s: abs(s.contribution), reverse=True)[:2]
            explanation = " | ".join(s.explanation for s in top_signals)
            
            agents[agent_name] = AgentContribution(
                name=agent_name.title() + " Agent",
                weight=float(weights.get(agent_name, 0.2)),
                raw_score=float(raw_score),
                adjusted_score=float(raw_score),  # Already adjusted via weights
                indicators=agent_signals,
                signal=agent_signal,
                confidence=float(avg_conf),
                explanation=explanation
            )
        
        return agents
    
    def _generate_trade_explanation(self, ticker: str, signal_type: SignalType,
                                    regime: MarketRegime, sector: SectorType,
                                    agents: Dict[str, AgentContribution],
                                    indicators: List[IndicatorSignal]) -> Tuple[str, str, List[str], List[str]]:
        """Generate comprehensive trade explanation."""
        
        # Count bullish/bearish indicators
        bullish = [i for i in indicators if i.is_bullish]
        bearish = [i for i in indicators if not i.is_bullish]
        
        # Top contributors
        sorted_indicators = sorted(indicators, key=lambda i: abs(i.contribution), reverse=True)
        top_bullish = [i for i in sorted_indicators if i.is_bullish][:3]
        top_bearish = [i for i in sorted_indicators if not i.is_bullish][:3]
        
        # Trade thesis
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            direction = "LONG"
            thesis_intro = f"🟢 **{ticker} presents a compelling LONG opportunity**"
            signal_str = "bullish"
        elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            direction = "SHORT"
            thesis_intro = f"🔴 **{ticker} shows bearish signals for a SHORT position**"
            signal_str = "bearish"
        else:
            direction = "NEUTRAL"
            thesis_intro = f"⚪ **{ticker} is currently in consolidation - no clear direction**"
            signal_str = "mixed"
        
        # Regime context
        regime_context = {
            MarketRegime.STRONG_TREND: "strong trending market conditions favor momentum plays",
            MarketRegime.MEAN_REVERTING: "range-bound conditions favor mean reversion strategies",
            MarketRegime.VOLATILITY_BREAKOUT: "volatility squeeze release suggests imminent directional move",
            MarketRegime.ILLIQUID_NOISY: "low liquidity warrants caution - signals may be less reliable",
            MarketRegime.NORMAL: "normal market conditions"
        }
        
        # Sector context
        sector_context = {
            SectorType.TECH_GROWTH: "Tech/Growth sector typically exhibits strong momentum characteristics",
            SectorType.DEFENSIVE: "Defensive sector tends toward mean reversion",
            SectorType.COMMODITY_ENERGY: "Energy/Commodity sector shows cyclical trends",
            SectorType.SMALL_CAP_BIOTECH: "Small-cap/Biotech requires focus on volume confirmation",
            SectorType.GENERAL: ""
        }
        
        trade_thesis = f"""{thesis_intro}

📊 **Market Context:**
- Regime: {regime.value.replace('_', ' ').title()} - {regime_context[regime]}
- Sector: {sector.value.replace('_', ' ').title()} {('- ' + sector_context[sector]) if sector_context[sector] else ''}

📈 **Signal Strength:**
- {len(bullish)}/18 indicators are bullish
- {len(bearish)}/18 indicators are bearish
- Overall signal: {signal_str.upper()}"""
        
        # Why this trade
        why_parts = []
        
        # Agent agreement
        bullish_agents = sum(1 for a in agents.values() if a.signal == "BUY")
        bearish_agents = sum(1 for a in agents.values() if a.signal == "SELL")
        
        if bullish_agents >= 4:
            why_parts.append("Strong cross-agent bullish agreement (4+ agents aligned)")
        elif bearish_agents >= 4:
            why_parts.append("Strong cross-agent bearish agreement (4+ agents aligned)")
        
        # Top signals
        if top_bullish and signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            why_parts.append(f"Key bullish drivers: {', '.join(i.name for i in top_bullish)}")
        if top_bearish and signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            why_parts.append(f"Key bearish drivers: {', '.join(i.name for i in top_bearish)}")
        
        # Regime advantage
        if regime == MarketRegime.STRONG_TREND and agents.get('trend', AgentContribution('', 0, 0, 0, [], '', 0, '')).signal in ["BUY", "SELL"]:
            why_parts.append("Trend Agent confirms direction in trending regime (high conviction)")
        if regime == MarketRegime.VOLATILITY_BREAKOUT:
            why_parts.append("Volatility squeeze breakout provides high-probability setup")
        
        why_this_trade = "\n".join(f"• {p}" for p in why_parts) if why_parts else "• Multiple indicators align with moderate conviction"
        
        # Key factors
        key_factors = []
        for ind in sorted_indicators[:5]:
            if signal_type in [SignalType.STRONG_BUY, SignalType.BUY] and ind.is_bullish:
                key_factors.append(f"✅ {ind.name}: {ind.explanation}")
            elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL] and not ind.is_bullish:
                key_factors.append(f"✅ {ind.name}: {ind.explanation}")
        
        # Risk factors
        risk_factors = []
        contrarian = bearish if signal_type in [SignalType.STRONG_BUY, SignalType.BUY] else bullish
        for ind in contrarian[:3]:
            risk_factors.append(f"⚠️ {ind.name}: {ind.explanation}")
        
        if regime == MarketRegime.ILLIQUID_NOISY:
            risk_factors.append("⚠️ Low liquidity regime - wider stops recommended")
        if regime == MarketRegime.VOLATILITY_BREAKOUT:
            risk_factors.append("⚠️ Post-squeeze volatility - expect large moves in either direction")
        
        return trade_thesis, why_this_trade, key_factors, risk_factors
    
    # ==================== INDICATOR CALCULATIONS ====================
    
    def _get_series(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Safely extract a series from DataFrame."""
        if col not in df.columns:
            return pd.Series(0, index=df.index)
        s = df[col]
        if isinstance(s, pd.DataFrame):
            return s.iloc[:, 0]
        return s
    
    def _safe_value(self, series, default=0.0):
        """Safely get last value from series."""
        try:
            if isinstance(series, (pd.Series, pd.DataFrame)):
                val = series.iloc[-1]
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    val = val.iloc[0]
                return default if pd.isna(val) else float(val)
            return float(series) if not pd.isna(series) else default
        except:
            return default
    
    def _calc_ema_alignment(self, features: pd.DataFrame, close: pd.Series) -> pd.Series:
        """Calculate EMA alignment score."""
        fast = features['ema_fast']
        med = features['ema_medium']
        slow = features['ema_slow']
        
        # Perfect bullish: close > fast > med > slow
        # Perfect bearish: close < fast < med < slow
        alignment = pd.Series(0.0, index=close.index)
        
        for i in range(len(close)):
            score = 0
            if close.iloc[i] > fast.iloc[i]:
                score += 1
            else:
                score -= 1
            if fast.iloc[i] > med.iloc[i]:
                score += 1
            else:
                score -= 1
            if med.iloc[i] > slow.iloc[i]:
                score += 1
            else:
                score -= 1
            alignment.iloc[i] = score / 3
        
        return alignment
    
    def _calc_adx(self, high, low, close, period=14):
        """Calculate ADX with +DI and -DI."""
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
        
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period).mean()
        
        return adx, plus_di, minus_di
    
    def _calc_supertrend(self, high, low, close, period=10, mult=3.0):
        """Calculate Supertrend indicator."""
        atr = self._calc_atr(high, low, close, period)
        hl2 = (high + low) / 2
        
        upper = hl2 + (mult * atr)
        lower = hl2 - (mult * atr)
        
        direction = pd.Series(1.0, index=close.index)
        supertrend = pd.Series(index=close.index, dtype=float)
        
        for i in range(period, len(close)):
            if close.iloc[i] > upper.iloc[i-1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
            
            supertrend.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
        
        return supertrend, direction
    
    def _calc_rsi(self, close, period=14):
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calc_stochastic(self, high, low, close, period=14):
        """Calculate Stochastic oscillator."""
        lowest = low.rolling(period).min()
        highest = high.rolling(period).max()
        
        k = 100 * (close - lowest) / (highest - lowest + 1e-10)
        d = k.rolling(3).mean()
        return k, d
    
    def _detect_divergences(self, price, indicator, lookback=20):
        """Detect price-indicator divergences."""
        bull = pd.Series(0, index=price.index)
        bear = pd.Series(0, index=price.index)
        
        for i in range(lookback, len(price)):
            p_win = price.iloc[i-lookback:i+1]
            i_win = indicator.iloc[i-lookback:i+1]
            
            # Bullish: price lower low, indicator higher low
            if price.iloc[i] < p_win.min() * 1.01:
                if indicator.iloc[i] > i_win.min():
                    bull.iloc[i] = 1
            
            # Bearish: price higher high, indicator lower high
            if price.iloc[i] > p_win.max() * 0.99:
                if indicator.iloc[i] < i_win.max():
                    bear.iloc[i] = 1
        
        return bull, bear
    
    def _calc_atr(self, high, low, close, period=14):
        """Calculate ATR."""
        tr = pd.DataFrame()
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
        return tr['tr'].ewm(span=period, adjust=False).mean()
    
    def _calc_obv(self, close, volume):
        """Calculate On-Balance Volume."""
        obv = pd.Series(0.0, index=close.index)
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv
    
    def _calc_cmf(self, high, low, close, volume, period=20):
        """Calculate Chaikin Money Flow."""
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume
        return mfv.rolling(period).sum() / volume.rolling(period).sum()
    
    def _detect_engulfing(self, open_p, close):
        """Detect engulfing patterns."""
        result = pd.Series(0, index=close.index)
        for i in range(1, len(close)):
            prev_body = close.iloc[i-1] - open_p.iloc[i-1]
            curr_body = close.iloc[i] - open_p.iloc[i]
            
            if prev_body < 0 and curr_body > 0 and open_p.iloc[i] < close.iloc[i-1] and close.iloc[i] > open_p.iloc[i-1]:
                result.iloc[i] = 1  # Bullish engulfing
            elif prev_body > 0 and curr_body < 0 and open_p.iloc[i] > close.iloc[i-1] and close.iloc[i] < open_p.iloc[i-1]:
                result.iloc[i] = -1  # Bearish engulfing
        return result
    
    def _detect_hammer(self, open_p, high, low, close):
        """Detect hammer patterns."""
        body = abs(close - open_p)
        lower_shadow = pd.concat([close, open_p], axis=1).min(axis=1) - low
        upper_shadow = high - pd.concat([close, open_p], axis=1).max(axis=1)
        return ((lower_shadow > 2 * body) & (upper_shadow < body * 0.5) & (body > 0)).astype(int)
    
    def _detect_sr_levels(self, high, low, close, lookback=50, tolerance=0.02):
        """Detect support/resistance levels."""
        near_support = pd.Series(0, index=close.index)
        near_resistance = pd.Series(0, index=close.index)
        
        for i in range(lookback, len(close)):
            win_low = low.iloc[i-lookback:i].min()
            win_high = high.iloc[i-lookback:i].max()
            
            if close.iloc[i] < win_low * (1 + tolerance):
                near_support.iloc[i] = 1
            if close.iloc[i] > win_high * (1 - tolerance):
                near_resistance.iloc[i] = 1
        
        return near_support, near_resistance
    
    def _calc_trend_structure(self, high, low, lookback=10):
        """Calculate trend structure (HH/HL or LH/LL)."""
        score = pd.Series(0.0, index=high.index)
        for i in range(lookback, len(high)):
            hh = high.iloc[i] > high.iloc[i-lookback:i].max()
            hl = low.iloc[i] > low.iloc[i-lookback:i-lookback//2].min()
            ll = low.iloc[i] < low.iloc[i-lookback:i].min()
            lh = high.iloc[i] < high.iloc[i-lookback:i-lookback//2].max()
            
            if hh and hl:
                score.iloc[i] = 1
            elif ll and lh:
                score.iloc[i] = -1
        return score
    
    # ==================== EXPLANATION GENERATORS ====================
    
    def _explain_ema(self, alignment):
        if alignment > 0.7:
            return "Strong bullish EMA alignment (price > fast > medium > slow)"
        elif alignment > 0.3:
            return "Moderate bullish EMA trend"
        elif alignment < -0.7:
            return "Strong bearish EMA alignment (price < fast < medium < slow)"
        elif alignment < -0.3:
            return "Moderate bearish EMA trend"
        return "Mixed EMA signals - consolidation"
    
    def _explain_macd(self, hist, features):
        if hist > 0.01:
            return f"MACD bullish - histogram expanding (+{hist:.4f})"
        elif hist < -0.01:
            return f"MACD bearish - histogram contracting ({hist:.4f})"
        return "MACD neutral - awaiting crossover"
    
    def _explain_adx(self, adx, plus_di, minus_di):
        strength = "strong" if adx > 25 else "weak"
        direction = "bullish (+DI > -DI)" if plus_di > minus_di else "bearish (-DI > +DI)"
        return f"ADX={adx:.1f} ({strength} trend) - {direction}"
    
    def _explain_rsi(self, rsi):
        if rsi < 30:
            return f"RSI={rsi:.1f} OVERSOLD - potential bounce setup"
        elif rsi > 70:
            return f"RSI={rsi:.1f} OVERBOUGHT - potential pullback"
        elif rsi > 50:
            return f"RSI={rsi:.1f} bullish momentum"
        return f"RSI={rsi:.1f} bearish momentum"
    
    def _score_rsi(self, rsi):
        """Score RSI for mean reversion signal."""
        if rsi < 30:
            return (30 - rsi) / 30  # Oversold = bullish
        elif rsi > 70:
            return -(rsi - 70) / 30  # Overbought = bearish
        return (rsi - 50) / 50  # Trending
    
    def _explain_stochastic(self, k, d):
        if k < 20:
            cross = "bullish crossover" if k > d else "awaiting crossover"
            return f"Stoch OVERSOLD (K={k:.1f}) - {cross}"
        elif k > 80:
            cross = "bearish crossover" if k < d else "awaiting crossover"
            return f"Stoch OVERBOUGHT (K={k:.1f}) - {cross}"
        return f"Stoch neutral (K={k:.1f})"
    
    def _score_stochastic(self, k, d):
        """Score stochastic oscillator."""
        if k < 20 and k > d:
            return 1.0  # Oversold with bullish cross
        elif k > 80 and k < d:
            return -1.0  # Overbought with bearish cross
        elif k < 30:
            return 0.5  # Oversold
        elif k > 70:
            return -0.5  # Overbought
        return (k - 50) / 50
    
    def _explain_divergences(self, bull, bear):
        if bull > 0:
            return "BULLISH DIVERGENCE detected - price making lows but momentum rising"
        elif bear > 0:
            return "BEARISH DIVERGENCE detected - price making highs but momentum falling"
        return "No divergences detected"
    
    def _explain_bollinger(self, bb_pct):
        if bb_pct < 0.1:
            return f"BB%={bb_pct:.2f} - BELOW lower band (oversold)"
        elif bb_pct > 0.9:
            return f"BB%={bb_pct:.2f} - ABOVE upper band (overbought)"
        elif bb_pct < 0.3:
            return "Near lower Bollinger Band"
        elif bb_pct > 0.7:
            return "Near upper Bollinger Band"
        return "Mid Bollinger Band range"
    
    def _score_bollinger(self, bb_pct):
        """Score Bollinger position for mean reversion."""
        if bb_pct < 0.1:
            return 1.0
        elif bb_pct > 0.9:
            return -1.0
        return -1 * (bb_pct - 0.5) * 2
    
    def _explain_squeeze(self, squeeze, release, ema_align):
        if release > 0:
            direction = "BULLISH" if ema_align > 0 else "BEARISH"
            return f"SQUEEZE BREAKOUT {direction} - expect directional move"
        elif squeeze > 0:
            return "SQUEEZE ACTIVE - volatility compression, breakout imminent"
        return "No squeeze - normal volatility"
    
    def _explain_obv(self, trend):
        if trend > 0.5:
            return "OBV rising - accumulation (smart money buying)"
        elif trend < -0.5:
            return "OBV falling - distribution (smart money selling)"
        return "OBV neutral"
    
    def _explain_vwap(self, dev):
        if dev < -0.02:
            return f"Trading {abs(dev)*100:.1f}% BELOW VWAP - potential value"
        elif dev > 0.02:
            return f"Trading {dev*100:.1f}% ABOVE VWAP - extended"
        return "Trading near VWAP"
    
    def _explain_cmf(self, cmf):
        if cmf > 0.2:
            return f"CMF={cmf:.2f} STRONG buying pressure"
        elif cmf < -0.2:
            return f"CMF={cmf:.2f} STRONG selling pressure"
        elif cmf > 0:
            return f"CMF={cmf:.2f} mild buying pressure"
        return f"CMF={cmf:.2f} mild selling pressure"
    
    def _explain_volume_climax(self, up, down):
        if up > 0:
            return "VOLUME CLIMAX on UP move - possible exhaustion top"
        elif down > 0:
            return "VOLUME CLIMAX on DOWN move - possible capitulation bottom"
        return "Normal volume"
    
    def _explain_candlesticks(self, engulfing, hammer):
        patterns = []
        if engulfing > 0:
            patterns.append("Bullish Engulfing")
        elif engulfing < 0:
            patterns.append("Bearish Engulfing")
        if hammer > 0:
            patterns.append("Hammer")
        return ", ".join(patterns) if patterns else "No significant patterns"
    
    def _explain_sr(self, support, resistance):
        if support > 0:
            return "Near SUPPORT level - potential bounce"
        elif resistance > 0:
            return "Near RESISTANCE level - potential rejection"
        return "Mid-range - no key levels nearby"
    
    def _explain_breakout(self, up, down):
        if up > 0:
            return "20-bar HIGH BREAKOUT - bullish momentum"
        elif down > 0:
            return "20-bar LOW BREAKDOWN - bearish momentum"
        return "No breakout detected"

