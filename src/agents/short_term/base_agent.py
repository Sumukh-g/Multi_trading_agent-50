"""
Base Agent Class for Short-Term Trading

All specialized agents inherit from this base class.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class AgentSignal:
    """Signal output from an agent."""
    signal_type: SignalType
    confidence: float  # 0-1
    strength: float    # 0-100
    reasoning: str
    metrics: Dict[str, float]
    timestamp: pd.Timestamp


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    weight: float = 1.0
    enabled: bool = True
    params: Dict = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Each agent specializes in a specific type of analysis:
    - Technical indicators
    - Price patterns
    - Volume analysis
    - Volatility modeling
    - Momentum detection
    - Market microstructure
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.weight = config.weight
        self.params = config.params
        self.last_signal: Optional[AgentSignal] = None
        self.signal_history: List[AgentSignal] = []
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame, ticker: str) -> AgentSignal:
        """
        Analyze price data and generate a trading signal.
        
        Args:
            df: OHLCV DataFrame with hourly data
            ticker: Stock symbol
            
        Returns:
            AgentSignal with the agent's recommendation
        """
        pass
    
    @abstractmethod
    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features specific to this agent's analysis type.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with agent-specific features
        """
        pass
    
    def update_signal(self, signal: AgentSignal):
        """Store the latest signal and update history."""
        self.last_signal = signal
        self.signal_history.append(signal)
        # Keep only last 100 signals
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
    
    def get_signal_consistency(self, lookback: int = 10) -> float:
        """
        Calculate how consistent recent signals have been.
        
        Returns:
            Float 0-1 indicating consistency (1 = all same direction)
        """
        if len(self.signal_history) < 2:
            return 0.5
        
        recent = self.signal_history[-lookback:]
        directions = [s.signal_type.value for s in recent]
        
        # Calculate standard deviation of directions
        if np.std(directions) == 0:
            return 1.0
        
        # Normalize to 0-1 (max std for 5-point scale is ~2)
        return max(0, 1 - np.std(directions) / 2)
    
    def _safe_value(self, series: pd.Series, default: float = 0.0) -> float:
        """Safely extract last value from series."""
        try:
            val = series.iloc[-1]
            if pd.isna(val):
                return default
            return float(val)
        except:
            return default

