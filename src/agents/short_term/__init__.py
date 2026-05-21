# Short-Term Multi-Agent Trading System
# Specialized for hourly data with 1-week prediction horizon

from .base_agent import BaseAgent, AgentSignal, AgentConfig, SignalType
from .trend_agent import TrendAgent
from .momentum_agent import MomentumAgent
from .volatility_agent import VolatilityAgent
from .volume_agent import VolumeAgent
from .pattern_agent import PatternAgent
from .meta_agent import MetaAgent, MetaSignal
from .orchestrator import ShortTermOrchestrator, ShortTermPrediction
from .advanced_weighted_agent import (
    AdvancedWeightedAgent, AdvancedSignal, 
    MarketRegime, SectorType, IndicatorSignal, AgentContribution
)
from .advanced_orchestrator import AdvancedOrchestrator, AdvancedPrediction

__all__ = [
    # Base classes
    'BaseAgent', 'AgentSignal', 'AgentConfig', 'SignalType',
    
    # Standard agents
    'TrendAgent', 'MomentumAgent', 'VolatilityAgent', 
    'VolumeAgent', 'PatternAgent',
    
    # Meta agent
    'MetaAgent', 'MetaSignal',
    
    # Standard orchestrator
    'ShortTermOrchestrator', 'ShortTermPrediction',
    
    # Advanced weighted system
    'AdvancedWeightedAgent', 'AdvancedSignal',
    'MarketRegime', 'SectorType', 'IndicatorSignal', 'AgentContribution',
    
    # Advanced orchestrator
    'AdvancedOrchestrator', 'AdvancedPrediction'
]
