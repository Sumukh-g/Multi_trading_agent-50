"""
Multi-Threshold Label Generation Module

Creates labels for:
- 2% move threshold (high frequency)
- 5% move threshold (medium frequency)  
- 10% move threshold (low frequency)
- Direction (up vs down)
- Raw forward returns
"""

import pandas as pd
import numpy as np
from typing import Optional


def label_bigmove_30d(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Legacy function: Create labels for 30-day forward returns.
    
    Args:
        df: OHLCV DataFrame
        threshold: Move threshold (default 5%)
        
    Returns:
        DataFrame with y_bigmove and y_fwd columns
    """
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    fwd_ret = close.pct_change(30).shift(-30)
    y_bigmove = (np.abs(fwd_ret) >= threshold).astype(int)
    
    out = pd.DataFrame({
        'y_bigmove': y_bigmove,
        'y_fwd': fwd_ret
    }, index=df.index)
    
    return out


def label_multi_threshold(
    df: pd.DataFrame,
    horizon: int = 30,
    thresholds: Optional[dict] = None
) -> pd.DataFrame:
    """
    Create multi-threshold labels for forward returns.
    
    Args:
        df: OHLCV DataFrame
        horizon: Prediction horizon in days (default 30)
        thresholds: Dictionary of threshold names to values
                   Default: {"2pct": 0.02, "5pct": 0.05, "10pct": 0.10}
    
    Returns:
        DataFrame with columns:
        - y_move_2pct: 1 if |return| >= 2%
        - y_move_5pct: 1 if |return| >= 5%
        - y_move_10pct: 1 if |return| >= 10%
        - y_direction: 1 if return > 0, else 0
        - y_fwd: raw forward return
        - y_bigmove: legacy (same as y_move_5pct)
    """
    if thresholds is None:
        thresholds = {
            "2pct": 0.02,
            "5pct": 0.05,
            "10pct": 0.10
        }
    
    # Get close price series
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    # Calculate forward return
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    
    # Create output DataFrame
    out = pd.DataFrame(index=df.index)
    
    # Multi-threshold labels
    for name, thresh in thresholds.items():
        out[f'y_move_{name}'] = (np.abs(fwd_ret) >= thresh).astype(int)
    
    # Direction label (1 = up, 0 = down)
    out['y_direction'] = (fwd_ret > 0).astype(int)
    
    # Raw forward return
    out['y_fwd'] = fwd_ret
    
    # Legacy compatibility
    out['y_bigmove'] = out['y_move_5pct']
    
    return out


def label_with_magnitude(
    df: pd.DataFrame,
    horizon: int = 30
) -> pd.DataFrame:
    """
    Create labels that include magnitude classification.
    
    Args:
        df: OHLCV DataFrame
        horizon: Prediction horizon
        
    Returns:
        DataFrame with multi-class magnitude labels
    """
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    
    out = pd.DataFrame(index=df.index)
    
    # Multi-class magnitude (0=small, 1=medium, 2=large)
    def classify_magnitude(r):
        if pd.isna(r):
            return np.nan
        abs_r = abs(r)
        if abs_r >= 0.10:
            return 2  # Large
        elif abs_r >= 0.05:
            return 1  # Medium
        else:
            return 0  # Small
    
    out['y_magnitude'] = fwd_ret.apply(classify_magnitude)
    
    # Signed magnitude (-2 to +2)
    def classify_signed(r):
        if pd.isna(r):
            return np.nan
        if r >= 0.10:
            return 2
        elif r >= 0.05:
            return 1
        elif r > -0.05:
            return 0
        elif r > -0.10:
            return -1
        else:
            return -2
    
    out['y_signed_magnitude'] = fwd_ret.apply(classify_signed)
    
    out['y_fwd'] = fwd_ret
    out['y_direction'] = (fwd_ret > 0).astype(int)
    
    return out


def label_risk_adjusted(
    df: pd.DataFrame,
    horizon: int = 30,
    vol_window: int = 20
) -> pd.DataFrame:
    """
    Create risk-adjusted labels using volatility-normalized returns.
    
    Args:
        df: OHLCV DataFrame
        horizon: Prediction horizon
        vol_window: Window for volatility calculation
        
    Returns:
        DataFrame with risk-adjusted labels
    """
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    # Forward return
    fwd_ret = close.pct_change(horizon).shift(-horizon)
    
    # Historical volatility
    ret = close.pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(252)  # Annualized
    
    # Risk-adjusted return (Sharpe-like)
    risk_adj_ret = fwd_ret / (vol + 1e-6)
    
    out = pd.DataFrame(index=df.index)
    out['y_fwd'] = fwd_ret
    out['y_fwd_risk_adj'] = risk_adj_ret
    out['y_volatility'] = vol
    
    # Risk-adjusted move classification
    out['y_risk_adj_move'] = (np.abs(risk_adj_ret) >= 1.0).astype(int)  # 1 std move
    out['y_direction'] = (fwd_ret > 0).astype(int)
    
    return out
