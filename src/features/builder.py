"""
Feature Builder Module

Combines all feature sources:
- Technical indicators (MA, RSI, MACD, BB)
- Volatility features (ATR, realized vol)
- Advanced features (ADX, Stochastic, MFI, Hurst)
- Cross-asset features (relative strength, correlation, regime)
"""

import pandas as pd
import numpy as np
from src.features.tech import basic_tech
from src.features.vol import vol_features
from src.features.advanced import make_advanced

# Try to import cross-asset features (optional)
try:
    from src.features.cross_asset import make_cross_asset_features, get_cached_market_data
    CROSS_ASSET_AVAILABLE = True
except ImportError:
    CROSS_ASSET_AVAILABLE = False


def make_features(df: pd.DataFrame, include_cross_asset: bool = True) -> pd.DataFrame:
    """
    Build complete feature set from OHLCV data.
    
    Args:
        df: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
        include_cross_asset: Whether to include cross-asset features
        
    Returns:
        DataFrame with all features
    """
    # Core feature sets
    tech = basic_tech(df)
    vol = vol_features(df)
    adv = make_advanced(df)
    
    # Join all features
    X = tech.join([vol, adv], how='outer')
    
    # Add cross-asset features if available and requested
    if include_cross_asset and CROSS_ASSET_AVAILABLE:
        try:
            market_df = get_cached_market_data()
            cross = make_cross_asset_features(df, market_df)
            X = X.join(cross, how='outer')
        except Exception as e:
            # Silently continue without cross-asset features
            pass
    
    # Handle missing values
    # Only drop rows where ALL values are NaN (preserves more data)
    X = X.dropna(how='all')
    
    # For remaining NaN values, forward fill then backward fill
    X = X.ffill().bfill()
    
    # Final dropna to remove any remaining NaN
    X = X.dropna()
    
    # Replace any infinities
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return X


def get_feature_names() -> dict:
    """
    Get categorized feature names.
    
    Returns:
        Dictionary mapping category to list of feature names
    """
    return {
        "technical": [
            "ma20", "ma50", "ma100", "ma200",
            "dma20", "dma50", "dma100", "dma200",
            "rsi14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_width", "bb_position",
            "volume_ratio", "price_volume"
        ],
        "volatility": [
            "vol_5d", "vol_10d", "vol_20d", "vol_60d",
            "atr14", "atr20", "atr14_pct", "atr20_pct",
            "vol_of_vol", "vol_regime", "downside_vol"
        ],
        "advanced": [
            "log_slope_20", "log_slope_50",
            "mom_20", "mom_60", "mom_120",
            "adx14", "stoch_k", "stoch_d", "mfi14",
            "down_vol", "up_vol", "vol_skew",
            "z_ret_5", "z_atr", "hurst"
        ],
        "cross_asset": [
            "rs_spy_20", "rs_spy_60", "beta_60", "alpha_60",
            "corr_spy_20", "corr_spy_60", "corr_change",
            "market_trend", "market_vol_regime", "market_momentum"
        ]
    }


def get_feature_importance_groups() -> dict:
    """
    Get feature importance groupings for analysis.
    
    Returns:
        Dictionary mapping group name to feature patterns
    """
    return {
        "momentum": ["mom_", "rsi", "macd", "stoch"],
        "trend": ["ma", "dma", "slope", "adx"],
        "volatility": ["vol_", "atr", "bb_"],
        "market": ["rs_spy", "corr_spy", "market_", "beta", "alpha"],
        "other": ["mfi", "hurst", "z_"]
    }


if __name__ == "__main__":
    # Test the builder
    import yfinance as yf
    
    print("Testing feature builder...")
    
    df = yf.download("AAPL", period="1y", interval="1d", auto_adjust=True, progress=False)
    df = df.rename(columns=str.title)
    
    X = make_features(df)
    print(f"\nGenerated {len(X.columns)} features, {len(X)} samples")
    print(f"\nFeature categories:")
    for category, features in get_feature_names().items():
        available = [f for f in features if f in X.columns]
        print(f"  {category}: {len(available)}/{len(features)} features")
    
    print(f"\nSample feature values (latest):")
    for col in X.columns[:10]:
        print(f"  {col}: {X[col].iloc[-1]:.4f}")
