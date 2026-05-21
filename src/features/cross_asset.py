"""
Cross-Asset Features Module

Adds features based on relationships between assets:
- Relative strength vs market (SPY)
- Correlation features
- Sector momentum
- Market regime indicators
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


def get_market_data(period: str = "2y") -> pd.DataFrame:
    """
    Download market benchmark data (SPY, VIX).
    
    Returns:
        DataFrame with SPY and VIX data
    """
    try:
        spy = yf.download("SPY", period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy = spy.droplevel(0, axis=1)
        spy = spy.rename(columns=str.title)
        return spy
    except:
        return pd.DataFrame()


def relative_strength_features(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate relative strength features vs market benchmark.
    
    Args:
        df: Stock OHLCV data
        market_df: Market benchmark (SPY) data
        
    Returns:
        DataFrame with relative strength features
    """
    out = pd.DataFrame(index=df.index)
    
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    if market_df.empty:
        # Return neutral features if no market data
        out['rs_spy_20'] = 0.0
        out['rs_spy_60'] = 0.0
        out['beta_60'] = 1.0
        out['alpha_60'] = 0.0
        return out
    
    spy_close = market_df['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    
    # Align indices
    common_idx = close.index.intersection(spy_close.index)
    if len(common_idx) < 60:
        out['rs_spy_20'] = 0.0
        out['rs_spy_60'] = 0.0
        out['beta_60'] = 1.0
        out['alpha_60'] = 0.0
        return out
    
    stock_ret = close.reindex(common_idx).pct_change()
    spy_ret = spy_close.reindex(common_idx).pct_change()
    
    # Relative strength: stock return - market return (rolling)
    out['rs_spy_20'] = (stock_ret.rolling(20).mean() - spy_ret.rolling(20).mean())
    out['rs_spy_60'] = (stock_ret.rolling(60).mean() - spy_ret.rolling(60).mean())
    
    # Rolling beta (60-day)
    def rolling_beta(y, x, window=60):
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var()
        return cov / (var + 1e-10)
    
    out['beta_60'] = rolling_beta(stock_ret, spy_ret)
    
    # Rolling alpha
    out['alpha_60'] = stock_ret.rolling(60).mean() - out['beta_60'] * spy_ret.rolling(60).mean()
    
    # Reindex to original index and fill
    out = out.reindex(df.index).ffill().bfill().fillna(0)
    
    return out


def correlation_features(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation-based features.
    
    Args:
        df: Stock OHLCV data
        market_df: Market benchmark data
        
    Returns:
        DataFrame with correlation features
    """
    out = pd.DataFrame(index=df.index)
    
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    if market_df.empty:
        out['corr_spy_20'] = 0.5
        out['corr_spy_60'] = 0.5
        out['corr_change'] = 0.0
        return out
    
    spy_close = market_df['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    
    common_idx = close.index.intersection(spy_close.index)
    if len(common_idx) < 60:
        out['corr_spy_20'] = 0.5
        out['corr_spy_60'] = 0.5
        out['corr_change'] = 0.0
        return out
    
    stock_ret = close.reindex(common_idx).pct_change()
    spy_ret = spy_close.reindex(common_idx).pct_change()
    
    # Rolling correlation
    out['corr_spy_20'] = stock_ret.rolling(20).corr(spy_ret)
    out['corr_spy_60'] = stock_ret.rolling(60).corr(spy_ret)
    
    # Correlation change (regime indicator)
    out['corr_change'] = out['corr_spy_20'] - out['corr_spy_60']
    
    out = out.reindex(df.index).ffill().bfill().fillna(0.5)
    
    return out


def market_regime_features(df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate market regime features.
    
    Args:
        df: Stock OHLCV data
        market_df: Market benchmark data
        
    Returns:
        DataFrame with regime features
    """
    out = pd.DataFrame(index=df.index)
    
    if market_df.empty:
        out['market_trend'] = 0.0
        out['market_vol_regime'] = 0.0
        out['market_momentum'] = 0.0
        return out
    
    spy_close = market_df['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    
    spy_ret = spy_close.pct_change()
    
    # Market trend (above/below 200 SMA)
    sma200 = spy_close.rolling(200).mean()
    out['market_trend'] = ((spy_close - sma200) / (sma200 + 1e-10)).reindex(df.index).ffill().bfill()
    
    # Market volatility regime
    vol_20 = spy_ret.rolling(20).std()
    vol_60 = spy_ret.rolling(60).std()
    out['market_vol_regime'] = ((vol_20 / (vol_60 + 1e-10)) - 1).reindex(df.index).ffill().bfill()
    
    # Market momentum
    out['market_momentum'] = spy_close.pct_change(60).reindex(df.index).ffill().bfill()
    
    out = out.fillna(0)
    
    return out


def make_cross_asset_features(df: pd.DataFrame, market_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generate all cross-asset features.
    
    Args:
        df: Stock OHLCV data
        market_df: Optional pre-loaded market data
        
    Returns:
        DataFrame with all cross-asset features
    """
    if market_df is None:
        market_df = get_market_data()
    
    rs = relative_strength_features(df, market_df)
    corr = correlation_features(df, market_df)
    regime = market_regime_features(df, market_df)
    
    out = rs.join([corr, regime], how='outer')
    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    
    return out


# Global market data cache
_market_data_cache = None


def get_cached_market_data() -> pd.DataFrame:
    """Get cached market data or download if not available."""
    global _market_data_cache
    if _market_data_cache is None:
        _market_data_cache = get_market_data()
    return _market_data_cache


if __name__ == "__main__":
    # Test the module
    import yfinance as yf
    
    print("Testing cross-asset features...")
    
    df = yf.download("AAPL", period="1y", interval="1d", auto_adjust=True, progress=False)
    df = df.rename(columns=str.title)
    
    features = make_cross_asset_features(df)
    print(f"\nGenerated {len(features.columns)} cross-asset features:")
    for col in features.columns:
        print(f"  - {col}: {features[col].iloc[-1]:.4f}")

