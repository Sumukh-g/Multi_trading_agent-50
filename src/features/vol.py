import pandas as pd
import numpy as np

def vol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility and risk features."""
    out = pd.DataFrame(index=df.index)
    
    # Ensure we're working with Series not DataFrame
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    ret = close.pct_change()
    
    # Realized volatility (rolling std)
    for n in [5, 10, 20, 60]:
        out[f'vol_{n}d'] = ret.rolling(n).std() * np.sqrt(252)  # annualized
    
    # ATR-based volatility
    if 'High' in df.columns and 'Low' in df.columns:
        high = df['High']
        low = df['Low']
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        
        tr1 = (high - low).abs()
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        for n in [14, 20]:
            atr = tr.rolling(n).mean()
            out[f'atr{n}'] = atr
            out[f'atr{n}_pct'] = atr / (close + 1e-10)
    
    # Volatility of volatility
    vol20 = ret.rolling(20).std()
    out['vol_of_vol'] = vol20.rolling(20).std()
    
    # Volatility regime (high/low)
    vol60 = ret.rolling(60).std()
    vol20 = ret.rolling(20).std()
    out['vol_regime'] = (vol20 / (vol60 + 1e-10) - 1.0)  # >0 means recent vol higher
    
    # Downside volatility
    downside = ret.mask(ret > 0, 0)
    out['downside_vol'] = downside.rolling(20).std() * np.sqrt(252)
    
    return out.replace([np.inf, -np.inf], np.nan)

