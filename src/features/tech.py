import pandas as pd
import numpy as np

def basic_tech(df: pd.DataFrame) -> pd.DataFrame:
    """Basic technical indicators: moving averages, RSI, etc."""
    out = pd.DataFrame(index=df.index)
    
    # Ensure we're working with Series not DataFrame
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    # Moving averages
    for n in [20, 50, 100, 200]:
        ma = close.rolling(n).mean()
        out[f'ma{n}'] = ma
        out[f'dma{n}'] = (close - ma) / (ma + 1e-10)  # distance from MA
    
    # RSI
    delta = close.diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    out['rsi14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out['macd'] = ema12 - ema26
    out['macd_signal'] = out['macd'].ewm(span=9, adjust=False).mean()
    out['macd_hist'] = out['macd'] - out['macd_signal']
    
    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    out['bb_upper'] = bb_upper
    out['bb_lower'] = bb_lower
    out['bb_width'] = (bb_upper - bb_lower) / (ma20 + 1e-10)
    out['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
    
    # Volume indicators
    if 'Volume' in df.columns:
        volume = df['Volume']
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        vol_ma = volume.rolling(20).mean()
        out['volume_ratio'] = volume / (vol_ma + 1e-10)
        out['price_volume'] = close * volume
    
    return out.replace([np.inf, -np.inf], np.nan)

