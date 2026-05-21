import pandas as pd
import numpy as np

def rolling_z(series: pd.Series, n:int=60):
    mu = series.rolling(n).mean()
    sd = series.rolling(n).std().replace(0, np.nan)
    return (series - mu) / sd

def slope_pct(series: pd.Series, n:int=20):
    # slope of log price over n days, expressed as pct per day
    y = np.log(series)
    x = np.arange(len(y))
    def _seg(i):
        xs = x[i-n+1:i+1]
        ys = y.iloc[i-n+1:i+1]
        if len(xs)<n: return np.nan
        b = np.polyfit(xs, ys, 1)[0]
        return b
    vals = [np.nan]*len(y)
    for i in range(len(y)):
        vals[i] = _seg(i)
    return pd.Series(vals, index=series.index)

def adx(df: pd.DataFrame, n:int=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Ensure Series not DataFrame
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (low.diff() < 0), -low.diff(), 0.0)
    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    pdi = 100 * (pd.Series(plus_dm, index=high.index).rolling(n).mean() / atr.replace(0,np.nan))
    mdi = 100 * (pd.Series(minus_dm, index=high.index).rolling(n).mean() / atr.replace(0,np.nan))
    dx = ( (pdi - mdi).abs() / (pdi + mdi).replace(0,np.nan) ) * 100
    adx_val = dx.rolling(n).mean()
    return adx_val.fillna(0)

def stochastic(df: pd.DataFrame, n:int=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Ensure Series not DataFrame
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    lowest = low.rolling(n).min()
    highest = high.rolling(n).max()
    k = 100 * (close - lowest) / (highest - lowest).replace(0,np.nan)
    d = k.rolling(3).mean()
    return k.fillna(50), d.fillna(50)

def mfi(df: pd.DataFrame, n:int=14):
    # Money Flow Index using typical price * volume
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    
    # Ensure Series not DataFrame
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]
    
    tp = (high + low + close) / 3.0
    mf = tp * volume
    pos = np.where(tp.diff() > 0, mf, 0.0)
    neg = np.where(tp.diff() < 0, mf, 0.0)
    pos_sum = pd.Series(pos, index=df.index).rolling(n).sum()
    neg_sum = pd.Series(neg, index=df.index).rolling(n).sum()
    rs = (pos_sum / (neg_sum + 1e-12))
    mfi_val = 100 - (100 / (1 + rs))
    return mfi_val.fillna(50)

def hurst_exponent(close: pd.Series, max_lag:int=20):
    # rough Hurst estimator (R/S)
    lags = range(2, max_lag+1)
    tau = [np.sqrt(np.std(close.diff(l))) for l in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0]*2.0
    return pd.Series([np.nan]*(len(close)-1) + [hurst], index=close.index)

def make_advanced(df: pd.DataFrame):
    out = pd.DataFrame(index=df.index)
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Ensure Series not DataFrame
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    
    # Trend & speed
    out['log_slope_20'] = slope_pct(close, 20)
    out['log_slope_50'] = slope_pct(close, 50)
    # Momentum ranks
    for n in [20,60,120]:
        out[f'mom_{n}'] = close.pct_change(n)
    # ADX strength
    out['adx14'] = adx(df, 14)
    # Stochastics
    k, d = stochastic(df, 14)
    out['stoch_k'] = k
    out['stoch_d'] = d
    # Money Flow
    out['mfi14'] = mfi(df, 14)
    # Volatility shape
    ret = close.pct_change()
    out['down_vol'] = ret.mask(ret>0, 0).rolling(20).std()
    out['up_vol'] = ret.mask(ret<0, 0).rolling(20).std()
    out['vol_skew'] = out['up_vol'] - out['down_vol']
    # Z-scores
    out['z_ret_5'] = rolling_z(close.pct_change(5), 60)
    atr_raw = (high - low).rolling(14).mean() / (close + 1e-10)
    out['z_atr'] = rolling_z(atr_raw, 60)
    # Hurst (coarse; constant within window)
    out['hurst'] = hurst_exponent(close).ffill()
    return out.replace([np.inf, -np.inf], np.nan)
