import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

def ledoit_wolf_cov(returns: pd.DataFrame):
    lw = LedoitWolf().fit(returns.dropna())
    return lw.covariance_, returns.columns.tolist()

def expected_returns(returns: pd.DataFrame, half_life: int = 60):
    # exponentially weighted mean
    ew = returns.ewm(halflife=half_life).mean().iloc[-1]
    return ew
