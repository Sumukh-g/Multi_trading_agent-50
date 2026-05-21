import numpy as np

def atr_stop(price: float, atr_pct: float, mult: float = 2.0, direction: int = 1) -> float:
    """
    Calculate ATR-based stop loss.
    direction: 1 for long, -1 for short
    """
    stop_dist = price * atr_pct * mult
    if direction > 0:
        return price - stop_dist  # long stop below
    else:
        return price + stop_dist  # short stop above

def position_size(prob: float, exp_ret_abs: float, vol_est: float, base_risk: float = 0.01) -> float:
    """
    Kelly-inspired position sizing.
    prob: probability of move
    exp_ret_abs: absolute expected return
    vol_est: volatility estimate (ATR%)
    base_risk: base risk per name
    """
    if vol_est <= 0 or exp_ret_abs <= 0:
        return 0.0
    # Simplified: size = base_risk * (prob * exp_ret) / vol
    size = base_risk * (prob * exp_ret_abs) / vol_est
    return min(size, 0.20)  # cap at 20% per name

