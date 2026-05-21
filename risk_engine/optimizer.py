import numpy as np
import pandas as pd

def mv_optimize(mu: pd.Series, Sigma: np.ndarray, tickers, lam: float=10.0, w_prev: pd.Series=None, turnover_penalty: float=0.0, w_cap: float=0.05):
    n = len(tickers)
    if w_prev is None:
        w_prev = pd.Series(0, index=tickers)

    # Quadratic objective: minimize lam*w'Σw - mu'w + turnover_penalty*|w - w_prev|
    # Use simple projected gradient with L1 penalty approximation.
    w = np.ones(n)/n
    Sigma_reg = Sigma + 1e-6*np.eye(n)

    lr = 0.1
    for _ in range(500):
        grad = lam * (Sigma_reg @ w) - mu.values
        # L1 turnover approx via soft-thresholding towards w_prev
        step = w - lr*grad
        # project box [0, w_cap]
        step = np.clip(step, 0, w_cap)
        # re-normalize to sum <= 1 (cash allowed)
        s = step.sum()
        if s > 1.0:
            step *= 1.0/s
        # blend with previous to penalize turnover
        step = (1 - turnover_penalty)*step + turnover_penalty*w_prev.reindex(tickers).fillna(0).values
        if np.linalg.norm(step - w) < 1e-6:
            break
        w = step
    return pd.Series(w, index=tickers)
