import numpy as np
import pandas as pd
from typing import Tuple, Dict

def selective_threshold_from_oof(y_true: pd.Series, p_hat: pd.Series, epsilon: float=0.15) -> Tuple[float, Dict]:
    y = y_true.values.astype(int)
    p = p_hat.values.astype(float)
    conf = np.maximum(p, 1-p)
    pred = (p >= 0.5).astype(int)
    order = np.argsort(-conf)
    conf_s = conf[order]
    y_s = y[order]
    pred_s = pred[order]
    errors = (pred_s != y_s).astype(int)
    cum_err = np.cumsum(errors)
    idx = np.arange(1, len(y_s)+1)
    err_rate = cum_err / idx
    ok = np.where(err_rate <= epsilon)[0]
    if len(ok)==0:
        tau = 1.01
        meta = {"coverage": 0.0, "emp_err_rate": None}
        return tau, meta
    k = ok[-1] + 1
    tau = float(conf_s[k-1])
    coverage = float(k/len(y_s))
    meta = {"coverage": coverage, "emp_err_rate": float(err_rate[k-1]), "k": int(k)}
    return tau, meta

def conformal_qhat_from_oof(y_true: pd.Series, y_pred: pd.Series, alpha: float=0.2) -> float:
    resid = np.abs(y_true.values - y_pred.values)
    resid = resid[~np.isnan(resid)]
    if resid.size == 0:
        return 0.05
    return float(np.quantile(resid, 1 - alpha))
