"""
Advanced Inference Module

Provides multi-threshold predictions with:
- 2%, 5%, 10% move probability
- Direction prediction
- Signal strength scoring
- Model agreement metrics
- Multiple prediction modes
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, Optional
from enum import Enum
from src.features.builder import make_features


class PredictionMode(Enum):
    """Prediction mode affecting confidence thresholds."""
    CONSERVATIVE = "conservative"  # Stricter thresholds, fewer signals
    BALANCED = "balanced"          # Default balanced approach
    AGGRESSIVE = "aggressive"      # Looser thresholds, more signals


def _load_unc() -> Dict:
    """Load uncertainty parameters."""
    try:
        with open("models/uncertainty.json", "r") as f:
            return json.load(f)
    except:
        return {
            "selective_threshold": 0.85,
            "q_hat": 0.05,
            "alpha": 0.2
        }


def _load_model(path: str):
    """Safely load a model file."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


def _get_mode_adjustments(mode: str) -> Dict:
    """Get threshold adjustments based on prediction mode."""
    adjustments = {
        "conservative": {
            "tau_mult": 1.15,      # Higher confidence required
            "strength_mult": 0.85, # Lower strength scores
            "min_agreement": 0.7   # Higher model agreement needed
        },
        "balanced": {
            "tau_mult": 1.0,
            "strength_mult": 1.0,
            "min_agreement": 0.6
        },
        "aggressive": {
            "tau_mult": 0.85,      # Lower confidence required
            "strength_mult": 1.15, # Higher strength scores
            "min_agreement": 0.5   # Lower agreement needed
        }
    }
    return adjustments.get(mode, adjustments["balanced"])


def calculate_signal_strength(
    prob_2pct: float,
    prob_5pct: float,
    prob_10pct: float,
    prob_direction: float,
    model_agreement: float,
    exp_return: float
) -> float:
    """
    Calculate overall signal strength (0-100 scale).
    
    Combines multiple factors into a single strength score.
    """
    # Base probability component (weighted average)
    prob_score = (
        prob_2pct * 0.2 +
        prob_5pct * 0.4 +
        prob_10pct * 0.3 +
        prob_direction * 0.1
    )
    
    # Expected return component (scaled to 0-1)
    ret_magnitude = min(abs(exp_return) / 0.15, 1.0)  # Cap at 15% expected
    
    # Combine with model agreement
    raw_strength = (
        prob_score * 0.5 +
        model_agreement * 0.3 +
        ret_magnitude * 0.2
    )
    
    # Scale to 0-100
    return min(max(raw_strength * 100, 0), 100)


def score_latest(df: pd.DataFrame, mode: str = "balanced") -> Dict:
    """
    Score the latest data point with comprehensive predictions.
    
    Args:
        df: OHLCV DataFrame for a single ticker
        mode: Prediction mode ("conservative", "balanced", "aggressive")
        
    Returns:
        Dictionary with all prediction metrics
    """
    # Build features
    X = make_features(df).iloc[-1:]
    
    # Get mode adjustments
    adj = _get_mode_adjustments(mode)
    
    # Load uncertainty parameters
    unc = _load_unc()
    
    # Initialize results
    result = {
        "prob_2pct": 0.5,
        "prob_5pct": 0.5,
        "prob_10pct": 0.5,
        "prob_direction": 0.5,
        "prob": 0.5,  # Legacy compatibility (5% prob)
        "exp_return": 0.0,
        "pi_low": 0.0,
        "pi_high": 0.0,
        "pi_width": 0.0,
        "confidence": 0.5,
        "tau": 0.85,
        "decision": "UNSURE",
        "soft_decision": "HOLD",
        "soft_strength": "NEUTRAL",
        "reason": "Models not loaded",
        "signal_strength": 0.0,
        "model_agreement": 0.0,
        "mode": mode
    }
    
    try:
        # ===== LOAD MODELS =====
        # Try new multi-threshold models first, fall back to legacy
        
        # 5% models (legacy compatible)
        lgb_5pct = _load_model("models/lgb_clf_5pct.pkl") or _load_model("models/lgb_clf.pkl")
        xgb_5pct = _load_model("models/xgb_clf_5pct.pkl") or _load_model("models/xgb_clf.pkl")
        rf_5pct = _load_model("models/rf_clf_5pct.pkl")
        cat_5pct = _load_model("models/cat_clf_5pct.pkl")
        meta_5pct = _load_model("models/meta_clf_5pct.pkl") or _load_model("models/meta_logit.pkl")
        
        # 2% models
        lgb_2pct = _load_model("models/lgb_clf_2pct.pkl")
        meta_2pct = _load_model("models/meta_clf_2pct.pkl")
        
        # 10% models
        lgb_10pct = _load_model("models/lgb_clf_10pct.pkl")
        meta_10pct = _load_model("models/meta_clf_10pct.pkl")
        
        # Direction models
        lgb_dir = _load_model("models/lgb_direction.pkl")
        xgb_dir = _load_model("models/xgb_direction.pkl")
        meta_dir = _load_model("models/meta_direction.pkl")
        
        # Regression model
        reg = _load_model("models/lgb_reg.pkl")
        
        if lgb_5pct is None or xgb_5pct is None:
            result["reason"] = "Core models not found"
            return result
        
        # ===== PREDICTIONS =====
        
        # 5% predictions (core)
        p_lgb_5 = lgb_5pct.predict_proba(X)[:, 1][0]
        p_xgb_5 = xgb_5pct.predict_proba(X)[:, 1][0]
        p_rf_5 = rf_5pct.predict_proba(X)[:, 1][0] if rf_5pct else p_lgb_5
        p_cat_5 = cat_5pct.predict_proba(X)[:, 1][0] if cat_5pct else p_xgb_5
        
        if meta_5pct:
            P_5 = pd.DataFrame({
                "p_lgb": [p_lgb_5],
                "p_xgb": [p_xgb_5],
                "p_rf": [p_rf_5]
            })
            if cat_5pct:
                P_5["p_cat"] = [p_cat_5]
            # Check expected features
            try:
                prob_5pct = meta_5pct.predict_proba(P_5)[:, 1][0]
            except:
                prob_5pct = (p_lgb_5 + p_xgb_5 + p_rf_5 + p_cat_5) / 4
        else:
            prob_5pct = (p_lgb_5 + p_xgb_5) / 2
        
        result["prob_5pct"] = prob_5pct
        result["prob"] = prob_5pct  # Legacy
        
        # 2% predictions
        if lgb_2pct:
            p_2 = lgb_2pct.predict_proba(X)[:, 1][0]
            result["prob_2pct"] = p_2
        else:
            result["prob_2pct"] = min(prob_5pct * 1.3, 0.95)  # Estimate
        
        # 10% predictions
        if lgb_10pct:
            p_10 = lgb_10pct.predict_proba(X)[:, 1][0]
            result["prob_10pct"] = p_10
        else:
            result["prob_10pct"] = max(prob_5pct * 0.6, 0.05)  # Estimate
        
        # Direction predictions
        if lgb_dir and xgb_dir:
            p_dir_lgb = lgb_dir.predict_proba(X)[:, 1][0]
            p_dir_xgb = xgb_dir.predict_proba(X)[:, 1][0]
            
            if meta_dir:
                P_dir = pd.DataFrame({"p_lgb": [p_dir_lgb], "p_xgb": [p_dir_xgb]})
                try:
                    prob_dir = meta_dir.predict_proba(P_dir)[:, 1][0]
                except:
                    prob_dir = (p_dir_lgb + p_dir_xgb) / 2
            else:
                prob_dir = (p_dir_lgb + p_dir_xgb) / 2
            
            result["prob_direction"] = prob_dir
        else:
            result["prob_direction"] = 0.55 if prob_5pct > 0.5 else 0.45
        
        # ===== EXPECTED RETURN =====
        if reg:
            exp_return = float(reg.predict(X)[0])
        else:
            # Estimate based on probabilities
            exp_return = (prob_5pct - 0.5) * 0.10
        
        result["exp_return"] = exp_return
        
        # ===== PREDICTION INTERVAL =====
        q_hat = float(unc.get("q_hat", 0.05))
        alpha = float(unc.get("alpha", 0.2))
        
        result["pi_low"] = exp_return - q_hat
        result["pi_high"] = exp_return + q_hat
        result["pi_width"] = 2 * q_hat
        
        # ===== MODEL AGREEMENT =====
        predictions = [p_lgb_5, p_xgb_5, p_rf_5, p_cat_5]
        pred_std = np.std(predictions)
        model_agreement = max(0, 1 - pred_std * 3)  # Lower std = higher agreement
        result["model_agreement"] = model_agreement
        
        # ===== CONFIDENCE & THRESHOLD =====
        conf = max(prob_5pct, 1 - prob_5pct)
        tau_base = float(unc.get("selective_threshold", 0.85))
        tau = tau_base * adj["tau_mult"]
        
        result["confidence"] = conf
        result["tau"] = tau
        
        # ===== SIGNAL STRENGTH =====
        strength = calculate_signal_strength(
            result["prob_2pct"],
            result["prob_5pct"],
            result["prob_10pct"],
            result["prob_direction"],
            model_agreement,
            exp_return
        ) * adj["strength_mult"]
        
        result["signal_strength"] = min(strength, 100)
        
        # ===== DECISION LOGIC =====
        
        # Soft decision (always provides a recommendation)
        if exp_return > 0.01 and result["prob_direction"] > 0.5:
            result["soft_decision"] = "LONG"
            if strength > 70:
                result["soft_strength"] = "STRONG"
            elif strength > 40:
                result["soft_strength"] = "MODERATE"
            else:
                result["soft_strength"] = "WEAK"
        elif exp_return < -0.01 and result["prob_direction"] < 0.5:
            result["soft_decision"] = "SHORT"
            if strength > 70:
                result["soft_strength"] = "STRONG"
            elif strength > 40:
                result["soft_strength"] = "MODERATE"
            else:
                result["soft_strength"] = "WEAK"
        else:
            result["soft_decision"] = "HOLD"
            result["soft_strength"] = "NEUTRAL"
        
        # Hard decision (with abstention)
        if conf < tau:
            result["decision"] = "UNSURE"
            result["reason"] = f"Confidence ({conf:.2f}) below threshold ({tau:.2f})"
        elif result["pi_low"] <= 0 <= result["pi_high"]:
            result["decision"] = "UNSURE"
            result["reason"] = "Prediction interval crosses zero"
        elif model_agreement < adj["min_agreement"]:
            result["decision"] = "UNSURE"
            result["reason"] = f"Low model agreement ({model_agreement:.2f})"
        else:
            result["decision"] = "LONG" if prob_5pct >= 0.5 else "SHORT"
            result["reason"] = f"High confidence ({conf:.2f}) + clear interval"
        
    except Exception as e:
        result["reason"] = f"Inference error: {str(e)[:50]}"
    
    return result


def score_batch(dfs: Dict[str, pd.DataFrame], mode: str = "balanced") -> pd.DataFrame:
    """
    Score multiple tickers at once.
    
    Args:
        dfs: Dictionary of {ticker: DataFrame}
        mode: Prediction mode
        
    Returns:
        DataFrame with predictions for all tickers
    """
    results = []
    
    for ticker, df in dfs.items():
        try:
            score = score_latest(df, mode=mode)
            score["ticker"] = ticker
            results.append(score)
        except Exception as e:
            print(f"Error scoring {ticker}: {e}")
            continue
    
    return pd.DataFrame(results)
