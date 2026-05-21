import pandas as pd, numpy as np, yaml, joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm
import os, json

from src.utils.io import load_df
from src.features.builder import make_features
from src.labels.labels import label_multi_threshold
from src.modelling.uncertainty import selective_threshold_from_oof, conformal_qhat_from_oof
import yfinance as yf

def load_xy_for(ticker: str):
    """Load features and labels for a single ticker."""
    import os
    import time
    import random
    import warnings
    from io import StringIO
    import sys
    
    warnings.filterwarnings('ignore')
    
    class SuppressStderr:
        def __init__(self):
            self.original_stderr = sys.stderr
        def __enter__(self):
            sys.stderr = StringIO()
            return self
        def __exit__(self, *args):
            sys.stderr = self.original_stderr
    
    parquet_path = f"raw/{ticker}.parquet"
    if os.path.exists(parquet_path):
        df = load_df(parquet_path)
    else:
        print(f"Downloading {ticker}...", end=" ", flush=True)
        df = None
        for attempt in range(3):
            try:
                with SuppressStderr():
                    if attempt == 0:
                        df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False, show_errors=False, threads=False)
                        if isinstance(df.columns, pd.MultiIndex):
                            df = df.droplevel(0, axis=1)
                    elif attempt == 1:
                        ticker_obj = yf.Ticker(ticker)
                        df = ticker_obj.history(period="2y", interval="1d", auto_adjust=True, timeout=60, raise_errors=False)
                    else:
                        from datetime import datetime, timedelta
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=730)
                        df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), 
                                       interval="1d", auto_adjust=True, progress=False, show_errors=False, threads=False)
                        if isinstance(df.columns, pd.MultiIndex):
                            df = df.droplevel(0, axis=1)
                
                if df is None or df.empty or len(df) < 50:
                    if attempt < 2:
                        time.sleep(3 + random.uniform(0, 2))
                        continue
                    raise ValueError(f"No data for {ticker}")
                
                df = df.rename(columns=str.title)
                if 'Volume' not in df.columns:
                    df['Volume'] = 0
                df = df.dropna(subset=['Close'])
                
                if len(df) < 50:
                    if attempt < 2:
                        time.sleep(3)
                        continue
                    raise ValueError(f"Insufficient data for {ticker}")
                
                os.makedirs("raw", exist_ok=True)
                df.to_parquet(parquet_path, compression='snappy', index=True)
                print(f"✓ {len(df)} rows")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"Retrying...", end=" ", flush=True)
                    time.sleep(3 + random.uniform(0, 2))
                else:
                    print(f"✗ Failed")
                    raise ValueError(f"Could not download data for {ticker}")
        
        if df is None or df.empty:
            raise ValueError(f"No data available for {ticker}")
    
    # Build features
    X = make_features(df)
    
    # Create multi-threshold labels
    y = label_multi_threshold(df)
    
    # Align indices
    common_idx = X.index.intersection(y.index)
    if len(common_idx) == 0:
        raise ValueError(f"No overlapping indices between features and labels for {ticker}")
    
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # Drop rows where labels are NaN
    y = y.dropna()
    X = X.loc[y.index]
    
    if len(X) < 100:
        raise ValueError(f"Insufficient data after processing for {ticker}: {len(X)} rows")
    
    return X, y


def oof_stack_fit(X: pd.DataFrame, y_target: pd.Series, n_splits: int = 5, seed: int = 42, threshold_name: str = "5pct"):
    """
    Out-of-fold stacking with multiple base models.
    Returns trained models and OOF predictions.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Enhanced ensemble with 4 models
    lgb = LGBMClassifier(n_estimators=600, learning_rate=0.03, subsample=0.8, 
                         colsample_bytree=0.8, random_state=seed, verbose=-1)
    xgb = XGBClassifier(n_estimators=700, learning_rate=0.03, subsample=0.8, 
                        colsample_bytree=0.8, max_depth=6, eval_metric="logloss", 
                        random_state=seed, verbosity=0)
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=seed, n_jobs=-1)
    
    # Try to use CatBoost if available
    try:
        cat = CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, 
                                 random_state=seed, verbose=False)
        use_catboost = True
    except:
        cat = None
        use_catboost = False

    oof_p_lgb = pd.Series(index=X.index, dtype=float)
    oof_p_xgb = pd.Series(index=X.index, dtype=float)
    oof_p_rf = pd.Series(index=X.index, dtype=float)
    oof_p_cat = pd.Series(index=X.index, dtype=float) if use_catboost else None

    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr = y_target.iloc[train_idx]
        
        lgb.fit(Xtr, ytr)
        xgb.fit(Xtr, ytr)
        rf.fit(Xtr, ytr)
        
        oof_p_lgb.iloc[test_idx] = lgb.predict_proba(Xte)[:, 1]
        oof_p_xgb.iloc[test_idx] = xgb.predict_proba(Xte)[:, 1]
        oof_p_rf.iloc[test_idx] = rf.predict_proba(Xte)[:, 1]
        
        if use_catboost:
            cat.fit(Xtr, ytr)
            oof_p_cat.iloc[test_idx] = cat.predict_proba(Xte)[:, 1]

    # Build stacking features
    P_dict = {"p_lgb": oof_p_lgb, "p_xgb": oof_p_xgb, "p_rf": oof_p_rf}
    if use_catboost:
        P_dict["p_cat"] = oof_p_cat
    
    P = pd.DataFrame(P_dict).dropna()
    
    # Use .loc to avoid reindex issues with duplicates
    y_aligned = y_target.loc[P.index].astype(int)
    
    # Meta learner
    meta = LogisticRegression(max_iter=500)
    meta.fit(P, y_aligned)

    # Refit all models on full data
    lgb.fit(X, y_target)
    xgb.fit(X, y_target)
    rf.fit(X, y_target)
    if use_catboost:
        cat.fit(X, y_target)

    models = {"lgb": lgb, "xgb": xgb, "rf": rf, "meta": meta}
    if use_catboost:
        models["cat"] = cat
    
    return models, P, y_aligned


def oof_direction_model(X: pd.DataFrame, y_direction: pd.Series, n_splits: int = 5, seed: int = 42):
    """
    Train a direction prediction model (up vs down).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    lgb = LGBMClassifier(n_estimators=500, learning_rate=0.03, subsample=0.8, 
                         colsample_bytree=0.8, random_state=seed, verbose=-1)
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.03, subsample=0.8, 
                        colsample_bytree=0.8, max_depth=5, eval_metric="logloss", 
                        random_state=seed, verbosity=0)

    oof_p_lgb = pd.Series(index=X.index, dtype=float)
    oof_p_xgb = pd.Series(index=X.index, dtype=float)

    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr = y_direction.iloc[train_idx]
        
        lgb.fit(Xtr, ytr)
        xgb.fit(Xtr, ytr)
        
        oof_p_lgb.iloc[test_idx] = lgb.predict_proba(Xte)[:, 1]
        oof_p_xgb.iloc[test_idx] = xgb.predict_proba(Xte)[:, 1]

    P = pd.DataFrame({"p_lgb": oof_p_lgb, "p_xgb": oof_p_xgb}).dropna()
    y_aligned = y_direction.loc[P.index].astype(int)
    
    meta = LogisticRegression(max_iter=300)
    meta.fit(P, y_aligned)

    lgb.fit(X, y_direction)
    xgb.fit(X, y_direction)

    return {"lgb": lgb, "xgb": xgb, "meta": meta}, P, y_aligned


def oof_regressor(X: pd.DataFrame, y_fwd: pd.Series, mask_big: pd.Series, n_splits: int = 5, seed: int = 42):
    """
    Regression model for expected return prediction.
    """
    reg = LGBMRegressor(n_estimators=700, learning_rate=0.03, subsample=0.8, 
                        colsample_bytree=0.8, random_state=seed, verbose=-1)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_pred = pd.Series(index=X.index, dtype=float)
    
    for train_idx, test_idx in tscv.split(X):
        tr_mask = mask_big.iloc[train_idx]
        Xtr, Xte = X.iloc[train_idx][tr_mask.values], X.iloc[test_idx]
        ytr = y_fwd.iloc[train_idx][tr_mask.values]
        if len(Xtr) < 50:
            continue
        reg.fit(Xtr, ytr)
        oof_pred.iloc[test_idx] = reg.predict(Xte)
    
    reg.fit(X[mask_big], y_fwd[mask_big])
    return reg, oof_pred


def main():
    print("=" * 70, flush=True)
    print("🚀 TRAINING ADVANCED QUANTITATIVE TRADING MODELS", flush=True)
    print("=" * 70, flush=True)
    
    uni = yaml.safe_load(open("config/universe.yaml"))
    tickers = uni["tickers"]
    print(f"\nLoading data for {len(tickers)} tickers...\n", flush=True)

    framesX, framesY = [], []
    successful_tickers = []
    
    for tk in tqdm(tickers, desc="Building panels"):
        try:
            X, y = load_xy_for(tk)
            if X.empty or y.empty:
                print(f"\n⚠️  {tk}: Empty data, skipping")
                continue
            X["ticker"] = tk
            y["ticker"] = tk
            framesX.append(X)
            framesY.append(y)
            successful_tickers.append(tk)
        except Exception as e:
            print(f"\n⚠️  {tk}: {str(e)[:50]}")
            continue
    
    if len(framesX) == 0:
        raise ValueError("No data loaded for any ticker!")
    
    print(f"\n✅ Loaded data for {len(successful_tickers)}/{len(tickers)} tickers", flush=True)
    print(f"   Successful: {', '.join(successful_tickers[:10])}{'...' if len(successful_tickers) > 10 else ''}", flush=True)
    
    print(f"\nCombining data panels...", flush=True)
    Xall = pd.concat(framesX).dropna()
    yall = pd.concat(framesY)
    
    # Reset index to avoid duplicate label issues
    Xall = Xall.reset_index(drop=True)
    yall = yall.reset_index(drop=True)
    
    # Align indices
    common_idx = Xall.index.intersection(yall.index)
    Xall = Xall.loc[common_idx]
    yall = yall.loc[common_idx].dropna()
    Xall = Xall.loc[yall.index]
    
    X = Xall.drop(columns=["ticker"])
    
    # Extract all labels
    y_2pct = yall["y_move_2pct"]
    y_5pct = yall["y_move_5pct"]
    y_10pct = yall["y_move_10pct"]
    y_direction = yall["y_direction"]
    y_fwd = yall["y_fwd"]
    
    print(f"   Total samples: {len(X)}", flush=True)
    print(f"   Features: {X.shape[1]}", flush=True)
    print(f"   2%+ moves: {y_2pct.sum()} ({y_2pct.mean()*100:.1f}%)", flush=True)
    print(f"   5%+ moves: {y_5pct.sum()} ({y_5pct.mean()*100:.1f}%)", flush=True)
    print(f"   10%+ moves: {y_10pct.sum()} ({y_10pct.mean()*100:.1f}%)", flush=True)

    os.makedirs("models", exist_ok=True)
    uncertainty_data = {}
    
    # ==========================================================================
    # TRAIN 2% MOVE MODEL
    # ==========================================================================
    print(f"\n{'='*70}", flush=True)
    print("📊 Training 2%+ Move Model (4-model ensemble)...", flush=True)
    print(f"{'='*70}", flush=True)
    
    models_2pct, P_oof_2pct, y_aligned_2pct = oof_stack_fit(X, y_2pct, threshold_name="2pct")
    oof_p_meta_2pct = pd.Series(models_2pct["meta"].predict_proba(P_oof_2pct)[:, 1], index=P_oof_2pct.index)
    
    epsilon_2pct = 0.20  # More lenient for 2% moves
    tau_2pct, meta_sel_2pct = selective_threshold_from_oof(y_aligned_2pct, oof_p_meta_2pct, epsilon=epsilon_2pct)
    
    # Save 2% models
    for name, model in models_2pct.items():
        joblib.dump(model, f"models/{name}_clf_2pct.pkl")
    print(f"  ✓ Saved 2%+ move models (tau={tau_2pct:.3f})", flush=True)
    
    uncertainty_data["2pct"] = {
        "epsilon": epsilon_2pct,
        "selective_threshold": tau_2pct,
        "selective_meta": meta_sel_2pct
    }
    
    # ==========================================================================
    # TRAIN 5% MOVE MODEL
    # ==========================================================================
    print(f"\n{'='*70}", flush=True)
    print("📊 Training 5%+ Move Model (4-model ensemble)...", flush=True)
    print(f"{'='*70}", flush=True)
    
    models_5pct, P_oof_5pct, y_aligned_5pct = oof_stack_fit(X, y_5pct, threshold_name="5pct")
    oof_p_meta_5pct = pd.Series(models_5pct["meta"].predict_proba(P_oof_5pct)[:, 1], index=P_oof_5pct.index)
    
    epsilon_5pct = 0.15
    tau_5pct, meta_sel_5pct = selective_threshold_from_oof(y_aligned_5pct, oof_p_meta_5pct, epsilon=epsilon_5pct)
    
    # Save 5% models
    for name, model in models_5pct.items():
        joblib.dump(model, f"models/{name}_clf_5pct.pkl")
    # Legacy compatibility
    joblib.dump(models_5pct["lgb"], "models/lgb_clf.pkl")
    joblib.dump(models_5pct["xgb"], "models/xgb_clf.pkl")
    joblib.dump(models_5pct["meta"], "models/meta_logit.pkl")
    print(f"  ✓ Saved 5%+ move models (tau={tau_5pct:.3f})", flush=True)
    
    uncertainty_data["5pct"] = {
        "epsilon": epsilon_5pct,
        "selective_threshold": tau_5pct,
        "selective_meta": meta_sel_5pct
    }
    
    # ==========================================================================
    # TRAIN 10% MOVE MODEL
    # ==========================================================================
    print(f"\n{'='*70}", flush=True)
    print("📊 Training 10%+ Move Model (4-model ensemble)...", flush=True)
    print(f"{'='*70}", flush=True)
    
    models_10pct, P_oof_10pct, y_aligned_10pct = oof_stack_fit(X, y_10pct, threshold_name="10pct")
    oof_p_meta_10pct = pd.Series(models_10pct["meta"].predict_proba(P_oof_10pct)[:, 1], index=P_oof_10pct.index)
    
    epsilon_10pct = 0.12  # Stricter for larger moves
    tau_10pct, meta_sel_10pct = selective_threshold_from_oof(y_aligned_10pct, oof_p_meta_10pct, epsilon=epsilon_10pct)
    
    # Save 10% models
    for name, model in models_10pct.items():
        joblib.dump(model, f"models/{name}_clf_10pct.pkl")
    print(f"  ✓ Saved 10%+ move models (tau={tau_10pct:.3f})", flush=True)
    
    uncertainty_data["10pct"] = {
        "epsilon": epsilon_10pct,
        "selective_threshold": tau_10pct,
        "selective_meta": meta_sel_10pct
    }
    
    # ==========================================================================
    # TRAIN DIRECTION MODEL
    # ==========================================================================
    print(f"\n{'='*70}", flush=True)
    print("📈 Training Direction Model (UP vs DOWN)...", flush=True)
    print(f"{'='*70}", flush=True)
    
    models_dir, P_oof_dir, y_aligned_dir = oof_direction_model(X, y_direction)
    oof_p_meta_dir = pd.Series(models_dir["meta"].predict_proba(P_oof_dir)[:, 1], index=P_oof_dir.index)
    
    epsilon_dir = 0.45  # ~55% accuracy target for direction
    tau_dir, meta_sel_dir = selective_threshold_from_oof(y_aligned_dir, oof_p_meta_dir, epsilon=epsilon_dir)
    
    # Save direction models
    for name, model in models_dir.items():
        joblib.dump(model, f"models/{name}_direction.pkl")
    print(f"  ✓ Saved direction models (tau={tau_dir:.3f})", flush=True)
    
    uncertainty_data["direction"] = {
        "epsilon": epsilon_dir,
        "selective_threshold": tau_dir,
        "selective_meta": meta_sel_dir
    }
    
    # ==========================================================================
    # TRAIN REGRESSION MODEL
    # ==========================================================================
    print(f"\n{'='*70}", flush=True)
    print("📉 Training Regression Model (Expected Return)...", flush=True)
    print(f"{'='*70}", flush=True)
    
    mask_5pct = y_5pct == 1
    reg, oof_pred = oof_regressor(X, y_fwd, mask_5pct)
    y_fwd_aligned = y_fwd.loc[oof_pred.dropna().index]
    
    alpha = 0.2
    q_hat = conformal_qhat_from_oof(y_fwd_aligned, oof_pred.dropna(), alpha=alpha)
    
    joblib.dump(reg, "models/lgb_reg.pkl")
    print(f"  ✓ Saved regression model (q_hat={q_hat:.4f})", flush=True)
    
    uncertainty_data["regression"] = {
        "alpha": alpha,
        "q_hat": q_hat
    }
    
    # ==========================================================================
    # SAVE ALL UNCERTAINTY DATA
    # ==========================================================================
    with open("models/uncertainty.json", "w") as f:
        # Flatten for legacy compatibility + new structure
        flat_data = {
            "epsilon": epsilon_5pct,
            "selective_threshold": tau_5pct,
            "selective_meta": meta_sel_5pct,
            "alpha": alpha,
            "q_hat": q_hat,
            "models": uncertainty_data
        }
        json.dump(flat_data, f, indent=2)
    print("  ✓ Saved models/uncertainty.json", flush=True)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n{'='*70}", flush=True)
    print("✅ TRAINING COMPLETE!", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n📊 Model Summary:", flush=True)
    print(f"   ├── 2%+ Move Model:  tau={tau_2pct:.3f}, coverage={meta_sel_2pct.get('coverage',0)*100:.1f}%", flush=True)
    print(f"   ├── 5%+ Move Model:  tau={tau_5pct:.3f}, coverage={meta_sel_5pct.get('coverage',0)*100:.1f}%", flush=True)
    print(f"   ├── 10%+ Move Model: tau={tau_10pct:.3f}, coverage={meta_sel_10pct.get('coverage',0)*100:.1f}%", flush=True)
    print(f"   ├── Direction Model: tau={tau_dir:.3f}, coverage={meta_sel_dir.get('coverage',0)*100:.1f}%", flush=True)
    print(f"   └── Regression:      q_hat={q_hat:.4f} (~{(1-alpha)*100:.0f}% coverage)", flush=True)
    print(f"\nModels saved to models/ directory", flush=True)


if __name__ == "__main__":
    main()
