# Multi-Trading Agent

A production-oriented quantitative research platform that combines ensemble machine learning, regime-aware risk controls, and a multi-agent inference layer for systematic equity signal generation. The system spans two complementary horizons: a **30-day swing model** built on daily OHLCV features, and a **short-term multi-agent stack** operating on hourly data with specialized agents for trend, momentum, volatility, volume, and pattern recognition.

Designed as a research-to-signals pipeline rather than a black-box recommender, every output is decomposed into calibrated probabilities, explicit gate failures, position sizing logic, and optional LLM-assisted narrative explanations.

---

## Architecture

```
Market Data (yfinance)
        │
        ▼
┌───────────────────┐     ┌─────────────────────────────┐
│ Feature Engine    │     │ Short-Term Agent Layer      │
│ 54 cross-sectional│     │ Trend · Momentum · Vol ·    │
│ + vol + cross-    │     │ Volume · Pattern → Meta     │
│ asset signals     │     │ Agent (1d / 3d / 1w)        │
└─────────┬─────────┘     └──────────────┬──────────────┘
          │                              │
          ▼                              ▼
┌───────────────────┐     ┌─────────────────────────────┐
│ Ensemble Models   │     │ Hourly inference + stops /  │
│ LGBM · XGB · RF · │     │ take-profit / sizing        │
│ CatBoost + stack  │     └──────────────┬──────────────┘
└─────────┬─────────┘                    │
          │                              │
          ▼                              ▼
┌───────────────────┐          ┌─────────────────┐
│ Selective         │          │ Rulebook gates  │
│ classification +  │─────────▶│ + trade engine  │
│ conformal intervals          │ + portfolio opt │
└─────────┬─────────┘          └────────┬────────┘
          │                             │
          └──────────────┬──────────────┘
                         ▼
              Signals · Portfolio · Dashboards
```

### Design principles

| Layer | Responsibility |
|-------|----------------|
| **Feature engineering** | Point-in-time safe transforms: technicals, realized vol, ADX/Stochastic/MFI/Hurst, cross-asset beta and relative strength vs SPY |
| **Labeling** | Multi-threshold binary labels (2%, 5%, 10% forward moves) plus direction classification on a 30-day horizon |
| **Modelling** | Time-series cross-validated ensemble with a logistic meta-learner; separate return regressor for expected move magnitude |
| **Uncertainty** | Selective prediction thresholds (target ε error rate) and conformal prediction intervals for return estimates |
| **Decision logic** | Hard decisions (LONG / SHORT / UNSURE) gated by confidence; soft decisions always emit a directional lean |
| **Risk engine** | Per-name risk budget, trailing stops, scale-out targets, regime multipliers (low / normal / high vol) |
| **Portfolio** | Mean-variance optimization with turnover penalty and concentration limits |
| **Agents** | Short-horizon specialist agents aggregated by a meta-agent; optional Gemini narrative layer for signal context |

---

## Repository layout

```
├── config/universe.yaml          # Investable universe (20 large-cap names)
├── docs/rulebook.yaml            # Gates, decision thresholds, risk & regime params
├── src/
│   ├── features/                 # Feature builders (tech, vol, advanced, cross-asset)
│   ├── labels/                   # Multi-threshold label generation
│   ├── modelling/                # Training, inference, uncertainty calibration
│   ├── rules/                    # Rulebook loader + position sizing engine
│   ├── live/                     # Signal generation, portfolio, HTML export
│   └── agents/
│       ├── gemini_agent.py       # Optional LLM explanation layer
│       └── short_term/           # Multi-agent short-horizon system
├── risk_engine/                  # Optimizer + risk model
├── models/                       # Serialized estimators + calibration artifacts
├── data/                         # Latest signals and portfolio outputs
├── raw/                          # Cached parquet price history
├── web/                          # Static signal sheet
├── dashboard.py                  # Primary Streamlit dashboard (30d horizon)
├── dashboard_shortterm.py        # Short-term multi-agent dashboard
├── dashboard_longterm.py         # Long-horizon view
└── run_complete_pipeline.py      # End-to-end orchestrator
```

---

## Quick start

### Prerequisites

- Python 3.10+
- pip

```bash
pip install -r requirements.txt
```

### Run the full pipeline

```bash
python run_complete_pipeline.py
```

Common flags:

```bash
python run_complete_pipeline.py --mode conservative   # tighter confidence filters
python run_complete_pipeline.py --mode aggressive     # lower thresholds, more signals
python run_complete_pipeline.py --skip-training       # reuse existing model artifacts
python run_complete_pipeline.py --with-ai --api-key $GEMINI_API_KEY
python run_complete_pipeline.py --dashboard           # launch Streamlit after completion
```

### Run individual stages

```bash
python -m src.modelling.train              # train + calibrate
python -m src.live.run_signals             # generate signals
python -m src.live.portfolio_from_signals    # optimize weights
python -m src.live.render_sheet              # export HTML report
```

### Dashboards

```bash
streamlit run dashboard.py --server.port 8502
streamlit run dashboard_shortterm.py --server.port 8503
streamlit run dashboard_longterm.py --server.port 8504
```

If port 8501 is occupied, specify an alternate port as shown above.

---

## Signal schema

Each row in `data/signals_latest.csv` represents one name in the universe at the latest observation date.

| Field | Description |
|-------|-------------|
| `prob_2pct`, `prob_5pct_30d`, `prob_10pct` | Calibrated probability of exceeding move threshold within 30 sessions |
| `prob_direction` | P(up) from the direction classifier |
| `exp_30d_return` | Point estimate from the return regressor |
| `pi_low`, `pi_high` | Conformal 80% prediction interval bounds |
| `confidence` | Model confidence after selective thresholding |
| `signal_strength` | Composite score (0–100) weighting move prob, direction, expected return, and precision |
| `decision` | Hard label: LONG / SHORT / UNSURE |
| `soft_decision` | Directional lean regardless of hard gate |
| `action` | Actionable tag: BUY / SELL / HOLD / AVOID |
| `gates_pass` | Whether all rulebook filters cleared |
| `gates_reason` | Human-readable gate failure detail |

**Signal strength composition:** move probability (25%), direction confidence (25%), expected return magnitude (20%), model confidence (15%), interval precision (15%).

---

## Rulebook and risk configuration

All trading logic externalized in `docs/rulebook.yaml`:

**Technical gates** — price above 200 DMA, ADX ≥ 18, 60-day momentum ≥ 3%, ATR/Close ≤ 8%, minimum dollar volume proxy.

**Decision policy** — target misclassification rate ε = 0.12 on accepted predictions; minimum expected move 3%; 80% conformal intervals (α = 0.2).

**Risk** — 1% base risk per name, 1.5× ATR trailing stop, 2.0× hard stop, scale-out at +10%.

**Regime scaling** — position size and stop width adjusted across low / normal / high volatility regimes.

**Portfolio constraints** — 5% max single-name weight, 25% sector cap (when sector mapping available), turnover penalty λ = 0.001.

Edit `config/universe.yaml` to change the ticker set.

---

## Model stack

### Long-horizon (30-day)

- **Classifiers:** LightGBM, XGBoost, RandomForest, CatBoost → logistic stacking meta-learner
- **Separate heads:** 2%, 5%, 10% move thresholds + direction
- **Regressor:** LightGBM for expected 30-day return
- **Validation:** expanding-window time-series splits (no lookahead)
- **Calibration:** out-of-fold selective thresholds + conformal residual quantiles persisted to `models/uncertainty.json`

### Short-horizon (multi-agent)

Specialist agents each produce independent scores on hourly bars:

- **TrendAgent** — moving-average structure and slope
- **MomentumAgent** — rate-of-change and RSI dynamics
- **VolatilityAgent** — ATR regime and vol expansion/contraction
- **VolumeAgent** — volume profile and accumulation/distribution
- **PatternAgent** — candlestick and microstructure patterns
- **MetaAgent** — weighted fusion → 1-day, 3-day, 1-week forecasts with stop-loss, take-profit, and position size

---

## Feature set (54 inputs)

| Category | Examples |
|----------|----------|
| Technical (18) | SMA 20/50/100/200, distance from MA, RSI-14, MACD, Bollinger width, volume ratio |
| Volatility (11) | Realized vol 5/10/20/60d, ATR-14/20, vol-of-vol, downside vol |
| Advanced (15) | ADX, Stochastic, MFI, Hurst exponent, momentum ranks, z-scores |
| Cross-asset (10) | Relative strength vs SPY, rolling beta/alpha, correlation, regime flags |

All features are computed strictly from data available at decision time.

---

## Optional AI explanation layer

When a Gemini API key is provided, `src/agents/gemini_agent.py` generates structured narratives covering signal rationale, risk context, and execution considerations. This layer is **read-only** — it does not alter model scores or portfolio weights.

```bash
export GEMINI_API_KEY=your_key
python -m src.live.run_signals --api-key $GEMINI_API_KEY
```

If the API is unavailable, the pipeline falls back to deterministic summary text.

---

## Outputs

| Artifact | Path |
|----------|------|
| Trained models | `models/*.pkl` |
| Calibration params | `models/uncertainty.json` |
| Latest signals | `data/signals_latest.csv` |
| Portfolio weights | `data/portfolio_latest.csv` |
| Short-term signals | `data/short_term_signals.json` |
| HTML report | `web/signals.html` |
| Research figures | `figures/` |

---

## Development notes

- Raw prices cached as parquet under `raw/` to avoid repeated API calls during training.
- The `.env` file is gitignored; set `GEMINI_API_KEY` there or pass via CLI.
- Re-run `python -m src.modelling.train` after universe or feature changes.
- For reproducibility, pin dependency versions in production deployments.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute investment advice. Past model performance does not guarantee future results. Always validate signals independently and respect your firm's compliance requirements before deploying capital.

---

## License

MIT
