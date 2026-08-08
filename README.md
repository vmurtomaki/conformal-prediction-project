# Probabilistic Time Series Forecasting with Conformal Prediction

A production-minded analytical pipeline that wraps any Scikit-Learn regressor with **distribution-free, mathematically guaranteed uncertainty bounds** — turning point forecasts into actionable risk ranges for volatile demand signals.

> [!NOTE]
> _Add a screenshot of the Streamlit dashboard here, e.g._ `![Dashboard Preview](docs/screenshot.png)`

## Overview & Impact

- **Quantifies forecast risk, not just the forecast.** Produces prediction intervals with a guaranteed marginal coverage rate, independent of the base model's assumptions — critical for demand planning and capacity decisions.
- **Handles non-stationary, shock-prone series.** Adaptive Conformal Inference (ACI) automatically widens or tightens intervals in response to real-time coverage drift, so the system self-corrects after volatility events instead of silently degrading.
- **Built for sequential, live inference.** The conformal layer (EnbPI) updates its residual matrices as new ground truth arrives, preserving chronological order rather than relying on the exchangeability assumptions standard conformal methods require.
- **Interactive validation, not just a static report.** A Streamlit dashboard exposes live empirical coverage vs. the target confidence level, letting a stakeholder stress-test the model against a synthetic demand shock in real time.

## Tech Stack

`Python 3.12`, `Streamlit`, `MAPIE`, `Scikit-Learn`, `Pandas`, `NumPy`, `Plotly`, `uv`, `Docker`, `Pytest`, `Ruff`, `mypy`

## Engineering Rigor

- **Deterministic environments:** dependency resolution and installs are locked via `uv` (`uv.lock`), with the identical `uv sync --frozen` flow used in both local development and the Docker image — eliminating "works on my machine" drift.
- **Automated quality gates:** `Ruff` for linting/formatting and `mypy` for static typing are wired into the `Makefile`, alongside a `Pytest` suite (with coverage reporting) that isolates and regression-tests the core conformal math, not just the UI.
- **Config-driven experimentation:** hyperparameters (`alpha`, `gamma`, bootstrap block count) are centralized in `config/hyperparameters.yaml` and loaded through a single typed accessor, keeping tuning out of application code and CI/CD-friendly.

## Quick Start

```bash
# 1. Install uv (deterministic dependency manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and sync the locked environment
git clone <repo-url> && cd Conformal_Prediction_Project
uv sync --all-extras --dev

# 3. Launch the dashboard
uv run streamlit run app/main.py
```
_No local dataset or pre-trained model? The app falls back to a synthetic demand series automatically, so it runs out of the box on a cold clone._

## Project Architecture

```
Ingestion (data_processing.py) → Feature Engineering (lags, rolling stats)
        → Base Model Training (model_training.py, TimeSeriesSplit CV)
        → Conformal Calibration (conformal_engine.py — EnbPI + ACI)
        → Presentation Layer (app/main.py, Streamlit UI + live metrics)
```

| Path | Purpose |
|---|---|
| `src/data_processing.py` | Ingestion, resampling, imputation, feature extraction |
| `src/model_training.py` | Base regressor training and hyperparameter search |
| `src/conformal_engine.py` | EnbPI calibration and per-timestep ACI inference |
| `src/config.py` | Centralized hyperparameter loading |
| `app/` | Streamlit UI, controls, and visualizations |
| `tests/` | Pytest coverage for conformal math and data integrity |

## Trade-offs & Roadmap

- **Batched-chunk inference vs. true per-observation ACI:** the alpha update math runs per-timestep internally but is applied to the model in fixed-size chunks, trading a small amount of reactivity for lower latency and a UI that doesn't freeze on every observation.
- **Symmetric residual intervals:** EnbPI currently uses absolute residuals, producing symmetric bounds. Given non-linear variance in the target signal, the planned v2.0 move is Conformalized Quantile Regression (CQR) for asymmetric, heteroscedasticity-aware intervals.
- **Synchronous inference coupling:** inference currently runs in the Streamlit presentation thread (guarded by a deep copy of cached conformity scores). Decoupling this into a stateless inference service is the next architectural step to remove UI-thread memory overhead.