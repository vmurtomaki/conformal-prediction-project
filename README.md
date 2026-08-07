# Probabilistic Time Series Forecasting with Conformal Prediction

Enterprise-style Streamlit app for distribution-free uncertainty quantification on time series, using EnbPI and Adaptive Conformal Inference (ACI) on top of the MAPIE framework. Includes a live Coverage Metric Dashboard for real-time validation of theoretical guarantees.

## Why This Approach

Standard forecasting models capture data uncertainty but ignore model uncertainty, producing fragile confidence intervals. Conformal prediction fixes this with model-agnostic prediction regions that carry a mathematically guaranteed marginal coverage rate, computed from conformity scores on a calibration set — independent of the base learner's assumptions.

- **EnbPI (Ensemble Batch Prediction Intervals):** Time series break the exchangeability assumption standard conformal methods rely on. EnbPI uses block bootstrapping and leave-one-out ensemble aggregation to produce robust residuals while preserving chronological order, enabling sliding-window updates during live inference.
- **ACI (Adaptive Conformal Inference):** Handles non-stationarity and concept drift by tracking empirical coverage in real time and adjusting the significance level via gradient-descent stochastic approximation. Consecutive miscoverage events widen the intervals (via the gamma learning rate) so the system recovers from shocks automatically.

## Dataset

Built around the UCI Electricity Load Diagrams dataset, targeting volatile client profiles.

| Feature | Detail |
|---|---|
| Domain & Granularity | Portuguese client panel data, hourly resampled |
| Primary Complexity | High dimensionality, volatile client behavior |
| Data Quality Nuances | DST anomalies handled via continuous spline imputation |

## Implementation

Powered by `MapieTimeSeriesRegressor`, wrapping standard Scikit-Learn regressors. Partial fitting runs continuously during inference, so the model updates its residual matrices as ground truth is revealed sequentially — producing narrower, more precise intervals over time.

## Roadmap

1. **Environment:** Deterministic dependency locking via `uv`.
2. **Data pipeline:** Temporal resampling, imputation, lag feature extraction.
3. **Model tuning:** Sequential `TimeSeriesSplit` cross-validation.
4. **Conformal layer:** EnbPI block bootstrap + residual generation.
5. **Adaptive tuning:** ACI step-size calibration for interval reactivity.
6. **Deployment:** Streamlit dashboard, live metrics, containerization.

## Folder Structure

| Path | Purpose |
|---|---|
| `data/01_raw/` | Immutable original data; keeps the pipeline reproducible |
| `src/conformal_engine.py` | Isolated MAPIE/bootstrapping logic |
| `app/main.py` | Presentation layer — UI routing and state |
| `config/hyperparameters.yaml` | Centralized hyperparameters for CI/CD experimentation |

## Quick Start & Execution

Requires **Python 3.12+**. Uses [`uv`](https://astral.sh/uv) for deterministic dependency management and virtual environment isolation.

### 1. Environment bootstrap

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync the locked environment (includes dev/test dependencies)
uv sync --all-extras --dev
```

### 2. Validation & static analysis

```bash
# Run the full test suite
make test

# Run type checking and linting
make lint
```

### 3. Launch the dashboard

```bash
uv run streamlit run app/main.py
```

If the raw UCI data files aren't present locally, the app falls back to a synthetic autoregressive dataset automatically.

## Limitations (v1.0)

Trade-offs made to prioritize rapid prototyping and deployment. These define the roadmap for v2.0:

* **Batched ACI:** True ACI updates the target quantile after every observation. This engine batches inference in user-defined chunks instead — the gradient shift is computed per-timestep internally but applied per-block — trading micro-level reactivity for lower latency and no UI freezing.
* **Symmetric residuals:** EnbPI currently uses absolute residuals, giving symmetric intervals. Since electrical demand variance is non-linear, v2.0 will move to Conformalized Quantile Regression (CQR) with asymmetric pinball loss for local heteroscedasticity.
* **Monolithic inference coupling:** Inference runs synchronously in the presentation thread; a deep-copy guards the cached conformity scores from mutation. v2.0 will decouple this into a stateless microservice to remove UI-thread memory overhead.
* **Static block bootstrap:** Block size is currently hardcoded. Future versions will select it dynamically from the series' autocorrelation function to avoid breaking long-range dependencies.
* **Topological test coverage gap:** CI validates ACI bound validity (no inversion), but doesn't yet assert chronological expansion rates within a chunk — only endpoint validity. Closing this gap is next.