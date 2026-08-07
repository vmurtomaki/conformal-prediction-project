# Probabilistic Time Series Forecasting with Conformal Prediction

## Overview

This repository implements an advanced probabilistic time series forecasting pipeline utilizing Conformal Prediction. By leveraging Ensemble Batch Prediction Intervals (EnbPI) and Adaptive Conformal Inference (ACI), the pipeline generates distribution-free, mathematically guaranteed prediction intervals that dynamically adapt to real-world volatility and concept drift.

The core mathematical guarantee provided is the marginal coverage:

$$P(y \in \Gamma_{1-\alpha}(x)) \ge 1 - \alpha$$

## Architecture

Built on a strict, modern Python enterprise standard:

* **Environment & Lockfile:** Managed deterministically via `uv` (Rust-based).
* **Modeling Engine:** Scikit-learn integrated with the `mapie` framework for robust conformal wrapping and block bootstrapping.
* **UI/UX Deployment:** Fully interactive Streamlit dashboard allowing real-time simulated interventions and risk-tolerance adjustments.
* **Static Analysis:** Aggressive linting and formatting via `Ruff`; strict static type checking via `MyPy`.

## Directory Structure

```text
Conformal_Prediction_Project/
├── uv.lock                 # Deterministic dependency graph
├── pyproject.toml          # Declarative configuration
├── Makefile                # Orchestration commands
├── app/                    # Interactive UI layer
│   ├── main.py             # Streamlit application
│   └── visualizations.py   # Plotly/Altair charting
├── config/                 # Centralized hyperparameters
├── src/                    # Core pipeline modules
│   ├── config.py           # Config loader
│   ├── conformal_engine.py # MAPIE, EnbPI, and ACI logic
│   ├── data_processing.py  # Resampling and feature engineering
│   └── main.py             # CLI execution entry point
├── tests/                  # Pytest suite validating mathematical bounds
└── Dockerfile              # Containerization instructions
```

## Quick Start

This project requires **Python 3.12+** and **uv**. The virtual environment is provisioned and managed automatically.

1. **Install uv:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone & Sync:**

```bash
git clone <repo_url> && cd Conformal_Prediction_Project
uv sync --all-extras --dev
```

3. **Execute Pipeline:**

```bash
# Run the test suite
make test

# Launch the interactive scenario simulator
uv run streamlit run app/main.py
```

## Technical Roadmap & Known Limitations

To maintain prototyping velocity and inference speed, specific theoretical trade-offs were made. These establish the roadmap for productionization:

* **Absence of Exact Conditional Coverage:** The pipeline guarantees marginal coverage over the dataset but cannot mathematically guarantee finite-sample conditional coverage without strict parametric assumptions. Localized coverage gaps may occasionally occur during severe volatility.
* **Sub-optimal Interval Efficiency:** The current methodology relies on absolute residual nonconformity scores, producing symmetric prediction intervals. Future iterations will integrate Conformalized Quantile Regression (CQR) to dynamically adapt interval widths to localized variance.
* **Statistical Inefficiency via Data Splitting:** The EnbPI block bootstrapping framework inherently reduces the effective training volume provided to the base estimator. More computationally expensive cross-conformal techniques (CV+, Jackknife+) were deliberately bypassed to ensure real-time inference speed.
* **Lack of Conformal Risk Control (CRC):** The system does not currently bound specific asymmetric operational risks or monotonic loss functions, limiting immediate applicability in highly risk-averse environments where controlling asymmetric costs supersedes generalized marginal coverage.