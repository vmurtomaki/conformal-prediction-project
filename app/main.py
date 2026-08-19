import copy
import sys
from pathlib import Path

# Fix path for 'src' module discovery when running via Streamlit
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from visualizations import plot_conformal_intervals

from src.config import load_config
from src.conformal_engine import calibrate_mapie_model, run_conformal_inference
from src.data_processing import (
    apply_synthetic_shock,
    create_features,
    load_and_clean_data,
    resample_and_impute,
)

st.set_page_config(page_title="Dynamic Conformal Forecasting", layout="wide")

# Load centralized hyperparameter configuration
CONFIG = load_config()

# Pass the parameter into the function so Streamlit binds the cache to the config value.
@st.cache_resource(show_spinner="Loading data and calibrating model...")
def initialize_system(bootstrap_estimators: int):
    """
    Loads the data and calibrates the MAPIE model once per session.

    Cached because calibration is the slow step; the cache key is
    bootstrap_estimators, so changing it in the YAML rebuilds the model.
    Returns (calibrated model, raw target series, test-set start timestamp).
    """
    # 1. Fallback for Data Ingestion on Cold Clones
    data_path = Path("data/01_raw/LD2011_2014.txt")
    if not data_path.exists():
        st.warning("Raw data not found. Using a synthetic demand series so the app runs on a cold clone.")
        dates = pd.date_range(start="2023-01-01", periods=2000, freq="h")
        raw_target_series = pd.Series(
            np.sin(np.linspace(0, 100, 2000)) * 50 + 100 + np.random.normal(0, 5, 2000), 
            index=dates, 
            name="MT_320"
        )
        processed_df = raw_target_series.to_frame()
    else:
        raw_df = load_and_clean_data(str(data_path))
        processed_df = resample_and_impute(raw_df)
        raw_target_series = processed_df['MT_320']
    
    df_ml = create_features(processed_df, "MT_320")
    train_ratio = 0.8
    split_idx = int(len(df_ml) * train_ratio)
    train = df_ml.iloc[:split_idx]
    
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    
    # 2. Fallback for Model Ingestion on Cold Clones
    model_path = Path("data/02_processed/base_model.pkl")
    if not model_path.exists():
        from sklearn.ensemble import RandomForestRegressor
        st.warning("Trained base model not found. Fitting a default RandomForestRegressor.")
        base_model = RandomForestRegressor(n_estimators=10, random_state=42)
    else:
        base_model = joblib.load(model_path)
    
    # The cache is now bound to the parameter. If updated in YAML, it will rebuild.
    cached_mapie = calibrate_mapie_model(
        base_model, X_train, y_train, n_blocks=bootstrap_estimators
    )
    test_start_date = df_ml.index[split_idx]
    
    return cached_mapie, raw_target_series, test_start_date


def main() -> None:
    st.title("Probabilistic Time Series Forecasting")
    st.markdown("### Distribution-free prediction intervals for volatile time series")
    
    try:
        # Pass the config parameter dynamically to bind the cache state
        cached_mapie, raw_target_series, test_start_date = initialize_system(CONFIG["bootstrap_estimators"])
    except Exception as e:  # noqa: BLE001
        st.error(f"Initialization failed: {e!s}")
        return
    st.sidebar.header("Model Parameters")
    
    # A form defers reruns until submit, so moving a slider doesn't trigger inference
    with st.sidebar.form("conformal_config"):
        st.markdown("**Target miscoverage (alpha)**")
        alpha_val = st.slider(
            "Target Miscoverage Rate", min_value=0.01, max_value=0.50, 
            value=CONFIG["alpha"], step=0.01
        )
        
        st.markdown("**ACI step size (gamma)**")
        gamma_val = st.slider(
            "Adaptive Step Size", min_value=0.00, max_value=0.20, 
            value=CONFIG["gamma"], step=0.01
        )
        
        st.markdown("---")
        st.markdown("**Synthetic demand shock**")
        min_date, max_date = test_start_date.date(), raw_target_series.index.max().date()
        shock_start = st.date_input("Shock Start Date", value=min_date, min_value=min_date, max_value=max_date)
        shock_end = st.date_input("Shock End Date", value=max_date, min_value=min_date, max_value=max_date)
        shock_multiplier = st.number_input("Demand Multiplier", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        
        submitted = st.form_submit_button("Run Forecast")
    if not submitted:
        st.info("👈 Set the parameters in the sidebar, then run the forecast.")
        return
    with st.spinner("Running inference..."):
        # Deep copy: run_conformal_inference calls update(), which mutates the
        # cached model's conformity scores and would leak across sessions.
        working_model = copy.deepcopy(cached_mapie)
        
        # Apply shock multiplier to evaluation series before interval calculation.
        shocked_df = apply_synthetic_shock(
            raw_target_series, 
            'MT_320',
            float(shock_multiplier), 
            str(shock_start), 
            str(shock_end)
        )
        
        df_ml = create_features(shocked_df, "MT_320")
        
        # Cap evaluation window to prevent high memory usage.
        MAX_INFERENCE_ROWS = 1500
        test_df = df_ml.loc[df_ml.index >= test_start_date].tail(MAX_INFERENCE_ROWS)
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        results_df = run_conformal_inference(
            working_model, 
            X_test, 
            y_test, 
            base_alpha=alpha_val, 
            gamma=gamma_val, 
            step_size=168
        )

    is_covered = (results_df['true_value'] >= results_df['lower_bound']) & \
                 (results_df['true_value'] <= results_df['upper_bound'])
    empirical_coverage = is_covered.mean()
    mean_width = (results_df['upper_bound'] - results_df['lower_bound']).mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target coverage", f"{(1 - alpha_val):.2%}")
    with col2:
        delta_cov = empirical_coverage - (1 - alpha_val)
        st.metric("Empirical coverage", f"{empirical_coverage:.2%}", f"{delta_cov:+.2%}")
    with col3:
        st.metric("Mean interval width", f"{mean_width:.2f}")

    st.markdown("---")
    
    fig = plot_conformal_intervals(results_df, shock_start, shock_end, float(shock_multiplier))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()