import streamlit as st
import pandas as pd
import copy
import joblib
from visualizations import plot_conformal_intervals
from src.data_processing import apply_synthetic_shock, create_features, resample_and_impute, load_and_clean_data
from src.conformal_engine import calibrate_mapie_model, run_conformal_inference

st.set_page_config(page_title="Dynamic Conformal Forecasting", layout="wide")

@st.cache_resource(show_spinner="Orchestrating Base Engine Operations...")
def initialize_system():
    """
    Executes intensive I/O operations and model calibration precisely once per session.
    Caches the complex MAPIE object to prevent severe latency degradation.
    """
    raw_df = load_and_clean_data("data/01_raw/LD2011_2014.txt")
    processed_df = resample_and_impute(raw_df)
    
    # Isolate continuous 1D timeline to facilitate future leakage-free shock injection
    raw_target_series = processed_df['MT_320']
    
    df_ml = create_features(processed_df, "MT_320")
    train_ratio = 0.8
    split_idx = int(len(df_ml) * train_ratio)
    train = df_ml.iloc[:split_idx]
    
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    
    base_model = joblib.load("data/02_processed/base_model.pkl")
    
    cached_mapie = calibrate_mapie_model(base_model, X_train, y_train, n_blocks=15)
    test_start_date = df_ml.index[split_idx]
    
    return cached_mapie, raw_target_series, test_start_date

def main() -> None:
    st.title("Probabilistic Time Series Forecasting")
    st.markdown("### Enterprise Uncertainty Quantification via Conformal Prediction")
    
    try:
        cached_mapie, raw_target_series, test_start_date = initialize_system()
    except Exception as e:
        st.error(f"Framework Initialization Failure: {str(e)}")
        return

    st.sidebar.header("Algorithmic Optimization Controls")
    
    # Enforce execution barrier to neutralize UI thread locking
    with st.sidebar.form("conformal_config"):
        st.markdown("**Risk Tolerance Limits (Alpha)**")
        alpha_val = st.slider("Target Miscoverage Rate", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
        
        st.markdown("**Algorithmic Reactivity (Gamma)**")
        gamma_val = st.slider("Adaptive Step Size", min_value=0.00, max_value=0.20, value=0.05, step=0.01)
        
        st.markdown("---")
        st.markdown("**Synthetic Exogenous Shock Simulator**")
        min_date, max_date = test_start_date.date(), raw_target_series.index.max().date()
        shock_start = st.date_input("Shock Start Vector", value=min_date, min_value=min_date, max_value=max_date)
        shock_end = st.date_input("Shock End Vector", value=max_date, min_value=min_date, max_value=max_date)
        shock_multiplier = st.number_input("Demand Modification Scale", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        
        submitted = st.form_submit_button("Execute Probabilistic Inference")

    if not submitted:
        st.info("👈 System active. Define exogenous parameters and engage the inference loop.")
        return

    with st.spinner("Processing dynamic temporal equations..."):
        # Deep copy operation prevents global mutation of the cached residual matrices
        working_model = copy.deepcopy(cached_mapie)
        
        # Apply shock multiplier to the unbroken series prior to topological extraction
        shocked_df = apply_synthetic_shock(
            raw_target_series, 
            'MT_320',
            float(shock_multiplier), 
            str(shock_start), 
            str(shock_end)
        )
        
        df_ml = create_features(shocked_df, "MT_320")
        
        test_df = df_ml.loc[df_ml.index >= test_start_date]
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        results_df = run_conformal_inference(
            working_model, 
            X_test, 
            y_test, 
            base_alpha=alpha_val, 
            gamma=gamma_val, 
            step_size=24
        )

    is_covered = (results_df['true_value'] >= results_df['lower_bound']) & \
                 (results_df['true_value'] <= results_df['upper_bound'])
    empirical_coverage = is_covered.mean()
    mean_width = (results_df['upper_bound'] - results_df['lower_bound']).mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Defined Confidence Threshold", f"{(1 - alpha_val):.2%}")
    with col2:
        delta_cov = empirical_coverage - (1 - alpha_val)
        st.metric("Realized Empirical Coverage", f"{empirical_coverage:.2%}", f"{delta_cov:+.2%}")
    with col3:
        st.metric("Average Boundary Width", f"{mean_width:.2f}")

    st.markdown("---")
    
    fig = plot_conformal_intervals(results_df, shock_start, shock_end, float(shock_multiplier))
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()