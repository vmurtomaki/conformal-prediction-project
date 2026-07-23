import streamlit as st
import pandas as pd
from visualizations import plot_conformal_intervals

# Must be the first Streamlit command
st.set_page_config(page_title="Conformal Forecasting", layout="wide")

@st.cache_data
def load_data():
    # Using a relative path assuming the app is run from the project root
    return pd.read_parquet("data/02_processed/conformal_results.parquet")

def main():
    st.title("Probabilistic Time Series Forecasting")
    st.markdown("### Enterprise Uncertainty Quantification via Conformal Prediction")
    
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Conformal results not found. Please run src/conformal_engine.py first.")
        return

    # Sidebar Controls
    st.sidebar.header("Dashboard Controls")
    st.sidebar.markdown("Filter the test set window to observe local coverage.")
    
    # Date filtering
    min_date, max_date = df.index.min().date(), df.index.max().date()
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    filtered_df = df.loc[mask]

    if filtered_df.empty:
        st.warning("No data selected in this date range.")
        return

    # Calculate live metrics
    # Empirical coverage: % of time true value falls within bounds
    is_covered = (filtered_df['true_value'] >= filtered_df['lower_bound']) & \
                 (filtered_df['true_value'] <= filtered_df['upper_bound'])
    empirical_coverage = is_covered.mean()
    
    mean_width = (filtered_df['upper_bound'] - filtered_df['lower_bound']).mean()
    
    # KPI Dashboard
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Confidence Level", "90.00%")
    with col2:
        delta_cov = empirical_coverage - 0.90
        st.metric("Empirical Coverage", f"{empirical_coverage:.2%}", f"{delta_cov:+.2%}")
    with col3:
        st.metric("Mean Interval Width", f"{mean_width:.2f}")

    st.markdown("---")
    
    # Visualization
    fig = plot_conformal_intervals(filtered_df)
    st.plotly_chart(fig, use_container_width=True)

    # Raw Data Expander
    with st.expander("View Raw Conformal Output"):
        st.dataframe(filtered_df)

if __name__ == "__main__":
    main()