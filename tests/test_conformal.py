import pandas as pd
import os

def test_feature_matrix_integrity():
    """Validates that the ML feature matrix contains no nulls and has the correct columns."""
    filepath = "data/02_processed/ml_features_MT_320.parquet"
    assert os.path.exists(filepath), "Feature matrix file is missing!"
    
    df = pd.read_parquet(filepath)
    assert df.isnull().sum().sum() == 0, "Data leakage/Nulls detected in features!"
    assert 'target' in df.columns, "Target column is missing!"
    assert 'lag_24' in df.columns, "Temporal lags were not engineered correctly!"

def test_conformal_output_bounds():
    """Validates the mathematical logic that the upper bound is always >= lower bound."""
    filepath = "data/02_processed/conformal_results.parquet"
    assert os.path.exists(filepath), "Conformal results file is missing!"
    
    df = pd.read_parquet(filepath)
    # Check that intervals are mathematically logical
    invalid_bounds = df[df['lower_bound'] > df['upper_bound']]
    assert len(invalid_bounds) == 0, "Mathematical error: Lower bound exceeds upper bound!"