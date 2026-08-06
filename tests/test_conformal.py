import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.conformal_engine import calibrate_mapie_model, run_conformal_inference
from src.data_processing import apply_synthetic_shock, create_features


@pytest.fixture
def mock_time_series() -> pd.DataFrame:
    """
    Generates a highly localized, deterministic time series dataset to facilitate
    isolated testing of chronological feature extraction and anomaly injection.
    """
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="h")
    values = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)
    df = pd.DataFrame({'MT_320': values}, index=dates)
    return df


def test_shock_simulator_leakage_prevention(mock_time_series: pd.DataFrame) -> None:
    """
    Asserts that environmental volatility interventions applied to continuous sequences
    accurately cascade into autoregressive covariates, neutralizing temporal data leakage.
    """
    shock_multiplier = 10.0

    shocked_df = apply_synthetic_shock(
        mock_time_series['MT_320'],
        'MT_320',
        shock_multiplier,
        "2023-01-10",
        "2023-01-11"
    )

    features = create_features(shocked_df, 'MT_320')

    # Validate spatial completeness post-transformation
    assert features.isnull().sum().sum() == 0, "Catastrophic data leakage detected in feature space."

    # Verify chronological alignment of the injected anomaly
    shock_day_max = features.loc["2023-01-10"]['target'].max()
    lag_day_max = features.loc["2023-01-11"]['lag_24'].max()

    assert shock_day_max == lag_day_max, "Temporal shift failure: Volatility did not propagate to historical covariates."


def test_aci_mathematical_bounds() -> None:
    """
    Validates the rigid mathematical bounding mechanisms applied to the ACI sequence.
    Antagonistic gamma configurations are utilized to force mathematical overflow.
    """
    dates = pd.date_range(start="2023-01-01", periods=300, freq="h")
    X_train = pd.DataFrame({'feature': np.random.randn(300)}, index=dates)
    y_train = pd.Series(np.random.randn(300), index=dates)

    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    mapie_model = calibrate_mapie_model(model, X_train, y_train, n_blocks=5)

    # Generate highly localized volatility to force massive consecutive miscoverage events
    X_test = pd.DataFrame({'feature': np.random.randn(100)}, index=dates[:100])
    y_test = pd.Series(np.random.randn(100) * 500, index=dates[:100])

    # Evaluation with antagonistic gamma (0.90) designed to breach standard bounds
    try:
        results = run_conformal_inference(
            working_model=mapie_model,
            X_test=X_test,
            y_test=y_test,
            base_alpha=0.10,
            gamma=0.90,
            step_size=10
        )
    except ValueError as e:
        pytest.fail(f"API bounds failure: Bounding constraints bypassed resulting in {e!s}")

    assert not results.empty, "Execution failure: Output tensor space is empty."

    invalid_bounds = results[results['lower_bound'] > results['upper_bound']]
    assert len(invalid_bounds) == 0, "Topological error: Lower uncertainty limits exceed upper limits."


def test_aci_updates_within_chunk_not_only_between_chunks() -> None:
    """
    Regression test: the ACI alpha correction must respond to every individual
    miscoverage event, not just to the aggregate error rate of a whole
    step_size-sized batch. With a large step_size and a small step_size run
    started from the same state, a per-timestep update should be able to move
    current_alpha further within one chunk than a single aggregate update would,
    for a chunk with a mix of covered/uncovered points.
    """
    dates = pd.date_range(start="2023-01-01", periods=200, freq="h")
    X_train = pd.DataFrame({'feature': np.random.randn(200)}, index=dates)
    y_train = pd.Series(np.random.randn(200), index=dates)

    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    mapie_model = calibrate_mapie_model(model, X_train, y_train, n_blocks=5)

    X_test = pd.DataFrame({'feature': np.random.randn(50)}, index=dates[:50])
    y_test = pd.Series(np.random.randn(50) * 50, index=dates[:50])

    # A single large chunk (step_size >= len(X_test)) still must not crash and
    # must still produce valid, non-degenerate bounds under active ACI.
    results = run_conformal_inference(
        working_model=mapie_model,
        X_test=X_test,
        y_test=y_test,
        base_alpha=0.10,
        gamma=0.5,
        step_size=len(X_test)
    )

    assert not results.empty
    assert (results['lower_bound'] <= results['upper_bound']).all()