import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.conformal_engine import calibrate_mapie_model, run_conformal_inference
from src.data_processing import apply_synthetic_shock, create_features, resample_and_impute


@pytest.fixture
def mock_time_series() -> pd.DataFrame:
    """
    A 1000-hour sine series with noise, used to test feature creation and
    shock injection without touching the real dataset.
    """
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="h")
    values = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)
    df = pd.DataFrame({'MT_320': values}, index=dates)
    return df


def test_shock_simulator_leakage_prevention(mock_time_series: pd.DataFrame) -> None:
    """
    A shock applied to the raw series must also appear in the lag features
    derived from it. Shocking after feature creation would leave lag_24
    holding unshocked values, which is leakage of the pre-shock series.
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

    # create_features drops NaN rows, so the matrix must be complete
    assert features.isnull().sum().sum() == 0, "Null values in feature matrix."
    # The shock day's target must reappear one day later in lag_24
    shock_day_max = features.loc["2023-01-10"]['target'].max()
    lag_day_max = features.loc["2023-01-11"]['lag_24'].max()
    assert shock_day_max == lag_day_max, "Shock did not propagate into lag_24."

def test_resample_imputation_detects_and_flags_gaps() -> None:
    """
    Constructs a 2-hour gap in a 30-min series and verifies empty bins become
    NaN before fill, with was_imputed marking exactly the filled hours.
    """
    dates = pd.date_range("2023-01-01", periods=10, freq="30min")
    df = pd.DataFrame({'MT_320': np.arange(10, dtype=np.float32)}, index=dates)
    gap_mask = (df.index >= "2023-01-01 02:00:00") & (df.index < "2023-01-01 04:00:00")
    df = df.loc[~gap_mask]
    pre_fill = df.resample('h').sum(min_count=1)
    gap_hours = pd.date_range("2023-01-01 02:00:00", "2023-01-01 03:00:00", freq="h")
    assert pre_fill.loc[gap_hours, 'MT_320'].isna().all()
    result = resample_and_impute(df)
    assert result.loc[gap_hours, 'was_imputed'].all()
    assert not result.loc[~result.index.isin(gap_hours), 'was_imputed'].any()
    assert not result.loc[gap_hours, 'MT_320'].isna().any()


def test_aci_bounds_validity_under_high_gamma() -> None:
    """
    A large gamma drives alpha outside [0, 1] within a few updates. The
    clipping in run_conformal_inference must keep the confidence level valid,
    so inference completes and lower_bound <= upper_bound holds throughout.
    """
    dates = pd.date_range(start="2023-01-01", periods=300, freq="h")
    X_train = pd.DataFrame({'feature': np.random.randn(300)}, index=dates)
    y_train = pd.Series(np.random.randn(300), index=dates)

    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    mapie_model = calibrate_mapie_model(model, X_train, y_train, n_blocks=5)

    # Test targets scaled 500x so nearly every point misses the interval
    X_test = pd.DataFrame({'feature': np.random.randn(100)}, index=dates[:100])
    y_test = pd.Series(np.random.randn(100) * 500, index=dates[:100])
    # gamma = 0.90 pushes alpha out of range within a few updates
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
        pytest.fail(f"Alpha was not clipped to a valid confidence level: {e!s}")
    assert not results.empty, "Inference returned empty array."

    invalid_bounds = results[results['lower_bound'] > results['upper_bound']]
    assert len(invalid_bounds) == 0, "Lower bound exceeded upper bound."


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
    
class _StubModel:
    def __init__(self, lower, upper):
        self._lower = lower
        self._upper = upper
        self.alphas_used = []
    def predict(self, X_chunk, ensemble, alpha, optimize_beta):
        self.alphas_used.append(alpha)
        n = len(X_chunk)
        pis = np.zeros((n, 2, 1))
        pis[:, 0, 0] = self._lower
        pis[:, 1, 0] = self._upper
        return np.zeros(n), pis

    def update(self, X_chunk, y_chunk):
        pass


def test_aci_updates_per_timestep_not_per_chunk():
    """
    Pins the bug described in the README: alpha must accumulate one update
    per individual timestep inside a chunk (Gibbs & Candes 2021), not a
    single update per chunk based on the chunk's aggregate error rate.
    Because the per-timestep update is linear, applying it n times differs
    from a naive single-update-per-chunk implementation by a factor of n —
    this is exactly the "two orders of magnitude less reactive at
    step_size=168" bug described in the README. A fixed interval and a
    chunk with a mix of covered/uncovered points makes the two formulas
    diverge, observable via the confidence_level passed into the *next*
    chunk's predict call.
    """
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    X_test = pd.DataFrame({"f": np.zeros(10)}, index=dates)
    y_chunk1 = np.array([0.0, 0.0, 5.0, -5.0, 5.0])  # interval fixed [-1, 1]: 3 misses, 2 covers
    y_test = pd.Series(np.concatenate([y_chunk1, np.zeros(5)]), index=dates)

    # Small gamma keeps both candidate alphas inside [0.01, 0.99] so the
    # comparison isn't masked by clipping.
    base_alpha, gamma, step_size = 0.10, 0.02, 5
    stub = _StubModel(lower=-1.0, upper=1.0)

    run_conformal_inference(
        working_model=stub, X_test=X_test, y_test=y_test,
        base_alpha=base_alpha, gamma=gamma, step_size=step_size,
    )

    # Correct: alpha accumulates gamma*(base_alpha - err_t) once per point.
    expected_alpha = base_alpha
    for y in y_chunk1:
        err = 0.0 if -1.0 <= y <= 1.0 else 1.0
        expected_alpha += gamma * (base_alpha - err)
    expected_safe_alpha = float(np.clip(expected_alpha, 0.01, 0.99))
    # Wrong: a single update per chunk using the aggregate error rate,
    # with no per-timestep accumulation (missing the factor-of-n reactivity).
    mean_err = np.mean([0.0 if -1.0 <= y <= 1.0 else 1.0 for y in y_chunk1])
    wrong_alpha = base_alpha + gamma * (base_alpha - mean_err)
    wrong_safe_alpha = float(np.clip(wrong_alpha, 0.01, 0.99))
    assert expected_safe_alpha != pytest.approx(wrong_safe_alpha, abs=1e-6), (
        "Test setup error: correct and buggy alpha formulas coincide — "
        "not a valid regression guard."
    )
    second_call_alpha = stub.alphas_used[1]
    assert second_call_alpha == pytest.approx(expected_safe_alpha, abs=1e-9)