import warnings

import numpy as np
import pandas as pd
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from sklearn.base import RegressorMixin


def calibrate_mapie_model(
    base_model: RegressorMixin, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    n_blocks: int = 30
) -> TimeSeriesRegressor:
    cv_mapi = BlockBootstrap(
        n_resamplings=n_blocks, 
        n_blocks=n_blocks, 
        random_state=42
    )
    
    mapie_model = TimeSeriesRegressor(
        estimator=base_model,
        method="enbpi",
        cv=cv_mapi,
        agg_function="mean",
        n_jobs=-1
    )
    
    mapie_model.fit(X_train, y_train)
    return mapie_model

def run_conformal_inference(
    working_model: TimeSeriesRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    base_alpha: float,
    gamma: float,
    step_size: int = 24
) -> pd.DataFrame:
    y_preds = []
    y_pis_lower = []
    y_pis_upper = []
    
    current_alpha = base_alpha
    
    warnings.filterwarnings("ignore", category=UserWarning)

    for i in range(0, len(X_test), step_size):
        X_chunk = X_test.iloc[i:i+step_size]
        y_chunk = y_test.iloc[i:i+step_size]
        
        safe_alpha = float(np.clip(current_alpha, 0.01, 0.99))
        safe_confidence = float(np.clip(1.0 - safe_alpha, 0.01, 0.99))
        
        pred, pis = working_model.predict(
            X_chunk, 
            ensemble=True,
            confidence_level=safe_confidence, 
            optimize_beta=False
        )
        
        y_preds.extend(pred)
        
        chunk_lower = pis[:, 0, 0]
        chunk_upper = pis[:, 1, 0]
        
        y_pis_lower.extend(chunk_lower)
        y_pis_upper.extend(chunk_upper)
        
        working_model.update(X_chunk, y_chunk)
        
        if gamma > 0.0:
            is_covered = (y_chunk.values >= chunk_lower) & (y_chunk.values <= chunk_upper)
            local_error_rate = 1.0 - is_covered.mean()
            current_alpha = current_alpha + gamma * (base_alpha - local_error_rate)

    results_df = pd.DataFrame({
        'timestamp': X_test.index,
        'true_value': y_test.values,
        'prediction': np.array(y_preds),
        'lower_bound': np.array(y_pis_lower),
        'upper_bound': np.array(y_pis_upper)
    }).set_index('timestamp')
    
    return results_df