import pandas as pd
import numpy as np
import joblib
import yaml
import os

# NEW IMPORTS FOR MAPIE >= 1.0
from mapie.regression import TimeSeriesRegressor 
from mapie.subsample import BlockBootstrap
# FIX 1: Import metrics from the regression submodule
from mapie.metrics.regression import regression_coverage_score, regression_mean_width_score

def load_config():
    with open("config/hyperparameters.yaml", "r") as file:
        return yaml.safe_load(file)

def run_conformal_engine():
    config = load_config()
    alpha = config.get("alpha", 0.10)
    n_blocks = config.get("bootstrap_estimators", 30)

    print("Loading data and serialized base model...")
    df = pd.read_parquet("data/02_processed/ml_features_MT_320.parquet")
    base_model = joblib.load("data/02_processed/base_model.pkl")

    # Re-split data to match training
    train_ratio = 0.8
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    X_train, y_train = train.drop(columns=['target']), train['target']
    X_test, y_test = test.drop(columns=['target']), test['target']

    print(f"Initializing MAPIE EnbPI with {n_blocks} block bootstraps...")
    cv_mapi = BlockBootstrap(
        n_resamplings=n_blocks, 
        n_blocks=n_blocks, 
        random_state=42
    )
    
    # FIX 1: Remove confidence_level from here
    mapie_model = TimeSeriesRegressor(
        estimator=base_model,
        method="enbpi",
        cv=cv_mapi,
        agg_function="mean",
        n_jobs=-1
    )

    print("Fitting MAPIE model (Calibrating residuals)...")
    mapie_model.fit(X_train, y_train)

    pprint("Executing sequential prediction loop on test set...")
    step_size = 24 # Update residuals daily
    
    y_preds = []
    y_pis = []
    
    # FIX: Suppress MAPIE's noisy API deprecation warning
    warnings.filterwarnings(
        action="ignore", 
        category=UserWarning, 
        message=".*This function behavior has been changed.*"
    )
    
    for i in range(0, len(X_test), step_size):
        X_chunk = X_test.iloc[i:i+step_size]
        y_chunk = y_test.iloc[i:i+step_size]
        
        # Predict intervals
        pred, pis = mapie_model.predict(
            X_chunk, 
            ensemble=True,
            confidence_level=1 - alpha, 
            optimize_beta=True
        )
        
        y_preds.append(pred)
        y_pis.append(pis) 
        
        # Update sliding window of residuals (Warning is now muted here)
        mapie_model.update(X_chunk, y_chunk)

    # Consolidate results
    y_pred_final = np.concatenate(y_preds)
    y_pis_final = np.concatenate(y_pis, axis=0) # Shape: (n_samples, 2, 1)
    
    # Metrics correctly take the 3D array
    coverage_array = regression_coverage_score(y_test, y_pis_final)
    width_array = regression_mean_width_score(y_pis_final)
    
    coverage = float(coverage_array[0]) if isinstance(coverage_array, np.ndarray) else float(coverage_array)
    width = float(width_array[0]) if isinstance(width_array, np.ndarray) else float(width_array)
    
    print(f"\n--- Conformal Engine Results ---")
    print(f"Target Coverage:   {1 - alpha:.2%}")
    print(f"Empirical Coverage:{coverage:.2%}")
    print(f"Mean Interval Width: {width:.2f}")

    results_df = pd.DataFrame({
        'timestamp': test.index,
        'true_value': y_test.values,
        'prediction': y_pred_final,
        # Flatten the 3rd dimension for the dataframe columns
        'lower_bound': y_pis_final[:, 0, 0],
        'upper_bound': y_pis_final[:, 1, 0]
    }).set_index('timestamp')
    
    output_path = "data/02_processed/conformal_results.parquet"
    results_df.to_parquet(output_path)
    print(f"\nResults saved to {output_path} for UI rendering.")

if __name__ == "__main__":
    run_conformal_engine()