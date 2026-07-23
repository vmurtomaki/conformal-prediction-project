import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import joblib

def split_time_series(df: pd.DataFrame, train_ratio: float = 0.8):
    """Splits time series data strictly chronologically."""
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    X_train, y_train = train.drop(columns=['target']), train['target']
    X_test, y_test = test.drop(columns=['target']), test['target']
    
    return X_train, X_test, y_train, y_test

def train_base_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Trains a RandomForest utilizing TimeSeriesSplit cross-validation."""
    print("Initializing TimeSeriesSplit cross-validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Random Forests are highly stable and capture non-linear covariate relationships
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Parameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    
    print("Tuning hyperparameters (this may take a minute)...")
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=5, # Kept low for execution speed; increase for production
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    print(f"Best parameters found: {search.best_params_}")
    
    return search.best_estimator_

if __name__ == "__main__":
    DATA_PATH = "data/02_processed/ml_features_MT_320.parquet"
    MODEL_PATH = "data/02_processed/base_model.pkl"
    
    print("Loading feature matrix...")
    df = pd.read_parquet(DATA_PATH)
    
    X_train, X_test, y_train, y_test = split_time_series(df)
    print(f"Training set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")
    
    best_model = train_base_model(X_train, y_train)
    
    # Evaluate statically to establish a baseline MAE before applying MAPIE
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Baseline Test MAE (Point Forecast): {mae:.2f}")
    
    joblib.dump(best_model, MODEL_PATH)
    print(f"Serialized base model saved to {MODEL_PATH}")