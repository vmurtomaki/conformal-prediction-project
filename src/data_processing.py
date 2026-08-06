import numpy as np
import pandas as pd


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath, 
        sep=';', 
        decimal=',', 
        parse_dates=[0], 
        index_col=0,
        dtype=np.float32
    )
    df.index.name = 'timestamp'
    return df

def resample_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    df_hourly = df.resample('h').sum()
    # FIX: Replaced time-based interpolation with forward fill.
    # Time interpolation across the global index leaks future values into 
    # historical training and calibration matrices, violating disjoint set rules.
    df_hourly = df_hourly.ffill()
    return df_hourly

def apply_synthetic_shock(
    raw_series: pd.Series, 
    target_client: str, 
    shock_multiplier: float, 
    start_date: str, 
    end_date: str
) -> pd.DataFrame:
    df_shocked = raw_series.to_frame(name=target_client).copy()
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    mask = (df_shocked.index >= start_dt) & (df_shocked.index <= end_dt)
    df_shocked.loc[mask, target_client] *= shock_multiplier
    
    return df_shocked

def create_features(df: pd.DataFrame, target_client: str) -> pd.DataFrame:
    data = df[[target_client]].copy()
    data.rename(columns={target_client: 'target'}, inplace=True)
    
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['day_of_month'] = data.index.day
    data['month'] = data.index.month
    
    data['lag_1'] = data['target'].shift(1)
    data['lag_24'] = data['target'].shift(24)
    data['lag_168'] = data['target'].shift(168)
    
    data['rolling_mean_24'] = data['target'].shift(1).rolling(window=24).mean()
    data['rolling_std_24'] = data['target'].shift(1).rolling(window=24).std()
    
    data = data.dropna()
    
    return data

if __name__ == "__main__":
    import os
    
    # We use relative paths assuming execution from the project root
    RAW_PATH = "data/01_raw/LD2011_2014.txt"
    PROCESSED_PATH = "data/02_processed/hourly_electricity.parquet"
    MODELING_PATH = "data/02_processed/ml_features_MT_320.parquet"
    
    # 1. Load and Clean (Skip if parquet already exists to save time)
    if not os.path.exists(PROCESSED_PATH):
        raw_df = load_and_clean_data(RAW_PATH)
        processed_df = resample_and_impute(raw_df)
        processed_df.to_parquet(PROCESSED_PATH)
    else:
        print("Loading previously processed hourly data...")
        processed_df = pd.read_parquet(PROCESSED_PATH)
    
    # 2. Feature Engineering
    # MT_320 is a highly volatile client, perfect for testing Conformal Prediction
    ml_df = create_features(processed_df, target_client="MT_320")
    
    # 3. Save final feature matrix
    ml_df.to_parquet(MODELING_PATH)
    print(f"Feature engineering complete. Matrix shape: {ml_df.shape}")
    print(f"Saved to {MODELING_PATH}")