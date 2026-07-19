import pandas as pd
import numpy as np

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw UCI electricity dataset, fixes formatting, and sets a datetime index.
    """
    print("Loading raw data... This might take a moment.")
    # The dataset uses European formatting: ';' separator and ',' for decimals
    df = pd.read_csv(
        filepath, 
        sep=';', 
        decimal=',', 
        parse_dates=[0], 
        index_col=0,
        dtype=np.float32 # Optimize memory
    )
    
    # Rename the index for clarity
    df.index.name = 'timestamp'
    return df

def resample_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples 15-minute data to hourly and imputes DST missing values.
    """
    print("Resampling to hourly frequency...")
    # Resample to hourly by summing the 15-minute intervals
    df_hourly = df.resample('h').sum()
    
    print("Imputing missing values caused by Daylight Saving Time...")
    # Use time-based interpolation to fill the missing March hour
    df_hourly = df_hourly.interpolate(method='time')
    
    return df_hourly

def create_features(df: pd.DataFrame, target_client: str) -> pd.DataFrame:
    """
    Transforms a univariate time series into a feature matrix for ML models.
    """
    print(f"Engineering features for client: {target_client}...")
    
    # Extract the specific client's data
    data = df[[target_client]].copy()
    data.rename(columns={target_client: 'target'}, inplace=True)
    
    # Force Pylance to recognize the DatetimeIndex
    # data.index = pd.to_datetime(data.index)
    
    # 1. Calendar Features
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['day_of_month'] = data.index.day
    data['month'] = data.index.month
    
    # 2. Historical Lags (Autocorrelation)
    data['lag_1'] = data['target'].shift(1)       # Previous hour
    data['lag_24'] = data['target'].shift(24)     # Same time yesterday
    data['lag_168'] = data['target'].shift(168)   # Same time last week
    
    # 3. Rolling Statistics (Volatility/Trend)
    # We shift by 1 to prevent data leakage (the model can't know the current hour's target)
    data['rolling_mean_24'] = data['target'].shift(1).rolling(window=24).mean()
    data['rolling_std_24'] = data['target'].shift(1).rolling(window=24).std()
    
    # Drop NaN values created by shifting/rolling (this drops the first 168 rows)
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