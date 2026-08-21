import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from src.config import load_config
from src.conformal_engine import calibrate_mapie_model, run_conformal_inference
from src.data_processing import create_features, load_and_clean_data, resample_and_impute

CONFIG = load_config()

data_path = Path("data/01_raw/LD2011_2014.txt")
raw_df = load_and_clean_data(str(data_path))
processed_df = resample_and_impute(raw_df)
df_ml = create_features(processed_df, "MT_320")

split_idx = int(len(df_ml) * 0.8)
train, test = df_ml.iloc[:split_idx], df_ml.iloc[split_idx:].tail(1500)
X_train, y_train = train.drop(columns=["target"]), train["target"]
X_test, y_test = test.drop(columns=["target"]), test["target"]

base_model = joblib.load("data/02_processed/base_model.pkl")
mapie = calibrate_mapie_model(base_model, X_train, y_train, n_blocks=CONFIG["bootstrap_estimators"])

def summarize(gamma):
    import copy
    res = run_conformal_inference(
        copy.deepcopy(mapie), X_test, y_test,
        base_alpha=CONFIG["alpha"], gamma=gamma, step_size=168
    )
    covered = (res.true_value >= res.lower_bound) & (res.true_value <= res.upper_bound)
    width = (res.upper_bound - res.lower_bound).mean()
    return covered.mean(), width

static_cov, static_width = summarize(0.0)
print(f"Static  (gamma=0):    coverage={static_cov:.2%}, width={static_width:.2f}")

cov, width = summarize(CONFIG["gamma"])
reduction = (static_width - width) / static_width * 100
print(f"Adaptive(gamma={CONFIG['gamma']}): coverage={cov:.2%}, width={width:.2f}, reduction={reduction:.1f}%")
