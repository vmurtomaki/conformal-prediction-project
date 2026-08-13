# Navigate to the project root (one level up from the scripts directory)
Set-Location "$PSScriptRoot\.."

# 1. Install dependencies (uses uv.lock, matches the Dockerfile)
uv sync --all-extras --dev

# 2. Run the full test suite with coverage (as configured in pyproject.toml)
uv run pytest tests/ -v

# 3. Lint and type-check
uv run ruff check src/ app/ tests/
uv run mypy src/ app/

# 4. Confirm config.py actually loads hyperparameters.yaml
uv run python -c "from src.config import load_config; print(load_config())"
# expect: {'alpha': 0.1, 'gamma': 0.01, 'bootstrap_estimators': 30}

# 5. Confirm the per-timestep ACI update behaves sanely on a quick synthetic run
# (Using PowerShell Here-String piped into Python)
@"
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.conformal_engine import calibrate_mapie_model, run_conformal_inference

dates = pd.date_range("2023-01-01", periods=300, freq="h")
X = pd.DataFrame({"f": np.random.randn(300)}, index=dates)
y = pd.Series(np.random.randn(300), index=dates)

model = calibrate_mapie_model(RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0), X, y, n_blocks=5)

X_test = pd.DataFrame({"f": np.random.randn(60)}, index=dates[:60])
y_test = pd.Series(np.random.randn(60) * 20, index=dates[:60])

res = run_conformal_inference(model, X_test, y_test, base_alpha=0.10, gamma=0.3, step_size=20)
assert (res["lower_bound"] <= res["upper_bound"]).all()
print("OK — bounds valid, coverage:", ((res.true_value >= res.lower_bound) & (res.true_value <= res.upper_bound)).mean())
"@ | uv run python -

# 6. Build and sanity-check the Docker image (validates the Dockerfile fix)
docker build -t conformal-prediction-project .
docker run --rm conformal-prediction-project uv run python -c "import streamlit, mapie, sklearn; print('deps OK')"

# 7. (Optional, needs data/02_processed/base_model.pkl and data/01_raw/LD2011_2014.txt present)
# Launch the app itself and check it serves on :8501
Write-Host "Starting Streamlit in the background..."
$appProcess = Start-Process -FilePath "uv" -ArgumentList "run", "streamlit", "run", "app/main.py", "--server.headless=true" -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 5

# Use curl.exe to avoid PS alias conflicts and verify health
curl.exe -sf http://localhost:8501/_stcore/health
if ($LASTEXITCODE -eq 0) {
    Write-Host "Streamlit up" -ForegroundColor Green
} else {
    Write-Host "Streamlit healthcheck failed" -ForegroundColor Red
}

# Kill the background process
Write-Host "Stopping Streamlit..."
Stop-Process -Id $appProcess.Id -Force
