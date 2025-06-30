import joblib
import pandas as pd
from src.utils.data_loader import load_config, load_hormone_data, DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import json

# === 1. Load config and data ===
config_path = "config/training_config.yaml"
config = load_config(config_path)
data = load_hormone_data(config['data']['labeled_data_path'])

# === 2. Preprocess all data ===
preprocessor = DataPreprocessor(config)
X_all = preprocessor.fit_transform(data, add_prior=True)
y_all = data['phase']

# === 3. Load best hyperparameters from tuning results ===
with open("best_rf_params.json", "r") as f:
    rf_best_params = json.load(f)
with open("best_lgb_params.json", "r") as f:
    lgb_best_params = json.load(f)

# === 4. Retrain Random Forest on all data ===
rf_best = RandomForestClassifier(**rf_best_params, random_state=42, n_jobs=-1)
rf_best.fit(X_all, y_all)
joblib.dump(rf_best, 'best_random_forest_full.joblib')
print("Retrained Random Forest on all data and saved as best_random_forest_full.joblib")

# === 5. Retrain LightGBM on all data ===
lgb_best = lgb.LGBMClassifier(**lgb_best_params, random_state=42, n_jobs=-1)
lgb_best.fit(X_all, y_all)
joblib.dump(lgb_best, 'best_lightgbm_full.joblib')
print("Retrained LightGBM on all data and saved as best_lightgbm_full.joblib") 