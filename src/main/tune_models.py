import os
import pandas as pd
from src.utils.data_loader import load_config, load_hormone_data, DataPreprocessor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import json

# === 1. Load config and data ===
config_path = "config/training_config.yaml"
config = load_config(config_path)
data = load_hormone_data(config['data']['labeled_data_path'])

# === 2. Preprocess data ===
preprocessor = DataPreprocessor(config)
X = preprocessor.fit_transform(data, add_prior=True)  # Use prior features if desired
y = data['phase']  # Or your target column

# === 3. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 4. Random Forest Grid Search ===
rf_param_grid = {
    'n_estimators': [100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 5],
    'max_features': ['sqrt', 0.8]
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
rf_grid.fit(X_train, y_train)
print("Random Forest best params:", rf_grid.best_params_)
print("Random Forest best CV score:", rf_grid.best_score_)
print("Random Forest test set score:", rf_grid.best_estimator_.score(X_test, y_test))

# Save best params for Random Forest
with open("best_rf_params.json", "w") as f:
    json.dump(rf_grid.best_params_, f)
print("Saved best Random Forest params to best_rf_params.json")

# === 5. LightGBM Grid Search ===
lgb_param_grid = {
    'n_estimators': [100],
    'max_depth': [5, 10, -1],
    'num_leaves': [15, 31],
    'min_child_samples': [20, 50],
    'reg_alpha': [0, 1],
    'reg_lambda': [0, 1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
lgbm = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
lgb_grid = GridSearchCV(lgbm, lgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
lgb_grid.fit(X_train, y_train)
print("LightGBM best params:", lgb_grid.best_params_)
print("LightGBM best CV score:", lgb_grid.best_score_)
print("LightGBM test set score:", lgb_grid.best_estimator_.score(X_test, y_test))

# Save best params for LightGBM
with open("best_lgb_params.json", "w") as f:
    json.dump(lgb_grid.best_params_, f)
print("Saved best LightGBM params to best_lgb_params.json") 