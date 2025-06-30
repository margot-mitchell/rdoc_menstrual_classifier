import pandas as pd
from src.temporal_models.rule_based_prior import RuleBasedPrior
from src.utils.data_loader import load_config

# Load the 10-sample-per-subject subset
unlabeled = pd.read_csv("outputs/data/hormone_data_unlabeled.csv")

# (Optional) For evaluation, merge with labeled data to get true labels
try:
    labeled = pd.read_csv("outputs/data/full_hormone_data_labeled.csv")
    data = unlabeled.merge(
        labeled[['subject_id', 'date', 'phase']],
        on=['subject_id', 'date'],
        how='left'
    )
except Exception:
    data = unlabeled

# Load config and prior
config = load_config('config/prediction_config.yaml')
prior = RuleBasedPrior(config)
prior.load_data()

# Make predictions only for these rows
prior_predictions = prior.predict_phases(data)
data['prior_prediction'] = prior_predictions

# Save predictions
output_path = "outputs/predictions/prior_predictions_unlabeled.csv"
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)
data.to_csv(output_path, index=False)
print(f"Prior predictions saved to {output_path}") 