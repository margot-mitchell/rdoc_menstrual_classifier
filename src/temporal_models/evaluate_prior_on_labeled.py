#!/usr/bin/env python3
"""
Evaluate the rule-based prior on labeled hormone data.
Reports accuracy and confusion matrix.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.temporal_models.rule_based_prior import RuleBasedPrior
from src.utils.data_loader import load_config

# Load config
config = load_config(os.path.join(project_root, 'config', 'training_config.yaml'))

# Load labeled hormone data
labeled_path = config['data']['labeled_data_path']
labeled_df = pd.read_csv(labeled_path)

# True labels
true_labels = labeled_df['phase']

# Run rule-based prior
prior = RuleBasedPrior(config)
prior.load_data()
pred_labels = prior.predict_phases(labeled_df)

# Compute accuracy
acc = accuracy_score(true_labels, pred_labels)
print(f"Rule-based prior accuracy on labeled data: {acc:.3f}")

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=prior.phases)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=prior.phases, yticklabels=prior.phases)
plt.xlabel('Predicted Phase')
plt.ylabel('True Phase')
plt.title('Rule-Based Prior Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(project_root, 'outputs', 'reports', 'prior_confusion_matrix.png'))
plt.show() 