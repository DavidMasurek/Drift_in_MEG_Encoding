import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import imageio
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Params
# Dataset
subject_id = "02"
session_id_num = "1"

# Model
model_name = "Resnet50_ecoset"
module_name = "fc"

# Get ResNet50 features
features = {"train": None, "test": None}
for split in features:
    feature_path = f"data_files/features/{split}_features_{model_name}_{module_name}_subj-{subject_id}_sess-{session_id_num}.npy"
    features[split] = np.load(feature_path)

# Get MEG data
meg_data = {"train": None, "test": None}
for split in features:
    meg_data_path = f"data_files/meg_data/meg_{split}_ds_subj_{subject_id}_sess_{session_id_num}.npy"
    meg_data[split] = np.load(meg_data_path)

X_train, Y_train = features['train'], meg_data['train']
X_test, Y_test = features['test'], meg_data['test']

print(f"X_train.shape: {X_train.shape}")
print(f"Y_train.shape: {Y_train.shape}")

print(f"X_test.shape: {X_test.shape}")
print(f"Y_test.shape: {Y_test.shape}")

# Train GLM on train set
# Create Wrapper for multidimensional ridge regression over timepoints
class MultiDimensionalRidge:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.models = []

    def fit(self, X, Y):
        n_timepoints = Y.shape[2]
        self.models = [Ridge(alpha=self.alpha) for _ in range(n_timepoints)]
        for t in range(n_timepoints):
            Y_t = Y[:, :, t]
            self.models[t].fit(X, Y_t)

    def predict(self, X):
        n_samples = X.shape[0]
        n_sensors = self.models[0].coef_.shape[0]
        n_timepoints = len(self.models)
        predictions = np.zeros((n_samples, n_sensors, n_timepoints))
        for t, model in enumerate(self.models):
            predictions[:, :, t] = model.predict(X)
        return predictions

model = MultiDimensionalRidge(alpha=0.5)

# Fit model
model.fit(X_train, Y_train)

# Generate predictions for test features and evaluate them 
predictions = model.predict(X_test)

print(f"Y_test.reshape(-1).shape: {Y_test.reshape(-1).shape}")
print(f"predictions.reshape(-1).shape: {predictions.reshape(-1).shape}")

# Calculate the mean squared error across all flattened features and timepoints
mse = mean_squared_error(Y_test.reshape(-1), predictions.reshape(-1))

print(f"mse: {mse}")