from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from utils import BasicOperationsHelper

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Choose params
subject_ids = ["02"]
normalizations = ["no_norm"] # ["min_max", "mean_centered_ch_t", "median_centered_ch_t", "robust_scaling", "no_norm"]
lock_event = "saccade"
meg_channels = [1731, 1921, 2111, 2341, 2511]
timepoint_min = 50
timepoint_max = 250
alphas = [1,10,100,1000,10000,100000,1000000] 
all_sessions_combined = True


ops_helper = BasicOperationsHelper()

ann_features_train_combined = None
meg_data_train_combined = None
# Collect ANN features and MEG data over sessions
for session_id_num in ops_helper.session_ids_num:
    ann_features_train = ops_helper.load_split_data_from_file(session_id_num=session_id_num, type_of_content="ann_features_pca", ann_model="Resnet50", module="fc")['train']
    meg_data_train = ops_helper.load_split_data_from_file(session_id_num=session_id_num, type_of_content="meg_data", type_of_norm="no_norm")['train']

    print(f"Train_mapping: ann_features_train.shape: {ann_features_train.shape}")
    print(f"Train_mapping: meg_data_train.shape: {meg_data_train.shape}")

    if session_id_num == "1":
        ann_features_train_combined = ann_features_train
        meg_data_train_combined = meg_data_train
    else:
        ann_features_train_combined = np.concatenate([ann_features_train_combined, ann_features_train], axis=0)
        meg_data_train_combined = np.concatenate([meg_data_train_combined, meg_data_train], axis=0)

print(f"Train_mapping: ann_features_train_combined.shape: {ann_features_train_combined.shape}")
print(f"Train_mapping: meg_data_train_combined.shape: {meg_data_train_combined.shape}")

#split_path = "data_files/ann_features_pca/Resnet50/fc/subject_02/session_1/train/ann_features_pca.npy"
split_path = "data_files/ann_features_pca/all_sessions_combined/Resnet50/fc/subject_02/train/ann_features_pca.npy"
train_features_pca = np.load(split_path)

print(f"train_features_pca.shape: {train_features_pca.shape}")