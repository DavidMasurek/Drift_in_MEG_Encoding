from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from utils import MetadataHelper, DatasetHelper, ExtractionHelper, GLMHelper

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Choose Parameters
subject_id = "02"
create_metadata = False
create_dataset = False
extract_features = False
train_GLM = False
generate_predictions_with_GLM = True

##### Process metadata for subject #####
if create_metadata:
    # Initialize metadata helper
    metadata_helper = MetadataHelper(subject_id=subject_id)

    # Read crop metadata over all sessions 
    metadata_helper.create_crop_metadata_dict()

    # Read meg metadata over all sessions
    metadata_helper.create_meg_metadata_dict()

    # Combine meg and crop metadata over all sessions
    metadata_helper.create_combined_metadata_dict()

##### Create crop and meg dataset based on metadata #####
if create_dataset:
    # Initialize dataset helper
    dataset_helper = DatasetHelper(subject_id=subject_id)

    # Create train/test split based on sceneIDs (based on trial_ids)
    dataset_helper.create_train_test_split()

    # Create crop numpy dataset based on split
    dataset_helper.create_crop_dataset()

    # Create torch dataset based on numpy datasets
    dataset_helper.create_pytorch_dataset()

    # Create meg dataset based on split
    dataset_helper.create_meg_dataset()

##### Extract features from crops #####
if extract_features:
    # Initialize extraction helper
    extraction_helper = ExtractionHelper(subject_id=subject_id)

    # Extract features
    extraction_helper.extract_features()

##### Train GLM from features to meg #####
if train_GLM or generate_predictions_with_GLM:
    # Initialize GLM helper
    glm_helper = GLMHelper(subject_id=subject_id)

if train_GLM:
    # Train GLM
    glm_helper.train_mapping()

if generate_predictions_with_GLM:
    # Generate meg predictions from GLMs
    glm_helper.predict_from_mapping()

print("Pipeline completed.")

