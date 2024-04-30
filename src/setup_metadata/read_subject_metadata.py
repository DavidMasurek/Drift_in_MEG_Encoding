from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from utils import MetadataHelper, DatasetHelper, ExtractionHelper

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Choose Parameters
subject_id = "02"
create_metadata = False
create_dataset = False
extract_features = True

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


print("Pipeline completed.")

