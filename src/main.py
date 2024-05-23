from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from utils import BasicOperationsHelper, MetadataHelper, DatasetHelper, ExtractionHelper, GLMHelper, VisualizationHelper

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

# Choose Calculations to be performed
create_metadata = False
create_non_meg_dataset = False
create_meg_dataset = True
extract_features = False
train_GLM = True
generate_predictions_with_GLM = True
visualization = True

for subject_id in subject_ids:
    print(f"Processing subject {subject_id}")

    ##### Process metadata for subject #####
    if create_metadata:
        metadata_helper = MetadataHelper(subject_id=subject_id, lock_event=lock_event)

        # Read crop metadata over all sessions 
        metadata_helper.create_crop_metadata_dict()
        # Read meg metadata over all sessions
        metadata_helper.create_meg_metadata_dict()
        # Combine meg and crop metadata over all sessions
        metadata_helper.create_combined_metadata_dict(investigate_missing_data=False)

        print("Metadata created.")

    ##### Create crop and meg dataset based on metadata #####
    if create_non_meg_dataset or create_meg_dataset:
        dataset_helper = DatasetHelper(subject_id=subject_id, normalizations=normalizations, chosen_channels=meg_channels, lock_event=lock_event)

        if create_non_meg_dataset:
            # Create train/test split based on sceneIDs (based on trial_ids)
            dataset_helper.create_train_test_split()
            # Create crop numpy dataset based on split
            dataset_helper.create_crop_dataset()
            # Create torch dataset based on numpy datasets
            dataset_helper.create_pytorch_dataset()

            print("Non-MEG datasets created.")

        if create_meg_dataset:
            # Create meg dataset based on split
            dataset_helper.create_meg_dataset()

            print("MEG datasets created.")


    ##### Extract features from crops #####
    if extract_features:
        extraction_helper = ExtractionHelper(subject_id=subject_id)

        # Extract features
        extraction_helper.extract_features()

        print("Features extracted.")

    ##### Train GLM from features to meg #####
    if train_GLM or generate_predictions_with_GLM:
        glm_helper = GLMHelper(norms=normalizations, subject_id=subject_id, chosen_channels=meg_channels)

        # Train GLM
        if train_GLM:
            glm_helper.train_mapping()

            print("GLMs trained.")

        # Generate meg predictions from GLMs
        if generate_predictions_with_GLM:
            glm_helper.predict_from_mapping(store_timepoint_based_losses=False)

            print("Predictions generated.")

    ##### Visualization #####
    if visualization:
        visualization_helper = VisualizationHelper(norms=normalizations, subject_id=subject_id)

        # Visualize meg data with mne
        #visualization_helper.visualize_meg_epochs_mne()

        # Visualize meg data ERP style
        #visualization_helper.visualize_meg_ERP_style(plot_norms=["no_norm", "mean_centered_ch_t"])  # ,"robust_scaling_ch_t", "z_score_ch_t", "robust_scaling", "z_score"

        # Visualize prediction results
        visualization_helper.visualize_GLM_results(by_timepoints=False, only_distance=False, omit_session_10=False, separate_plots=True)
        #visualization_helper.visualize_GLM_results(by_timepoints=False, only_distance=True, omit_session_10=False, separate_plots=False)
        #visualization_helper.visualize_GLM_results(by_timepoints=False, only_distance=True, omit_session_10=True, separate_plots=False)
        
        # Visualize model perspective (values by timepoint)
        #visualization_helper.visualize_model_perspective(plot_norms=["mean_centered_ch_t", "no_norm"], seperate_plots=False)  # , "no_norm"

        print("Visualization completed.")
        

print("Pipeline completed.")

