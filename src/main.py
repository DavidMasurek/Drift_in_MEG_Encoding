from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import json
import logging
from setup_logger import setup_logger
from datetime import datetime
from collections import defaultdict
from utils import BasicOperationsHelper, MetadataHelper, DatasetHelper, ExtractionHelper, GLMHelper, VisualizationHelper
from sklearn.preprocessing  import RobustScaler

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Choose params
subject_ids = ["02"]  # "01", "02", "03", "05"]  # "01", "02", "03", "04", "05" 
lock_event = "saccade" # "saccade" "fixation"

crop_size = 112  # 224 112

ann_model = "Alexnet"  # "Resnet50"
module_name =  "features.12" # "fc" # features.12 has 9216 dimensions
batch_size = 32

pca_components = 30

meg_channels = [1731, 1921, 2111, 2341, 2511]
n_grad = 0
n_mag = 5

best_timepoints_by_subject = {"fixation":  {"01": {"timepoint_min": 175, "timepoint_max": 275}, 
                                            "02": {"timepoint_min": 175, "timepoint_max": 255},
                                            "03": {"timepoint_min": 175, "timepoint_max": 225},
                                            "05": {"timepoint_min": 160, "timepoint_max": 250},},
                              "saccade":   {"01": {"timepoint_min": 425, "timepoint_max": 530}, 
                                            "02": {"timepoint_min": 425, "timepoint_max": 525},
                                            "03": {"timepoint_min": 400, "timepoint_max": 485},
                                            "04": {"timepoint_min": 430, "timepoint_max": 515},
                                            "05": {"timepoint_min": 420, "timepoint_max": 510},}
                            }
timepoint_min = 0  # fixation: 170, saccade: 275
timepoint_max = 650  # fixation: 250, saccade: 375

normalizations = ["mean_centered_ch_then_global_robust_scaling"] # , "no_norm", "mean_centered_ch_t"]  #, "no_norm", "mean_centered_ch_t", "robust_scaling"]  # ,  # ["min_max", , "median_centered_ch_t", "robust_scaling", "no_norm"]

#fractional_grid = np.array([0.000_000_000_000_000_1, 0.000_000_000_000_001, 0.000_000_000_000_01, 0.000_000_000_000_1, 0.000_000_000_001, 0.000_000_000_01, 0.000_000_000_1, 0.000_000_001, 0.000_000_01, 0.000_000_1, 0.000_001, 0.000_01, 0.000_1, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
fractional_grid = np.array([fraction/100 for fraction in range(1, 100, 3)]) # range from 0.01 to 1 in steps or 0.03
alphas = [1, 10, 100, 1000 ,10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000, 10_000_000_000, 100_000_000_000, 1_000_000_000_000, 10_000_000_000_000, 100_000_000_000_000] #, 10_000_000, 100_000_000, 1_000_000_000]  # ,10,100,1000 ,10000 ,100000,1000000

omit_sessions_by_subject = {"01": ["1"],
                            "02": ["4"],
                            "03": [],
                            "04": [],
                            "05": ["9"],
                            }

assert len(meg_channels) == n_grad+n_mag, "Inconsistency in chosen channels and n_grad/n_mag."

# Map timepoint idx to ms
# fixation locked: [-300 ms : 500 ms], 0.5Hz, 601 timepoints, 
def map_timepoint_idx_to_ms(timepoint_idx:int, lock_event:str) -> int:
    """
    Maps a timepoint index to its time in ms relative to the respective lock-event
    !!! Attention, ms cuts around lock_event different for ICA cleaned and non-ICA cleaned? ICA cleaned same for both lock events 401 timepoints from -0.3 to 0.5?
    """
    if lock_event == "fixation":
        available_ms_values = list(range(-300, 502, 2))
        ms_value = available_ms_values[timepoint_idx]
        return ms_value
    elif lock_event == "saccade":
        available_ms_values = list(range(-300, 502, 2))
        #raise NotImplementedError("Mapping from timepoint_idx to ms relative to lock_event has not been implemented for saccade-locked data yet.")
    else:
        raise ValueError(f"map_timepoint_idx_to_ms called with invalid lock_event {lock_event}")
        
map_timepoint_idx_to_ms(timepoint_idx=0, lock_event="saccade")
###

logger_level = 25
debugging = True if logger_level <= 23 else False  # TODO: Use this as class attribute rather than passing it to every function

# Choose Calculations to be performed
create_metadata = False
create_train_test_split = False  # Careful! Everytime this is set to true, all following steps will be misalligned
create_crop_datset_numpy = False
create_meg_dataset = False
extract_features = False
perform_pca = False
train_GLM = False
generate_predictions_with_GLM = False
visualization = True

z_score_features_before_pca = True
use_pca_features = True

use_ica_cleaned_data = True
clip_outliers = True
interpolate_outliers = False  # Currently only implemented for mean_centered_ch_then_global_z! Cuts off everything over +-3 std
use_best_timepoints_for_subject = True

fractional_ridge = False

fit_measure_storage_distinction = "session_level"

subtract_self_pred = False
time_window_n_indices = 10
all_windows_one_plot = True
cut_repeated_session = False
omit_non_generalizing_sessions = True


# Debugging
run_pipeline_n_times = 1
downscale_features = False
all_sessions_combined = False
investigate_missing_metadata = False
shuffle_train_labels = False
shuffle_test_labels = False  # shuffles the data that is to be predicted! (In control, this can be the train split aswell)

logging_setup = setup_logger(logger_level)
logger = logging.getLogger(__name__)

for run in range(run_pipeline_n_times):
    for subject_id in subject_ids:
        if use_best_timepoints_for_subject:
            timepoint_min = best_timepoints_by_subject[lock_event][subject_id]["timepoint_min"]
            timepoint_max = best_timepoints_by_subject[lock_event][subject_id]["timepoint_max"]
        if omit_non_generalizing_sessions:
            sessions_to_omit = omit_sessions_by_subject[subject_id]
        else:
            sessions_to_omit = []

        logger.custom_info(f"Processing subject {subject_id}.\n \n \n")

        ##### Process metadata for subject #####
        if create_metadata:
            metadata_helper = MetadataHelper(crop_size=crop_size, subject_id=subject_id, lock_event=lock_event)

            # Read metadata of all available crops/images
            metadata_helper.create_crop_metadata_dict()
            # Read metadata of all available meg datapoints
            metadata_helper.create_meg_metadata_dict()
            # Create combined metadata that only contains timepoints for which crop and meg information exists
            metadata_helper.create_combined_metadata_dict(investigate_missing_metadata=investigate_missing_metadata)

            logger.custom_info("Metadata created.\n \n")

        ##### Create crop and meg dataset based on metadata #####
        if create_train_test_split or create_crop_datset_numpy or create_meg_dataset:
            dataset_helper = DatasetHelper(subject_id=subject_id, normalizations=normalizations, chosen_channels=meg_channels, lock_event=lock_event, timepoint_min=timepoint_min, timepoint_max=timepoint_max, crop_size=crop_size)

            if create_train_test_split:
                # Create train/test split based on sceneIDs (based on trial_ids)
                dataset_helper.create_train_test_split(debugging=debugging)

                logger.custom_info("Train/Test split created. \n \n")

            if create_crop_datset_numpy:
                # Create crop dataset with images as numpy arrays
                dataset_helper.create_crop_dataset(debugging=debugging)

                logger.custom_info("Numpy crop datasets created. \n \n")

            if create_meg_dataset:
                # Create meg dataset based on split
                dataset_helper.create_meg_dataset(use_ica_cleaned_data=use_ica_cleaned_data, interpolate_outliers=interpolate_outliers, clip_outliers=clip_outliers)

                logger.custom_info("MEG datasets created. \n \n")


        ##### Extract features from crops and perform pca #####
        if extract_features or perform_pca:
            extraction_helper = ExtractionHelper(subject_id=subject_id, pca_components=pca_components, ann_model=ann_model, module_name=module_name, batch_size=batch_size, lock_event=lock_event)

            if extract_features:
                extraction_helper.extract_features()
                logger.custom_info("Features extracted. \n \n")

            if perform_pca:
                extraction_helper.reduce_feature_dimensionality(z_score_features_before_pca=z_score_features_before_pca, all_sessions_combined=all_sessions_combined)
                logger.custom_info("PCA applied to features. \n \n")
            

        ##### Train GLM from features to meg #####
        if train_GLM or generate_predictions_with_GLM:
            glm_helper = GLMHelper(fractional_ridge=fractional_ridge, fractional_grid=fractional_grid, normalizations=normalizations, subject_id=subject_id, chosen_channels=meg_channels, alphas=alphas, timepoint_min=timepoint_min, timepoint_max=timepoint_max, pca_features=use_pca_features, pca_components=pca_components, lock_event=lock_event, ann_model=ann_model, module_name=module_name, batch_size=batch_size, crop_size=crop_size)

            # Train GLM
            if train_GLM:
                glm_helper.train_mapping(all_sessions_combined=all_sessions_combined, shuffle_train_labels=shuffle_train_labels, downscale_features=downscale_features)

                logger.custom_info("GLMs trained. \n \n")

            # Generate meg predictions from GLMs
            if generate_predictions_with_GLM:
                glm_helper.predict_from_mapping(fit_measure_storage_distinction=fit_measure_storage_distinction, predict_train_data=False, all_sessions_combined=all_sessions_combined, shuffle_test_labels=shuffle_test_labels, downscale_features=downscale_features)
                glm_helper.predict_from_mapping(fit_measure_storage_distinction=fit_measure_storage_distinction, predict_train_data=True, all_sessions_combined=all_sessions_combined, shuffle_test_labels=shuffle_test_labels, downscale_features=downscale_features)
                glm_helper.predict_from_mapping(fit_measure_storage_distinction="timepoint_level", predict_train_data=False, all_sessions_combined=all_sessions_combined, shuffle_test_labels=shuffle_test_labels, downscale_features=downscale_features)
                glm_helper.predict_from_mapping(fit_measure_storage_distinction="timepoint_sensor_level", predict_train_data=False, all_sessions_combined=all_sessions_combined, shuffle_test_labels=shuffle_test_labels, downscale_features=downscale_features)


                logger.custom_info("Predictions generated. \n \n")

        ##### Visualization #####
        if visualization:
            visualization_helper = VisualizationHelper(normalizations=normalizations, subject_id=subject_id, chosen_channels=meg_channels, lock_event=lock_event, alphas=alphas, timepoint_min=timepoint_min, timepoint_max=timepoint_max, pca_features=use_pca_features, pca_components=pca_components, ann_model=ann_model, module_name=module_name, batch_size=batch_size, n_grad=n_grad, n_mag=n_mag, crop_size=crop_size, fractional_ridge=fractional_ridge, fractional_grid=fractional_grid, time_window_n_indices=time_window_n_indices)

            # Visualize meg data with mne
            #visualization_helper.visualize_meg_epochs_mne()

            # Visualize meg data ERP style
            #visualization_helper.visualize_meg_ERP_style(plot_norms=["no_norm", "mean_centered_ch_t"])  # ,"robust_scaling_ch_t", "z_score_ch_t", "robust_scaling", "z_score"

            # Visualize encoding model performance
            ###visualization_helper.visualize_self_prediction(var_explained=True, pred_splits=["train","test"], all_sessions_combined=all_sessions_combined)
            ##visualization_helper.visualize_self_prediction(var_explained=True, pred_splits=["test"], all_sessions_combined=all_sessions_combined)

            # Visualize prediction results
            #visualization_helper.visualize_GLM_results(by_timepoints=False, only_distance=False, omit_sessions=[], separate_plots=True)
            ##visualization_helper.visualize_GLM_results(only_distance=True, omit_sessions=sessions_to_omit, var_explained=True)
            ###visualization_helper.visualize_GLM_results(only_distance=True, omit_sessions=[], var_explained=True)
            ##visualization_helper.visualize_GLM_results(by_timepoints=True, var_explained=True, separate_plots=True)
            #visualization_helper.visualize_GLM_results(only_distance=True, omit_sessions=["4","10"], var_explained=False)

            # Visuzalize distance based predictions at timepoint scale
            ##visualization_helper.three_dim_timepoint_predictions(subtract_self_pred=subtract_self_pred) 
            visualization_helper.timepoint_window_drift(subtract_self_pred=subtract_self_pred, omitted_sessions=sessions_to_omit, all_windows_one_plot=all_windows_one_plot)  
            
            # Visualize drift topographically with mne based on sensor level data 
            visualization_helper.visualize_topo_with_drift_per_sensor(omitted_sessions=sessions_to_omit)

            # Visualize model perspective (values by timepoint)
            ##visualization_helper.new_visualize_model_perspective(plot_norms=["mean_centered_ch_then_global_robust_scaling"], seperate_plots=False)  # , "no_norm"

            logger.custom_info("Visualization completed. \n \n")
            

logger.custom_info("Pipeline completed.")


logger.warning("Using saccade for .fif file regardless of used lock_event for session date differences because files does not exist for fixations.")
if use_ica_cleaned_data:
    logger.warning("idx to ms timepoints mapping in plots is currently based on ica cleaned metadata. Validation is required before generalizing to other data files.")

