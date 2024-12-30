import os
import sys
import json
import cv2
import h5py
import pickle
import pandas as pd
import numpy as np
import itertools
import imageio
from PIL import Image
import mne
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import logging
import random
from matplotlib.lines import Line2D  
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import defaultdict, Counter
from typing import Tuple, Dict
import time
from datetime import date

# ML specific imports
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from thingsvision import get_extractor

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, ElasticNetCV
import fracridge
from fracridge import FracRidgeRegressorCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import linregress, pearsonr, spearmanr, permutation_test

# Logging related
logger = logging.getLogger(__name__)

mne.set_log_level(verbose="ERROR")

class BasicOperationsHelper:
    def __init__(self, subject_id:str, lock_event:str):
        self.subject_id = subject_id
        self.lock_event = lock_event
        self.session_ids_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        self.session_ids_num = [str(session_id) for session_id in range(1,11)]

        # Create linspace of available ms values in selected meg data type for mapping from indices. Based on ICA cleaned avs data time format!
        if lock_event == "saccade":
            # AVS ICA-claned: Saccade: 651 timepoints, -800 to 500.  Old, Andrej: Saccade: 401 timepoints, -500 to 300
            min_ms = -800
            max_ms = 500
        else:
            # AVS ICA-claned: Fixation: 651 timepoints, -500 to 800.  Old, Andrej: Fixation: 401 timepoints, -300 to 500
            min_ms = -500
            max_ms = 800

        self.all_ms_values = np.linspace(min_ms, max_ms, num=651)  # space containing the ms values for all 651 timepoints

        #logger.custom_debug(f"all_ms_values linspace: {self.all_ms_values}")

       
    def perform_permutation_test(self, x_array, y_array, n_permutations=10000, direction:str="left", generate_plot:bool=False):
        assert direction in ["left", "right", "both"], f"Unrecognized direction argument: {direction}"

        def correlation_statistic(x,y):
            return pearsonr(x,y)[0]

        empirical_corr = correlation_statistic(x_array, y_array)

        # Generate null distribution by shuffling distances n times
        x_array_permutate = np.copy(x_array)
        null_distribution_corrs = []
        for i in range(n_permutations):
            np.random.shuffle(x_array_permutate)
            permute_corr = correlation_statistic(x_array_permutate, y_array)
            null_distribution_corrs.append(permute_corr)
            #if i % 500 == 0:
            #    print(f"permute_corr {i}: {permute_corr}")
        null_distribution_corrs = np.array(null_distribution_corrs)
        p_0_05_corr = np.percentile(null_distribution_corrs, 5)
        p_0_95_corr = np.percentile(null_distribution_corrs, 95)

        #print(f"len(x_array): {len(x_array)}")
        #print(f"p_0_05_corr: {p_0_05_corr}")
        #print(f"p_0_95_corr: {p_0_95_corr}")
        #print(f"empirical_corr: {empirical_corr}")

        # Compute (directed) p-value: proportion of permutations yielding a correlation not greater (left-sided) or not smaller (right-sided) than the empirical correlation
        if direction == "left":
            num_corrs_stronger_equal_empirical = np.sum(null_distribution_corrs <= empirical_corr)
        elif direction == "right":
            num_corrs_stronger_equal_empirical = np.sum(null_distribution_corrs >= empirical_corr)
        else: 
            # beidseitig
            num_corrs_stronger_equal_empirical = np.sum(np.abs(null_distribution_corrs) >= abs(empirical_corr))
        p_value = num_corrs_stronger_equal_empirical / n_permutations

        #print(f"num_corrs_stronger_equal_empirical: {num_corrs_stronger_equal_empirical}")

        if generate_plot:
            # Create figure illustration permutation test result: Empirical correlation in comparison to null distribution
            permutation_fig, permutation_ax = plt.subplots()

            # Null distribution of correlations
            permutation_ax.hist(null_distribution_corrs, bins=30, color='lightgray', density=True)

            # Vertical line for the empirical correlation, zero correlation and dashed line for p 0.05 correlation
            permutation_ax.axvline(empirical_corr, color='black', linewidth=2)
            permutation_ax.axvline(0, color='black', linewidth=1)
            
            permutation_ax.axvline(p_0_05_corr, color='red', linestyle='--', linewidth=2, label=f'P=0.05 (r={p_0_05_corr:.2f})')

            permutation_ax.set_xlabel('correlation (r)')
            permutation_ax.set_ylabel('permutations prob.')

            # DEPRECATED: Perform scipy the permutation test for comparison; none of the permutation type options seem appropriate
            #p_scipy = permutation_test((x_array, y_array), correlation_statistic, permutation_type='pairings', alternative='less', n_resamples=10000)
            #print(f"p_scipy: {p_scipy}")

            return p_value, permutation_fig, permutation_ax
        else:
            return p_value


    def omit_selected_sessions_from_fit_measures(self, fit_measures_by_session:dict, omitted_sessions:list, sensors_seperated:bool) -> dict:
        """
        Filters out values for omitted sessions from standard fit measure cross-prediction dict.
        """
        def omit_sessions_from_fit_measures_by_session(fit_measures_by_session:dict, omitted_sessions:list):
            fit_measures_sessions_omitted = self.recursive_defaultdict()
            for session_train_id, fit_measures_train_session in fit_measures_by_session['session_mapping'].items():
                if session_train_id in omitted_sessions:
                    continue
                for session_pred_id, fit_measures_pred_session in fit_measures_train_session["session_pred"].items():
                    if session_pred_id in omitted_sessions:
                        continue
                    fit_measures_sessions_omitted['session_mapping'][session_train_id]["session_pred"][session_pred_id] = fit_measures_pred_session
        
            return fit_measures_sessions_omitted

        if not sensors_seperated:
            fit_measures_sessions_omitted_complete = omit_sessions_from_fit_measures_by_session(fit_measures_by_session=fit_measures_by_session, omitted_sessions=omitted_sessions)
        else:
            fit_measures_sessions_omitted_complete = self.recursive_defaultdict()
            for sensor_idx in fit_measures_by_session['sensor']:
                fit_measures_sessions_omitted_complete['sensor'][sensor_idx] = omit_sessions_from_fit_measures_by_session(fit_measures_by_session=fit_measures_by_session['sensor'][sensor_idx], omitted_sessions=omitted_sessions)

        return fit_measures_sessions_omitted_complete
        

    def map_timepoint_idx_to_ms(self, timepoint_idx):
        """
        Maps the index of a timepoint to it's latency in ms relative to lock event (saccade/fixation) onset.

        Based on ICA-cleaned metadata from f"/share/klab/datasets/avs/population_codes/as{self.subject_id}/sensor/erf/filter_0.2_200/ica" 
        """
        ms_index = timepoint_idx + self.timepoint_min  # The index for the linspace results from the chosen idx and the beginning of the selected timepoint window (i.e. self.timepoint_min)
        timepoint_ms = int(self.all_ms_values[ms_index])

        return timepoint_ms


    def get_relevant_meg_channels(self, chosen_channels: list):
        """
        Returns names of the chosen channels and their index in the meg dataset.

        Example in: [1731, 1921, 2111, 2341, 2511]
        Example out: {'grad': {}, 'mag': {'sensor_index_within_type': {64: 'MEG1731', 71: 'MEG1921', 78: 'MEG2111', 89: 'MEG2341', 94: 'MEG2511'},
                                            'sensor_index_total': {194: 'MEG1731', 215: 'MEG1921', 236: 'MEG2111', 269: 'MEG2341', 284: 'MEG2511'}}
        """
        # pick first session, and saccade-locked (only saccade-locked exists for all) the sensors should always be the same
        fif_file_path = f'/share/klab/datasets/avs/population_codes/as{self.subject_id}/sensor/filter_0.2_200/saccade_evoked_{self.subject_id}_01_.fif'
        
        processing_channels_indices = self.recursive_defaultdict()
        evoked = mne.read_evokeds(fif_file_path)[0]

        logger.custom_debug(f"evoked.info: {evoked.info}")

        for sensor_type in ["grad", "mag"]: # grad, mag
            channel_indices = mne.pick_types(evoked.info, meg=sensor_type)
            #logger.custom_debug(f"channel_indices: {channel_indices}")
            for sensor_index_within_type, sensor_index_total in enumerate(channel_indices):
                ch_name = evoked.ch_names[sensor_index_total]
                if str(ch_name[3:]) in chosen_channels:
                    #logger.custom_debug(f"sensor_index_within_type: {sensor_index_within_type}, sensor_index_total: {sensor_index_total}")
                    processing_channels_indices[sensor_type]["sensor_index_within_type"][sensor_index_within_type] = ch_name
                    processing_channels_indices[sensor_type]["sensor_index_total"][sensor_index_total] = ch_name
            #logger.custom_debug(f"{sensor_type}, last index: {sensor_index}")

        return processing_channels_indices

    
    def get_session_date_differences(self, subject_id=None):
        """
        Calculates the rounded differences in days between all sessions.
        Taken from: fif_folder = f'/share/klab/datasets/avs/population_codes/as02/sensor/erf/filter_0.2_200/', fif_file = f"as02{session_id_char}_et_epochs_info_saccade.fif"
        """
        subject_id = self.subject_id if subject_id is None else subject_id  # If we are on single subject level, this will be none and the class attribute will be used. In later processing stages where subjects are compared, the subject_id argument will be given.
        with open(f"data_files/session_metadata/session_datetimes/session_datetimes_subject_{subject_id}_dict.pkl", 'rb') as file:
            session_datetimes = pickle.load(file)
 
        # Get difference in days (as float) between sessions
        session_day_differences = self.recursive_defaultdict()
        for session_id_num in self.session_ids_num:
            og_session_date = session_datetimes[session_id_num]
            #logger.custom_info(f"Session {session_id_num} date: {og_session_date}")
            for session_comp_id_num in self.session_ids_num:
                if session_id_num != session_comp_id_num:
                    comp_session_date = session_datetimes[session_comp_id_num]

                    diff_hours =  round(abs((comp_session_date - og_session_date).total_seconds()) / 3600)
                    diff_days = round(diff_hours / 24)
                    session_day_differences[session_id_num][session_comp_id_num] = diff_days
                else:
                    session_day_differences[session_id_num][session_comp_id_num] = 0
        
        for session_id_num in self.session_ids_num:
            for session_comp_id_num in self.session_ids_num:
                assert session_day_differences[session_id_num][session_comp_id_num] == session_day_differences[session_comp_id_num][session_id_num], "Difference between Sessions inconsistent."

        return session_day_differences


    def recursive_defaultdict(self) -> dict:
        """
        Helper function to initialize a defaultdict that automatically adds missing intermediate dicts.
        """
        return defaultdict(self.recursive_defaultdict)


    def map_session_letter_id_to_num(self, session_id_letter: str) -> str:
        """
        Helper function to map the character id from a session to its number id.
        For example: Input "a" will return "1".
        """
        # Create mapping from letters 'a'-'j' to '1'-'11'
        session_mapping = {self.session_ids_char[num-1]: str(num) for num in range(1,11)}
        # Map letter to num
        session_id_num = session_mapping[session_id_letter]

        return session_id_num


    def read_dict_from_json(self, type_of_content: str, type_of_norm: str = None, predict_train_data:bool = False) -> dict:
        """
        Helper function to read json files of various content types into dicts. Not used for later data types because of exploding parameterization.
        """
        valid_types = ["combined_metadata", "meg_metadata", "crop_metadata", "mse_losses", "mse_losses_timepoint", "var_explained"]
        if type_of_content not in valid_types:
            raise ValueError(f"Function read_dict_from_json called with unrecognized type {type_of_content}.")

        if type_of_content in ["mse_losses", "mse_losses_timepoint"]:
            file_path_beginning = f"data_files/{self.lock_event}/mse_losses/{self.ann_model}/{self.module_name}/subject_{self.subject_id}"
            if type_of_content == "mse_losses":
                file_path_ending = f"norm_{type_of_norm}/mse_losses_{type_of_norm}_dict.json"
            else:  # mse_losses_timepoint
                file_path_ending = f"timepoints/norm_{type_of_norm}/mse_losses_timepoint_{type_of_norm}_dict.json"
            file_path = os.path.join(file_path_beginning, file_path_ending)
        elif type_of_content == "var_explained":
            file_path = f"data_files/{self.lock_event}/var_explained/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{type_of_norm}/predict_train_data_{predict_train_data}/var_explained_{type_of_norm}_dict.json"
        else:
            file_path = f"data_files/{self.lock_event}/metadata/{type_of_content}/subject_{self.subject_id}/{type_of_content}_dict.json"
            logger.custom_debug(f"Loading {type_of_content} from {file_path}.")
            
        try:
            with open(file_path, 'r') as data_file:
                logger.custom_debug(f"Loading dict from {file_path}")
                data_dict = json.load(data_file)
            return data_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"In Function read_dict_from_json: The file {file_path} does not exist.")

        return data_dict


    def save_dict_as_json(self, type_of_content: str, dict_to_store: dict, type_of_norm: str = None, predict_train_data:bool = False) -> None:
        """
        Helper function to store dicts of various content types as json files.
        """
        valid_types = ["combined_metadata", "meg_metadata", "crop_metadata", "mse_losses", "mse_losses_timepoint", "var_explained"]
        if type_of_content not in valid_types:
            raise ValueError(f"Function save_dict_as_json called with unrecognized type {type_of_content}.")

        # losses and variance explained
        if type_of_content in ["mse_losses", "mse_losses_timepoint", "var_explained"]:
            if type_of_content.startswith('mse'):
                if type_of_content == "mse_losses_timepoint":
                    timepoint_folder = "timepoints/"
                    timepoint_name = "_timepoints"
                else:
                    timepoint_folder = ""
                    timepoint_name = ""
                storage_folder = f"data_files/{self.lock_event}/mse_losses/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/{timepoint_folder}norm_{type_of_norm}"
            elif type_of_content == "var_explained":
                storage_folder = f"data_files/{self.lock_event}/var_explained/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{type_of_norm}/predict_train_data_{predict_train_data}"
            name_addition = f"_{type_of_norm}"
        # metadata
        elif type_of_content in ["combined_metadata", "meg_metadata", "crop_metadata"]:
            storage_folder = f'data_files/{self.lock_event}/metadata/{type_of_content}/subject_{self.subject_id}'
            name_addition = ""
            logger.custom_debug(f"Storing {type_of_content} in {storage_folder}.")

        os.makedirs(storage_folder, exist_ok=True)
        json_storage_file = f"{type_of_content}{name_addition}_dict.json"
        json_storage_path = os.path.join(storage_folder, json_storage_file)

        with open(json_storage_path, 'w') as file:
            logger.custom_debug(f"Storing dict to {json_storage_path}")
            # Serialize and save the dictionary to the file
            json.dump(dict_to_store, file, indent=4)


    def export_split_data_as_file(self, session_id: str, type_of_content: str, array_dict: Dict[str, np.ndarray], type_of_norm: str = None, ann_model: str = None, module: str = None) -> None:
        """
        Helper function to export train/test numpy arrays as .npz or .pt files.

        Parameters: 
            session_id (str): id of session that the arrays belong to
            type_of_content (str): Type of data in arrays. Allowed values: ["trial_splits", "crop_data", "meg_data", "torch_dataset", "ann_features"]
            np_array (Dict[str, ndarray]): Arrays in format split, array. split is "train" or "test".
        """
        valid_types = ["trial_splits", "crop_data", "meg_data", "torch_dataset", "ann_features", "ann_features_pca", "ann_features_pca_all_sessions_combined"]
        if type_of_content not in valid_types:
            raise ValueError(f"Function export_split_data_as_file called with unrecognized type {type_of_content}.")

        # Set file type
        file_type = ".pt" if type_of_content == "torch_dataset" else ".npy"
        
        # Add additional folder for norm and for model type and extraction layer for ann_features
        additional_model_folders = f"/{ann_model}/{module}/" if type_of_content.startswith("ann_features") else "/"
        # Add a folder for norm for meg data and folder that indicates an intermediate step in normalisation: Additional steps will later be performed across all sessions combined
        if type_of_content == "meg_data":
            additional_norm_folder = f"norm_{type_of_norm}/" 
            if type_of_norm.endswith("_intermediate"):
                intermediate_norm_folder = "/intermediate_norm_True"  
                type_of_norm = type_of_norm[:-len("_intermediate")]
            else:
                intermediate_norm_folder = ""
        else:
            additional_norm_folder = "" 
            intermediate_norm_folder = ""

        if type_of_content.endswith("_all_sessions_combined"):
            all_sessions_combined_folder = "/all_sessions_combined" 
            type_of_content = type_of_content.replace("_all_sessions_combined", "")
            session_folder = ""
        else:
            session_folder = f"/session_{session_id}"
            all_sessions_combined_folder = ""

        # Export train/test split arrays to .npz
        for split in array_dict:
            save_folder = f"data_files/{self.lock_event}/{type_of_content}{all_sessions_combined_folder}{additional_model_folders}{additional_norm_folder}{intermediate_norm_folder}subject_{self.subject_id}{session_folder}/{split}"  
            save_file = f"{type_of_content}{file_type}"
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, save_file)

            if file_type == ".npy":
                #if all_sessions_combined_folder != "":
                np.save(save_path, array_dict[split])
            else:
                torch.save(array_dict[split], save_path)
            logger.custom_debug(f"Exporting split data {type_of_content} to {save_path}")
            if split == "train" and (type_of_content == "crop_data" or type_of_content.startswith("ann_features")):
                logger.custom_debug(f"[Session {session_id}]: Train: Storing array of shape {array_dict[split].shape} to {save_path}")
            if split == "train" and type_of_content == "torch_dataset":
                logger.custom_debug(f"[Session {session_id}][Content TorchDataset]: Train: Saving dataset to {save_path}")

    
    def normalize_cross_session_preds_with_self_preds(self, fit_measures_by_session_by_timepoint: dict):
        """
        Normalizes the by-timepoint cross-session prediction performances based on the self-prediction performance of the predicted session.
        """
        for session_train_id in fit_measures_by_session_by_timepoint['session_mapping']:
            for session_pred_id in fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"]:
                if session_train_id != session_pred_id:
                    for timepoint_idx in fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"][session_pred_id]["timepoint"]:
                        timepoint_val_non_normalized = fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"][session_pred_id]["timepoint"][timepoint_idx]
                        timepoint_val_self_pred = fit_measures_by_session_by_timepoint['session_mapping'][session_pred_id]["session_pred"][session_pred_id]["timepoint"][timepoint_idx]

                        timepoint_val_normalized = timepoint_val_non_normalized - timepoint_val_self_pred
                        fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"][session_pred_id]["timepoint"][timepoint_idx] = timepoint_val_normalized
        # Now "normalize the self preds aswell. Needed to be kept constant before to apply same normalization to all sessions"
        for session_id in fit_measures_by_session_by_timepoint['session_mapping']:
            for timepoint_idx in fit_measures_by_session_by_timepoint['session_mapping'][session_id]["session_pred"][session_id]["timepoint"]:
                    fit_measures_by_session_by_timepoint['session_mapping'][session_id]["session_pred"][session_id]["timepoint"][timepoint_idx] = 0

        return fit_measures_by_session_by_timepoint


    def average_timepoint_data_per_session(self, fit_measures_by_session_by_timepoints:dict):
        """
        Converts cross-session fit measures at timepoint level to session level by simply averaging timepoints for each predicted session (and for each train session ofc).
        """
        fit_measures_by_session_averaged_over_timepoints = self.recursive_defaultdict()
        for session_train_id, fit_measures_train_session in fit_measures_by_session_by_timepoints['session_mapping'].items():
            for session_pred_id, fit_measures_pred_session in fit_measures_train_session["session_pred"].items():
                fit_sum_over_timepoints = sum(timepoint_value for timepoint_value in fit_measures_pred_session["timepoint"].values())
                n_timepoints_fit = len(fit_measures_pred_session["timepoint"].keys())

                fit_averaged_over_timepoints = fit_sum_over_timepoints / n_timepoints_fit
                fit_measures_by_session_averaged_over_timepoints['session_mapping'][session_train_id]["session_pred"][session_pred_id] = fit_averaged_over_timepoints

        return fit_measures_by_session_averaged_over_timepoints


    def calculate_fit_by_distances(self, fit_measures_by_session: dict, timepoint_level_input: bool, average_within_distances:bool, include_0_distance:bool = False, subject_id:str = None):
        """
        Aligns fit measures by distances. Input should be a dict containing all cross-session fit measures (on a timepoint level).

        subject_id (str): needs to be specified if we are currently operating over all subjects and self.subject is not accurate
        """
        session_day_differences = self.get_session_date_differences(subject_id=subject_id)

        if timepoint_level_input:
            fit_measures_by_session = self.average_timepoint_data_per_session(fit_measures_by_session)

        # Calculate fit measures relative to distance in time between train and pred session
        fit_by_distances = {}
        for session_train_id, fit_measures_train_session in fit_measures_by_session['session_mapping'].items():
            for session_pred_id, fit_measure_pred_session in fit_measures_train_session["session_pred"].items():
                if include_0_distance or session_train_id != session_pred_id:
                    distance = session_day_differences[session_train_id][session_pred_id] if session_train_id != session_pred_id else 0

                    if average_within_distances:
                        if distance not in fit_by_distances:
                            fit_by_distances[distance] = {"fit_measure": fit_measure_pred_session, "num_measures": 1}
                        else:
                            fit_by_distances[distance]["fit_measure"] += fit_measure_pred_session
                            fit_by_distances[distance]["num_measures"] += 1
                    else:
                        if distance not in fit_by_distances:
                            fit_by_distances[distance] = [fit_measure_pred_session]
                        else:
                            fit_by_distances[distance].append(fit_measure_pred_session)

        if average_within_distances:               
            for distance in fit_by_distances:
                fit_measure_distance = fit_by_distances[distance]["fit_measure"]
                num_measures_distance = fit_by_distances[distance]["num_measures"]
                fit_by_distances[distance]["fit_measure"] = fit_measure_distance / num_measures_distance

        return fit_by_distances

    
    def load_split_data_from_file(self, session_id_num: str, type_of_content: str, type_of_norm:str = None, ann_model: str = None, module: str = None) -> dict:
        """
        Helper function to load the split for a given session.

        Parameters: 
            session_id_num (str): id of session that the arrays belong to
            type_of_content (str): Type of data in arrays
        """
        valid_types = ["trial_splits", "crop_data", "meg_data", "torch_dataset", "ann_features", "ann_features_pca", "ann_features_pca_all_sessions_combined"]
        if type_of_content not in valid_types:
            raise ValueError(f"Function load_split_data_from_file called with unrecognized type {type_of_content}.")

        file_type = ".pt" if type_of_content == "torch_dataset" else ".npy"

        # Add additional folder for norm and for model type and extraction layer for ann_features
        additional_model_folders = f"/{ann_model}/{module}/" if type_of_content.startswith("ann_features") else "/"
        # Add a folder for norm for meg data and folder that indicates an intermediate step in normalisation: Additional steps will later be performed across all sessions combined
        additional_norm_folder, intermediate_norm_folder = "", ""
        if type_of_content == "meg_data":
            additional_norm_folder = f"norm_{type_of_norm}/" 
            if type_of_norm.endswith("_intermediate"):
                intermediate_norm_folder = "/intermediate_norm_True"  
                type_of_norm = type_of_norm[:-len("_intermediate")]

        all_sessions_combined_folder, session_folder = "", ""
        if type_of_content.endswith("_all_sessions_combined"):
            all_sessions_combined_folder = "/all_sessions_combined" 
            type_of_content = type_of_content.replace("_all_sessions_combined", "")
        else:
            session_folder = f"/session_{session_id_num}"

        split_dict = {}
        for split in ["train", "test"]:
            # Load split trial array
            split_path = f"data_files/{self.lock_event}/{type_of_content}{all_sessions_combined_folder}{additional_model_folders}{additional_norm_folder}subject_{self.subject_id}{session_folder}/{split}/{type_of_content}{file_type}"  
            split_data = np.load(split_path) if file_type == ".npy" else torch.load(split_path)
            split_dict[split] = split_data

            logger.custom_debug(f"Loading split data {type_of_content} from {split_path}")
            if split == "train":
                if type_of_content == "crop_data" or type_of_content.startswith("ann_features"):
                    logger.custom_debug(f"[Session {session_id_num}][Content {type_of_content}]: Train: Loaded array of shape {split_data.shape} from {split_path}")
                elif type_of_content == "torch_dataset":
                    logger.custom_debug(f"[Session {session_id_num}][Content TorchDataset]: Train: Loading dataset from {split_path}")
                
        return split_dict

    
    def save_plot_as_file(self, plt, plot_folder: str, plot_file: str, plot_type: str = None):
        """
        Helper function to save a plot as file.
        """
        os.makedirs(plot_folder, exist_ok=True)
        plot_path = os.path.join(plot_folder, plot_file)
        plt.savefig(plot_path)
        # Mne plots and some figures cannot be closed explicity, close all others to avoid memory leak
        try:
            plt.close()
        except AttributeError as e:  # Caught if type of plot cannot be closed
            pass


    def normalize_array(self, data: np.ndarray, normalization:str, n_channels:int = None, session_id:str = None):
        """
        Helper function to normalize meg
        normalization options: mean centered per channel and per timepoint, min-max over complete session, robust scaling, no normalization
                                ["min_max", "mean_centered_ch_t", "robust_scaling", "no_norm", "median_centered_ch_t"]
        """
        if session_id != None:
            logger.custom_debug(f"[session {session_id}] data.shape: {data.shape}.")  

        match normalization: 
            case "range_-1_to_1":
                min_val = -1
                max_val = 1
                data_min = np.min(data)
                data_max = np.max(data)
                normalized_data = min_val + (data - data_min) * (max_val - min_val) / (data_max - data_min)
                for normalized_value in np.nditer(normalized_data):
                    assert normalized_value >= -1 and normalized_value <= 1, f"normalization {normalization} did not work correctly"

            case "mean_centered_ch":
                # 0 centered by mean for each channel over all epochs and timepoints
                means_per_sensor = np.mean(data, axis=(0,2)).reshape(1,n_channels,1)
                normalized_data = data - means_per_sensor

            case "mean_centered_vox":
                # 0 centered by mean for each voxel over all epochs and timepoints (identical to mean_centered_ch for voxels instead of channels, added to avoid confusion due to name; n_channels is in thise n_voxels)
                means_per_voxel = np.mean(data, axis=(0,2)).reshape(1,n_channels,1)
                normalized_data = data - means_per_voxel

            case "min_max":
                data_min = data.min()
                data_max = data.max()
                normalized_data = (data - data_min) / (data_max - data_min)

            case "robust_scaling":
                median = np.median(data, axis=None)  # Median across flattened array
                q75, q25 = np.percentile(data, [75, 25], axis=None)
                iqr = q75 - q25
                normalized_data = (data - median) / iqr  # Subtract median and divide by IQR

            case "z_score":
                means = np.mean(data, axis=None)
                std_devs = np.std(data, axis=None)

                # Use an epsilon to prevent division by zero
                epsilon = 1e-100
                std_devs += epsilon
    
                normalized_data = (data - means) / std_devs

            case "mean_centered_ch_t":
                means = np.mean(data, axis=0)  # Compute means for each channel and timepoint, averaged over all epochs
                normalized_data = data - means  # Subtract the mean to center the data

            case "median_centered_ch_t":
                median = np.median(data, axis=0)  # Compute median for each channel and timepoint, averaged over all epochs
                normalized_data = data - median  # Subtract the median to center the data

            case "robust_scaling_ch_t":
                medians = np.median(data, axis=0)  # Median across epochs
                q75, q25 = np.percentile(data, [75, 25], axis=0)
                iqr = q75 - q25
                normalized_data = (data - medians) / iqr  # Subtract medians and divide by IQR

            case "min_max_ch_t":
                data_min = data.min(axis=0)
                data_max = data.max(axis=0)
                normalized_data = (data - data_min) / (data_max - data_min)

            case "z_score_ch_t":
                means = np.mean(data, axis=0)
                std_devs = np.std(data, axis=0)

                # Use an epsilon to prevent division by zero
                epsilon = 1e-100
                std_devs += epsilon

                normalized_data = (data - means) / std_devs

            case "no_norm":
                normalized_data = data

            case _:
                raise ValueError(f"normalize_array called with unrecognized type {normalization}")

        if (normalized_data == data).all() and normalization != "no_norm":
            logger.warning(f"[WARNING][normalize_array]: data the same before and after norm {normalization}")

        return normalized_data



class MetadataHelper(BasicOperationsHelper):
    def __init__(self, crop_size, **kwargs):
        super().__init__(**kwargs)

        self.crop_size = crop_size
        if crop_size == 100:
            self.crop_metadata_path = f"/share/klab/camme/avs-encoding/data/image_patches/crops_{crop_size}/as{self.subject_id}/metadata/as{self.subject_id}_crops_metadata.csv"
        else:
            self.crop_metadata_path = f"/share/klab/datasets/avs/input/fixation_crops/avs_meg_fixation_crops_scene_{crop_size}/metadata/as{self.subject_id}_crops_metadata.csv"
            #self.crop_metadata_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_{crop_size}/metadata/as{self.subject_id}_crops_metadata.csv"
        self.meg_metadata_folder = f"/share/klab/datasets/avs/population_codes/as{self.subject_id}/sensor/erf/filter_0.2_200/ica"  # f"/share/klab/datasets/avs/population_codes/as{self.subject_id}/sensor/filter_0.2_200"


    def extract_arousal_values(self):
        """
        Extracts arousal values for all sessions for the current subject. Awaiting response from Philipp detailing where this data can be found.
        """
        #crop_metadata_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_{crop_size}/metadata/as{self.subject_id}_crops_metadata.csv"
        pass


    def create_combined_metadata_dict(self, investigate_missing_metadata=False) -> None:
        """
        Creates the combined metadata dict with timepoints that can be found in both meg and crop metadata for the respective session and trial.
        """

        # Read crop and meg metadata from json
        crop_metadata = self.read_dict_from_json(type_of_content="crop_metadata")
        meg_metadata = self.read_dict_from_json(type_of_content="meg_metadata")

        # Store information about timepoints that are present in both meg and crop data

        # Use defaultdict to automatically create missing keys
        combined_metadata_dict = self.recursive_defaultdict()

        total_combined_datapoints = 0

        crop_missing_trials = []
        for session_id in meg_metadata["sessions"]:
            combined_datapoints_session = 0
            meg_index = 0
            for trial_id in meg_metadata["sessions"][session_id]["trials"]:
                for timepoint_id in meg_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]:
                    # For each timepoint in the meg metadata: Check if this timepoint is in the crop metadata and if so store it
                    try:
                        crop_identifier = crop_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["crop_identifier"]
                        sceneID = crop_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["sceneID"]

                        combined_metadata_dict["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["crop_identifier"] = crop_identifier
                        combined_metadata_dict["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["sceneID"] = sceneID
                        combined_metadata_dict["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["meg_index"] = meg_index

                        total_combined_datapoints += 1
                        combined_datapoints_session += 1
                    except Exception as e:
                        if investigate_missing_metadata and trial_id not in crop_missing_trials:
                            #logger.custom_debug(f"[Session {session_id}][Trial {trial_id}]: Within this Trial, data for at least one timepoint exists only in the meg-, and not the crop metadata.")
                            crop_missing_trials.append(trial_id)
                    meg_index += 1
            logger.custom_debug(f"[Session {session_id}]: combined_datapoints_session: {combined_datapoints_session}")

        if investigate_missing_metadata:
            logger.custom_debug(f"Number of trials for which least one timepoint exists only in the meg-, but not in the crop metadata: {len(crop_missing_trials)}")

        logger.custom_info(f"total_combined_datapoints: {total_combined_datapoints}")

        if investigate_missing_metadata:
            meg_missing_trials = []
            # Do the same from the perspective of the crop_metadata to find datapoints that only exist in the crop-, but not the meg-metadata
            for session_id in crop_metadata["sessions"]:
                for trial_id in crop_metadata["sessions"][session_id]["trials"]:
                    # For each timepoint in the crop metadata: Check if this timepoint is in the meg metadata and if not store the trial
                    for timepoint_id in crop_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]:
                        try:
                            meg_true = meg_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["meg"]
                        except Exception:
                            if trial_id not in meg_missing_trials:
                                #logger.custom_debug(f"[Session {session_id}][Trial {trial_id}]: Within this Trial, data for at least one timepoint exists only in the crop-, and not the meg metadata.")
                                meg_missing_trials.append(trial_id)
            logger.custom_debug(f"Number of trials for which least one timepoint exists only in the crop-, but not in the meg metadata: {len(meg_missing_trials)}")
            
        # Export dict to json 
        self.save_dict_as_json(type_of_content="combined_metadata", dict_to_store=combined_metadata_dict)


    def create_meg_metadata_dict(self) -> None:
        """
        Creates the meg metadata dict for the participant and stores it.
        """
        # Define which column holds the relevant data to match crops and meg epoch based on a unique identifier within a trial
        time_column = "fix_sequence"
        #time_column = "end_time" if self.lock_event == "saccade" else "time_in_trial"  # deorecate, does not work for subject 04 due to measurement issue

        logger.custom_debug(f"reading meg metadata from {self.meg_metadata_folder}.")

        data_dict = {"sessions": {}}
        # Read metadata for each session from csv
        for session_id_letter in self.session_ids_char:
            session_id_num = self.map_session_letter_id_to_num(session_id_letter)

            meg_metadata_file = f"as{self.subject_id}{session_id_letter}_et_epochs_metadata_{self.lock_event}.csv"
            meg_metadata_path = os.path.join(self.meg_metadata_folder, meg_metadata_file)

            # Read metadata from csv 
            df = pd.read_csv(meg_metadata_path, delimiter=";")
            data_dict["sessions"][session_id_num] = {"trials": {}}

            for counter, (index, row) in enumerate(df.iterrows(), start=1):
                trial_id = int(row["trial"])
                timepoint = row[f"{time_column}"]
                
                # Create dict to store values for current trial if this is the first timepoint in the trial
                if trial_id not in data_dict["sessions"][session_id_num]["trials"]:
                    data_dict["sessions"][session_id_num]["trials"][trial_id] = {"timepoints": {}}
                else:
                    # Make sure there are no duplicates: There should be no stored value for the current timepoint yet
                    try:
                        assert data_dict["sessions"][session_id_num]["trials"][trial_id]["timepoints"].get(timepoint, None) == None
                    except AssertionError as err:
                        logger.error(f"Found multiple datapoints with the same time_column in session {session_id_num}, trial {trial_id}")
                        raise AssertionError

                # Store availability of current timepoint in trial
                data_dict["sessions"][session_id_num]["trials"][trial_id]["timepoints"][timepoint] = {"meg":True}
            logger.custom_debug(f"Num Rows in MEG metadata: {counter}")

        # Export dict to json 
        self.save_dict_as_json(type_of_content="meg_metadata", dict_to_store=data_dict)


    def create_crop_metadata_dict(self) -> None:
        """
        Creates the crop metadata dict for the participant and stores it.
        """
        # Define which column holds the relevant data to match crops and meg epoch based on a unique identifier within a trial
        time_column = "fix_sequence"
        #time_column = "start_time" if self.lock_event == "saccade" else "time_in_trial"  # deprecated, matching based on time_in_trial invalid for subject 04 where a measurement error occured
        

        logger.custom_debug(f"reading crop metadata from {self.crop_metadata_path}.")

        df = pd.read_csv(self.crop_metadata_path, index_col = 'crop_filename')  # Read data from csv and set index to crop filename (crop_identifier before .png)
        df = df[df.index.notnull()]  # Remove rows/fixations without crop identifiers
        num_sessions = df["session"].max() #10 sessions 

        data_dict = {"sessions": {}}
        for nr_session in range(1,num_sessions+1):
            data_dict["sessions"][nr_session] = {}

            # Filter dataframe by session 
            session_df = df[df["session"] == nr_session]
            # Filter dataframe by type of recording (we are only interested in scene recordings, in the meg file I am using there is no data for caption of microphone recordings (or "nothing" recordings))
            session_df = session_df[session_df["recording"] == "scene"]

            # Get list of all trials in this session
            trial_numbers = list(map(int, set(session_df["trial"].tolist())))

            data_dict["sessions"][nr_session]["trials"] = {}
            # For each trial in the session
            for nr_trial in trial_numbers:
                # Filter session dataframe by trial
                trial_df = session_df[session_df["trial"] == nr_trial]

                # Get list of all timepoints in this trial
                timepoints = trial_df[f"{time_column}"].tolist()

                data_dict["sessions"][nr_session]["trials"][nr_trial] = {"timepoints": {}}
                # For each timepoint in this trial
                for timepoint in timepoints:
                    # Filter trial dataframe by timepoint
                    timepoint_df = trial_df[trial_df[f"{time_column}"] == timepoint]

                    try:
                        assert len(timepoint_df) == 1
                    except AssertionError as err:
                        logger.error(f"Found multiple datapoints with the same time_column in session {nr_session}, trial {nr_trial}")
                        raise AssertionError

                    # get sceneID for this timepoint
                    sceneID = timepoint_df['sceneID'].iloc[0]

                    data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint] = {}
                    data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint]["crop_identifier"] = timepoint_df.index.tolist()[0][:-4]  # slice with [0] to get single element instead of list, then slice off ".png" at the end
                    data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint]["sceneID"] = sceneID
        
        # Export dict to json 
        self.save_dict_as_json(type_of_content="crop_metadata", dict_to_store=data_dict)


        
class DatasetHelper(MetadataHelper):
    def __init__(self, normalizations:list,  chosen_channels:list , timepoint_min: int, timepoint_max:int, **kwargs):
        super().__init__(**kwargs) 

        self.normalizations = normalizations
        self.chosen_channels = chosen_channels
        self.timepoint_min = timepoint_min
        self.timepoint_max = timepoint_max

        if self.crop_size == 100:
            self.crop_folder_path = f"/share/klab/camme/avs-encoding/data/image_patches/crops_{self.crop_size}/as{self.subject_id}/crops/original/as{self.subject_id}"
        else:
            self.crop_folder_path = f"/share/klab/datasets/avs/input/fixation_crops/avs_meg_fixation_crops_scene_{self.crop_size}/crops/as{self.subject_id}"
            #self.crop_folder_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_{self.crop_size}/crops/as{self.subject_id}"
        self.coco_scenes_path = "/share/klab/psulewski/psulewski/active-visual-semantics/input/mscoco_scenes"


    def test_scene_repetition(self):
        """
        Tests if any scene is repeated in multiple trials for the subject over all sessions.
        """
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")
        all_scenes_n = self.recursive_defaultdict()  # Count occurances of all scenes over all sessions
        for session_id in combined_metadata["sessions"]:
            for trial_id in combined_metadata["sessions"][session_id]["trials"]:
                # Get sceneID of trial based on first timepoint
                first_timepoint = list(combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"].keys())[0]
                scene_id = combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][first_timepoint]["sceneID"]
                # Add count for scene id
                if scene_id not in all_scenes_n.keys():
                    all_scenes_n[scene_id] = 1
                else:
                    print(f"[Subject {self.subject_id}]: Scene {scene_id} occurs more than once!")
                    all_scenes_n[scene_id] += 1

        print(f"Num scenes {self.subject_id}: {len(all_scenes_n.keys())}")

    def aggregate_test_splits_into_semantic_clusters(self):
        """
        DEPRECATED:
        Creates a dict where for the test split of each session, the crops are within their semantic cluster group
        60 Clusters: 0 to 59
        """
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")
        semantic_clusters_path = "/share/klab/datasets/avs/input/scene_sampling_MEG/scenes-per-sub-active-visual-semantics-MEG.csv"
        semantic_clusters_df = pd.read_csv(semantic_clusters_path, delimiter='|', on_bad_lines='warn')

        def extract_cluster_from_scene_id(scene_id:float):
            df_scene_id_idx = semantic_clusters_df.index[semantic_clusters_df['cocoID'] == int(scene_id)].tolist()[0]  # Takes one df index out of all occurances of the scene_id in the dataset, based on which corresponding cluster and other info can be extracted 
            cluster_id = semantic_clusters_df['cluster'][df_scene_id_idx]  # int between 0 and 59

            return cluster_id

        test_split_crops_by_session_by_cluster = self.recursive_defaultdict()
        for session_id in self.session_ids_num:
            trials_test_split_dict = self.load_split_data_from_file(session_id_num=session_id, type_of_content="trial_splits")['test']
            for nr_trial, trial_id in enumerate(trials_test_split_dict):
                for timepoint_nr, (timepoint_id, timepoint_dict) in enumerate(combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"].items()):
                    if timepoint_nr == 0:
                        # Once per trial: Extract corresponding scene id and semantic cluster. Create key for cluster in session dict if not existent already
                        scene_id_trial = timepoint_dict["sceneID"]  
                        cluster_id_trial = extract_cluster_from_scene_id(scene_id_trial)
                        if cluster_id_trial not in test_split_crops_by_session_by_cluster["session"][session_id]["cluster"]:
                            test_split_crops_by_session_by_cluster["session"][session_id]["cluster"][cluster_id_trial] = []
                    # Store crop identifier in correct cluster
                    test_split_crops_by_session_by_cluster["session"][session_id]["cluster"][cluster_id_trial].append(timepoint_dict["crop_identifier"])

        # Check if I have at least 5 images for each semantic cluster in each sessions test split
        #for session_id, session_dict in test_split_crops_by_session_by_cluster["session"].items():
        #    for cluster_id, crop_id_list in session_dict["cluster"].items():
        #        if len(crop_id_list) < 5:
        #            print(f"Sesssion {session_id}, Cluster {cluster_id} contains only {len(crop_id_list)} images.")
        # No I do not, minimum is 1. May want to redo the splits with the clusters in mind, for the moment use 1 image from each cluster per session -> 600 images

        # Collect even distribution of images over clusters and session. The (10) images in each cluster will be ordered by session
        simulation_crops_by_cluster = self.recursive_defaultdict()  
        for session_id, session_dict in test_split_crops_by_session_by_cluster["session"].items():
            for cluster_id, crop_id_list in session_dict["cluster"].items():
                # Select random element from this session and cluster
                selected_crop_id = np.random.choice(crop_id_list)
                if cluster_id not in simulation_crops_by_cluster["cluster"]:
                    simulation_crops_by_cluster["cluster"][cluster_id] = []

                if cluster_id == 40:
                    print(f"selected_crop_id: {selected_crop_id}")

                # Retrieve and store image (potentially replace with complete image instead of crop)
                crop_filename = ''.join([selected_crop_id, ".png"])
                crop_path = os.path.join(self.crop_folder_path, crop_filename)
                crop = imageio.imread(crop_path)

                simulation_crops_by_cluster["cluster"][cluster_id].append(crop)
            
        print(f"len(list(simulation_crops_by_cluster['cluster'].keys())): {len(list(simulation_crops_by_cluster['cluster'].keys()))}")
                    
        for cluster_id, cluster_arr in simulation_crops_by_cluster["cluster"].items():
            assert len(cluster_arr) == 10, f"Wrong length for cluster {cluster_id}. Len: {len(cluster_arr)}"

        ### Not all test splits of all session contain atleast one crop belonging to each cluster! Deprecated until splits adjusted (if crops deemed sensible for simulation of MEG responses) ###
        sys.exit("Completed aggregate_test_splits_into_semantic_clusters.")


    def create_simulation_scene_dataset(self, n_scenes_per_cluster:int):
        """
        Chooses 5 scenes from each cluster that are not contained in any sessions train set.
        """
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")
        semantic_clusters_path = "/share/klab/datasets/avs/input/scene_sampling_MEG/scenes-per-sub-active-visual-semantics-MEG.csv"
        semantic_clusters_df = pd.read_csv(semantic_clusters_path, delimiter='|', on_bad_lines='warn')

        def extract_cluster_from_scene_id(scene_id:float):
            df_scene_id_idx = semantic_clusters_df.index[semantic_clusters_df['cocoID'] == int(scene_id)].tolist()[0]  # Takes one df index out of all occurances of the scene_id in the dataset, based on which corresponding cluster and other info can be extracted 
            cluster_id = semantic_clusters_df['cluster'][df_scene_id_idx]  # int between 0 and 59

            return cluster_id

        def collect_scene_ids_by_cluster():
            test_split_scene_ids_by_cluster = self.recursive_defaultdict()
            for session_id in self.session_ids_num:
                trials_test_split_dict = self.load_split_data_from_file(session_id_num=session_id, type_of_content="trial_splits")['test']
                for nr_trial, trial_id in enumerate(trials_test_split_dict):
                    # We don't care about the timepoints since the scene_id is the same for all. Select the first timepoint
                    first_timepoint = next(iter(combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]))
                    scene_id_trial = str(int(combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][first_timepoint]["sceneID"])) # convert xxxx.x to "xxxx"
                    cluster_id_trial = extract_cluster_from_scene_id(scene_id_trial)
                    if cluster_id_trial not in test_split_scene_ids_by_cluster["cluster"]:
                        test_split_scene_ids_by_cluster["cluster"][cluster_id_trial] = [scene_id_trial]
                    # Select n scenes for each cluster total
                    elif len(test_split_scene_ids_by_cluster["cluster"][cluster_id_trial]) < n_scenes_per_cluster:
                        test_split_scene_ids_by_cluster["cluster"][cluster_id_trial].append(scene_id_trial)
                
                # Stop as soon as we have n scenes for each cluster
                full_clusters = 0
                for cluster_id, cluster_arr in test_split_scene_ids_by_cluster["cluster"].items():
                    if len(cluster_arr) == n_scenes_per_cluster:
                        full_clusters += 1
                
                if full_clusters == 60:
                    return test_split_scene_ids_by_cluster
            return test_split_scene_ids_by_cluster

        test_split_scene_ids_by_cluster = collect_scene_ids_by_cluster()

        for cluster_id, cluster_arr in test_split_scene_ids_by_cluster["cluster"].items():
            assert len(cluster_arr) == n_scenes_per_cluster, f"Cluster {cluster_id} contains {len(cluster_arr)} scenes."

        def extract_scene_image_from_coco_folder(scene_id_jpg_name:str):
            """
            Returns array of image if scene exists in any of the split folders. Throws error otherwise.
            """
            for split_folder in ["train2017", "test2017", "val2017"]:
                split_path = os.path.join(self.coco_scenes_path, split_folder)
                potential_image_path = os.path.join(split_path, scene_id_jpg_name)
                if os.path.isfile(potential_image_path):
                    scene_array = imageio.imread(potential_image_path)
                    return scene_array
            raise ValueError(f"Could not find {scene_id_jpg_name}")

        # Sort dict by clusters, 0 to 59
        sorted_test_split_scene_ids_by_cluster = {"cluster": dict(sorted(test_split_scene_ids_by_cluster["cluster"].items(), key=lambda item: int(item[0])))}

        # Collect coco images from jpg
        numpy_images_by_cluster = []
        for cluster_id, cluster_arr in sorted_test_split_scene_ids_by_cluster["cluster"].items():
            scenes_in_cluster = []
            for scene_id in cluster_arr:
                # Pad scene id with 0s to fit jpg names. Examples: 000000000139.jpg, 000000023392.jpg
                scene_id_jpg_name = "".join([scene_id.zfill(12), ".jpg"])
                # Extract array from coco folder
                scene_array = extract_scene_image_from_coco_folder(scene_id_jpg_name)
                # Resize; coco images don't have uniform shape by default. Alexnet originally designed for 224x224
                scene_array_resized = cv2.resize(scene_array, (224, 224), interpolation=cv2.INTER_LINEAR)

                scenes_in_cluster.append(scene_array_resized)
            if len(scenes_in_cluster) != n_scenes_per_cluster:
                raise ValueError(f"cluster_id has {len(scenes_in_cluster)} scenes")
            numpy_images_by_cluster.append(np.array(scenes_in_cluster))
        numpy_images_by_cluster = np.array(numpy_images_by_cluster)

        #print(f"numpy_images_by_cluster.shape: {numpy_images_by_cluster.shape}")  # (60, 5, 224, 224, 3)
        
        # Store simulation dataset
        save_folder = f"data_files/{self.lock_event}/simulation_scenes/scenes_numpy/subject_{self.subject_id}"  
        save_file = f"simulation_scenes_by_clusters.npy"
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, save_file)
        np.save(save_path, numpy_images_by_cluster)


    def create_crop_dataset(self, debugging=False) -> None:
        """
        Creates the crop dataset with all crops in the combined_metadata (crops for which meg data exists)
        """
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")

        datapoints_by_session_and_split = {"sessions": {}}
        # For each session: create crop datasets based on respective splits
        for session_id in self.session_ids_num:
            trials_split_dict = self.load_split_data_from_file(session_id_num=session_id, type_of_content="trial_splits")  # Get train/test split based on trials (based on scenes)

            crop_split = {"train": [], "test": []}
            datapoints_by_session_and_split["sessions"][session_id] = {"splits": {"train": 0, "test": 0}}
            for split in crop_split:
                crop_split_index = 0
                # Iterate over split by trials (not over combined_metadata as before)
                # In this fashion, the first element in the crop and meg dataset of each split type will surely be the first element in the array of trials for that split
                for nr_trial, trial_id in enumerate(trials_split_dict[split]):
                    if debugging:
                        timepoints_in_trial = list(combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"].keys())
                        if len(timepoints_in_trial) > 10:
                            logger.custom_debug(f"[Session {session_id}][{split} split][Trial {trial_id}/Index {nr_trial}]: Timepoints found in metadata: {timepoints_in_trial}")
                            pass

                    # Get timepoints and corresponding crops from combined_metadata
                    for timepoint_id in combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]:
                        crop_filename = ''.join([combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["crop_identifier"], ".png"])
                        crop_path = os.path.join(self.crop_folder_path, crop_filename)

                        crop = imageio.imread(crop_path)
                        crop_split[split].append(crop)

                        if debugging and session_id in ["2", "5", "8"] and crop_split_index in [0, 10, 100, 1000]:
                            save_folder = f"data_files/{self.lock_event}/debugging/crop_data/numpy_dataset/session_{session_id}/{split}/crop_split_index_{crop_split_index}"
                            os.makedirs(save_folder, exist_ok=True)
                            save_path = os.path.join(save_folder, "crop_image_numpy")
                            np.save(save_path, crop)
                            assert np.all(crop_split[split][crop_split_index] == crop), "Storing the wrong crop_split index"

                        datapoints_by_session_and_split["sessions"][session_id]["splits"][split] += 1
                        crop_split_index +=1

            # Convert to numpy array
            for split in crop_split:
                crop_split[split] = np.stack(crop_split[split], axis=0)
                logger.custom_info(f"[Session {session_id}][{split} split]: Crop Numpy dataset is array of shape {crop_split[split].shape}")

            # Export numpy array to .npz
            self.export_split_data_as_file(session_id=session_id, 
                                        type_of_content="crop_data",
                                        array_dict=crop_split)

        if debugging:
            for session_id in datapoints_by_session_and_split["sessions"]:
                n_datapoints_session = 0
                for split in datapoints_by_session_and_split["sessions"][session_id]["splits"]:
                    n_datapoints_split = datapoints_by_session_and_split["sessions"][session_id]["splits"][split]
                    n_datapoints_session += n_datapoints_split
                    logger.custom_debug(f"Session {session_id} split {split} Num Datapoints: {n_datapoints_split}")
                logger.custom_debug(f"Session {session_id} Total Datapoints: {n_datapoints_session}")           


    def create_meg_dataset(self, use_ica_cleaned_data=True, interpolate_outliers=False, clip_outliers=True) -> None:
        """
        Creates the crop dataset with all crops in the combined_metadata (crops for which meg data exists)
        """
        if interpolate_outliers and clip_outliers:
            raise ValueError("create_meg_dataset called with invalid parameter configuration. Can either clip or interpolate eithers, not both.")
        if "mean_centered_voxel_then_global_robust_scaling" in self.normalizations:
            raise ValueError("create_meg_dataset is not designed to perform normalization mean_centered_voxel_then_global_robust_scaling")
        q_bottom, q_top = 0.3, 99.7  # set default values for clipping

        # Read combined and meg metadata from json
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")
        meg_metadata = self.read_dict_from_json(type_of_content="meg_metadata")

        meg_data_folder_beginning = f"/share/klab/datasets/avs/population_codes/as{self.subject_id}/sensor"
        if use_ica_cleaned_data:
            meg_data_folder_ending = "erf/filter_0.2_200/ica"
        else:
            meg_data_folder_ending = "filter_0.2_200"
        meg_data_folder = os.path.join(meg_data_folder_beginning, meg_data_folder_ending)

        selected_channel_indices = self.get_relevant_meg_channels(chosen_channels=self.chosen_channels)  # Get selected sensors to filter from meg file
        logger.custom_debug(f"selected_channel_indices: {selected_channel_indices}")

        # Debugging:
        n_epochs_two_step_norm = {"train": 0, "test": 0}

        for session_id_char in self.session_ids_char:
            session_id_num = self.map_session_letter_id_to_num(session_id_char)
            logger.custom_debug(f"Creating meg dataset for session {session_id_num}")
            # Load session MEG data from .h5
            meg_data_file = f"as{self.subject_id}{session_id_char}_population_codes_{self.lock_event}_500hz_masked_False.h5"
            with h5py.File(os.path.join(meg_data_folder, meg_data_file), "r") as f:
                meg_data = {}
                meg_data["grad"] = f['grad']['onset']  # shape participant 2, session a saccade: (2945, 204, 601), fixation: (2874, 204, 401) 
                meg_data["mag"] = f['mag']['onset']  # shape participant 2, session a saccade: (2945, 102, 601), fixation: (2874, 102, 401)

                logger.custom_debug(f"self.lock_event: {self.lock_event}")
                logger.custom_debug(f"H5 f.attrs['times']: {f.attrs['times']}")
                logger.custom_debug(f"H5 len(f.attrs['times']): {len(f.attrs['times'])}")

                num_meg_timepoints = meg_data['grad'].shape[0]

                logger.custom_debug(f"[Session {session_id_num}]: Pre filtering: meg_data['grad'].shape: {meg_data['grad'].shape}")
                logger.custom_debug(f"[Session {session_id_num}]: Pre filtering: meg_data['mag'].shape: {meg_data['mag'].shape}")

                # Filter considered meg data based on sensor type
                for sensor_type in selected_channel_indices:
                    if selected_channel_indices[sensor_type]:  # Checks if sensor type has been selected
                        channel_indices = list(selected_channel_indices[sensor_type]["sensor_index_within_type"].keys())
                        meg_data[sensor_type] = meg_data[sensor_type][:,channel_indices,:]
                    else:
                        del meg_data[sensor_type]
                if "grad" not in selected_channel_indices and "mag" not in selected_channel_indices:
                    raise ValueError("Neither mag or grad channels selected.")

                # Filter considered meg data based on relevant timepoints
                for sensor_type in meg_data:
                    meg_data[sensor_type] = meg_data[sensor_type][:,:,self.timepoint_min:self.timepoint_max+1]

                # Create datasets based on specified normalizations
                for normalization in self.normalizations:
                    # Set normalization that is to be performed per session; some normalizations require additional global computations across all sessions
                    if normalization == "mean_centered_ch_then_global_robust_scaling":
                        normalization_stage = "mean_centered_ch"
                    elif normalization == "global_robust_scaling":
                        normalization_stage = "no_norm"
                    else:
                        normalization_stage = normalization
                    # Debugging
                    if session_id_num == "1":
                        for sensor_type in selected_channel_indices:
                            if selected_channel_indices[sensor_type]:
                                logger.custom_info(f"[Session {session_id_num}]: Post filtering: meg_data['{sensor_type}'].shape: {meg_data[sensor_type].shape}")

                    meg_data_norm = {}
                    # Normalize grad and mag independently
                    for sensor_type in meg_data:
                        n_channels = meg_data[sensor_type].shape[1]
                        meg_data_norm[sensor_type] = self.normalize_array(np.array(meg_data[sensor_type]), normalization=normalization_stage, session_id=session_id_num, n_channels=n_channels)

                    # Combine grad and mag data
                    if selected_channel_indices["grad"] and selected_channel_indices["mag"]:
                        logger.custom_debug("Using both grad and mag data.")
                        combined_meg = np.concatenate([meg_data_norm["grad"], meg_data_norm["mag"]], axis=1) #(2874, 306, 601)
                    elif selected_channel_indices["grad"]:
                        raise NotImplementedError("Not yet implemented for grad channels aswell (need to adjust normalize_array() at the least.)")
                        logger.custom_debug("Using only grad data.")
                        combined_meg = meg_data_norm["grad"]
                    elif selected_channel_indices["mag"]:
                        logger.custom_debug("Using only mag data.")
                        combined_meg = meg_data_norm["mag"]

                    # Debugging: Count timepoints in meg metadata
                    num_meg_metadata_timepoints = 0
                    for trial_id in meg_metadata["sessions"][session_id_num]["trials"]:
                        for timepoint in meg_metadata["sessions"][session_id_num]["trials"][trial_id]["timepoints"]:
                            num_meg_metadata_timepoints += 1
                    #logger.custom_debug(f"[Session {session_id_num}]: Timepoints in meg metadata: {num_meg_metadata_timepoints}")
                    if num_meg_metadata_timepoints != combined_meg.shape[0]:
                        raise ValueError(f"Number of timepoints in meg metadata and in meg data loaded from h5 file are not identical. Metadata: {num_meg_metadata_timepoints}. Found: {combined_meg.shape[0]}")


                    # Debugging: Count timepoints in combined metadata
                    num_combined_metadata_timepoints = 0
                    for trial_id in combined_metadata["sessions"][session_id_num]["trials"]:
                        for timepoint in combined_metadata["sessions"][session_id_num]["trials"][trial_id]["timepoints"]:
                            num_combined_metadata_timepoints += 1
                    #logger.custom_debug(f"[Session {session_id_num}]: Timepoints in combined metadata: {num_combined_metadata_timepoints}")

                    ### Split meg data ###
                    trials_split_dict = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="trial_splits")  # Get train/test split based on trials (based on scenes)

                    # Iterate over train/test split by trials (not over meg_metadata as before)
                    # In this fashion, the first element in the crop and meg dataset of each split type will surely be the first element in the array of trials for that split
                    meg_split = {"train": [], "test": []}
                    for split in meg_split:
                        for trial_id in trials_split_dict[split]:
                            # Get timepoints from combined_metadata
                            for timepoint_id in combined_metadata["sessions"][session_id_num]["trials"][trial_id]["timepoints"]:
                                meg_index = combined_metadata["sessions"][session_id_num]["trials"][trial_id]["timepoints"][timepoint_id]["meg_index"]
                                meg_datapoint = combined_meg[meg_index]

                                meg_split[split].append(meg_datapoint)
    
                    # Convert meg data to numpy array
                    for split in meg_split:
                        meg_split[split] = np.array(meg_split[split])
                        # Debugging
                        if normalization == "mean_centered_ch_then_global_robust_scaling":
                            n_epochs_two_step_norm[split] += meg_split[split].shape[0]

                    logger.custom_debug(f"[Session {session_id_num}]: Storing (intermediate) meg array with train shape {meg_split['train'].shape}")

                    meg_timepoints_in_dataset = meg_split['train'].shape[0] + meg_split['test'].shape[0]

                    if meg_timepoints_in_dataset != num_combined_metadata_timepoints:
                        raise ValueError("Number of timepoints in meg dataset and in combined metadata are not identical.")

                    if session_id_num == "1":
                        for split in meg_split:
                            logger.custom_debug(f"[Session {session_id_num}]: {split} split shape: {meg_split[split].shape}")


                    # Clip out outliers based percentile (except for normals with global, here it will be done later)
                    if clip_outliers and "global" not in normalization:
                        cutoff_low, cutoff_high = np.percentile(np.concatenate((meg_split["train"], meg_split["test"])), [q_bottom, q_top], axis=None)
                        for split in meg_split:
                            meg_split[split] = np.clip(meg_split[split], a_min=cutoff_low, a_max=cutoff_high)

                    # Export meg dataset arrays to .npz
                    self.export_split_data_as_file(session_id=session_id_num, 
                                                type_of_content="meg_data",
                                                array_dict=meg_split,
                                                type_of_norm=normalization_stage)

        logger.custom_debug(f"meg_timepoints_in_dataset after per-session normalization: {n_epochs_two_step_norm}")
        logger.custom_debug(f"combined train+test: {n_epochs_two_step_norm['train'] + n_epochs_two_step_norm['test']}")

        
        for normalization in self.normalizations:
            if "global_robust_scaling" in normalization:
                self._apply_global_robust_scaling_across_all_sessions(current_norm=normalization, interpolate_outliers=interpolate_outliers, clip_outliers=clip_outliers, q_bottom=q_bottom, q_top=q_top)


    def _apply_global_robust_scaling_across_all_sessions(self, current_norm:str, q_bottom:float, q_top:float, interpolate_outliers:bool=False, clip_outliers:bool=True, glaser_region:str=None):
        """
        Applies robust scaling across all concatenated sessions, then stores them seperately again.
        """
        assert current_norm in ["mean_centered_ch_then_global_robust_scaling", "mean_centered_voxel_then_global_robust_scaling", "global_robust_scaling"], "Function currently limited to the three normalization methods."
        
        # Load data for all sessions with mean_centering_ch already applied
        meg_data_all_sessions = None
        metadata_by_session = self.recursive_defaultdict()

        if current_norm == "mean_centered_ch_then_global_robust_scaling":
            previous_norm_step = "mean_centered_ch"  
        elif current_norm == "mean_centered_voxel_then_global_robust_scaling":
            previous_norm_step = "mean_centered_vox"  
        else:
            previous_norm_step = "no_norm"

        glaser_region_folder = f"/{glaser_region}" if glaser_region is not None else ""
        
        for session_id in self.session_ids_num:
            # Get data after per-session normalization steps (may be no normalization if only global normalization is performed)
            if glaser_region is not None:
                storage_folder = f"data_files/{self.lock_event}/meg_data/source_space/{glaser_region}/{previous_norm_step}/subject_{self.subject_id}/session_{session_id}"  
                meg_data_session = {"train": None, "test": None}
                for split in meg_data_session:
                    storage_file = f"meg_data_{split}.npy"
                    storage_path = os.path.join(storage_folder, storage_file)

                    meg_data_session[split] = np.load(storage_path)
            else:
                meg_data_session = self.load_split_data_from_file(session_id_num=session_id, type_of_content="meg_data", type_of_norm=previous_norm_step)
            
            # Combine train and test
            # Store num of epochs for later concatenation
            metadata_by_session["session_id"][session_id]["n_train_epochs"] = np.shape(meg_data_session["train"])[0] 
            metadata_by_session["session_id"][session_id]["n_test_epochs"] = np.shape(meg_data_session["test"])[0] 

            meg_data_session = np.concatenate((meg_data_session["train"], meg_data_session["test"]))
            
            # TODO: Seperate sensor types if required

            # Concatenate over sessions
            meg_data_all_sessions = meg_data_session if meg_data_all_sessions is None else np.concatenate((meg_data_all_sessions, meg_data_session))

        # Apply robust scaling across complete dataset (all sessions)
        meg_data_normalized = self.normalize_array(meg_data_all_sessions, normalization="robust_scaling") # "robust_scaling, z_score"

        if clip_outliers: # 0.3 and 99.7 percentile is equal to 3 standard deviations
            # Get indices from z-scored robust scaled data
            logger.custom_debug(f"Clipping outliers.")
            cutoff_low, cutoff_high = np.percentile(meg_data_normalized, [q_bottom, q_top], axis=None)
            meg_data_normalized = np.clip(meg_data_normalized, a_min=cutoff_low, a_max=cutoff_high) # q0_3, q99_7

        logger.custom_debug(f"meg_data_normalized.shape: {meg_data_normalized.shape}")
        logger.custom_debug(f"meg_timepoints_in_dataset after final norm, before split into sessions: {meg_data_normalized.shape[0]}")

        # Seperate into sessions and train/split again
        meg_data_normalized_by_session = self.recursive_defaultdict()
        epoch_start_index = 0
        for session_id in self.session_ids_num:
            n_train_epochs = metadata_by_session["session_id"][session_id]["n_train_epochs"]
            n_test_epochs = metadata_by_session["session_id"][session_id]["n_test_epochs"]

            end_train_index = epoch_start_index + n_train_epochs
            end_test_index = end_train_index + n_test_epochs

            meg_data_session_train = meg_data_normalized[epoch_start_index:end_train_index,:,:]
            meg_data_session_test = meg_data_normalized[end_train_index:end_test_index,:,:]

            meg_data_normalized_by_session[session_id]["train"] = meg_data_session_train
            meg_data_normalized_by_session[session_id]["test"] = meg_data_session_test

            logger.custom_debug(f"[Session {session_id}]: meg_data_normalized['train'].shape: {meg_data_normalized_by_session[session_id]['train'].shape}")
            logger.custom_debug(f"[Session {session_id}]: meg_data_normalized['test'].shape: {meg_data_normalized_by_session[session_id]['test'].shape}")

            if interpolate_outliers:
                logger.custom_debug(f"\n \n Performing Interpolation for session {session_id}")
                logger.custom_debug(f"shapes before interpolation: Train: {meg_data_normalized_by_session[session_id]['train'].shape}, Test: {meg_data_normalized_by_session[session_id]['test'].shape}")
                meg_data_combined = np.concatenate((meg_data_normalized_by_session[session_id]["train"], meg_data_normalized_by_session[session_id]["test"]))
                n_outliers_in_session = 0
                for sensor in range(meg_data_combined.shape[1]):
                    for timepoint in range(meg_data_combined.shape[2]):
                        # Interpolate over all epochs for a given sensor, timepoint combination
                        meg_data_ch_t = meg_data_combined[:,sensor,timepoint] # n values where n = num epochs
                        indices = [index for index in range(len(meg_data_ch_t))]
                        meg_data_ch_t_non_outliers = {index: meg_data_ch_t[index] for index in indices if abs(meg_data_ch_t[index]) <= 3}
                        idx_to_be_interpolated = [idx for idx in indices if idx not in meg_data_ch_t_non_outliers.keys()]

                        # If outliers exist for this session, channel and timepoint
                        if idx_to_be_interpolated:
                            interpolated_values = np.interp(idx_to_be_interpolated, list(meg_data_ch_t_non_outliers.keys()), list(meg_data_ch_t_non_outliers.values()))
                            meg_data_ch_t_interpolated = meg_data_ch_t[idx_to_be_interpolated] = interpolated_values

                            meg_data_combined[idx_to_be_interpolated,sensor,timepoint] = meg_data_ch_t_interpolated
                            n_outliers_in_session += len(idx_to_be_interpolated)
                            
                # Build back into split # n_train_epochs
                meg_data_normalized_by_session[session_id]["train"] = meg_data_combined[:n_train_epochs,:,:]
                meg_data_normalized_by_session[session_id]["test"] = meg_data_combined[n_train_epochs:,:,:]

                logger.custom_debug(f"shapes after interpolation: Train: {meg_data_normalized_by_session[session_id]['train'].shape}, Test: {meg_data_normalized_by_session[session_id]['test'].shape}")
                logger.custom_debug(f"[session_id: {session_id}]n_outliers_in_session: {n_outliers_in_session}")


            # Export meg dataset arrays to .npz
            if glaser_region is not None:
                for split in ["train", "test"]:
                    save_folder = f"data_files/{self.lock_event}/meg_data/source_space/{glaser_region}/{current_norm}/subject_{self.subject_id}/session_{session_id}/{split}"  
                    os.makedirs(save_folder, exist_ok=True)
                    save_file = "meg_data.npy"
                    save_path = os.path.join(save_folder, save_file)

                    np.save(save_path, meg_data_normalized_by_session[session_id][split])
            else:
                self.export_split_data_as_file(session_id=session_id, 
                    type_of_content="meg_data",
                    array_dict=meg_data_normalized_by_session[session_id],
                    type_of_norm=current_norm)

            epoch_start_index = end_test_index
    
            # TODO: Combine grad and mag if both selected
        
        logger.custom_debug(f"end_test_index: {end_test_index}")


    def apply_pca_to_voxels(self, regions_of_interest:list, source_pca_type:str, whiten:bool) -> None:
        """
        Reduces dimensionality of estimated glaser region meg signal by applying PCA over voxels within regions.
        """
        whiten_folder = "/whiten" if whiten else ""
        for normalization in self.normalizations:
            for glaser_region in regions_of_interest:
                n_elements_by_session_by_split = self.recursive_defaultdict()  # Store number of elements to seperate combined meg data into sessions and splits again
                all_session_split_data_combined = []
                for session_id in self.session_ids_num:
                    # Load session and region specific data
                    meg_split = {"train": None, "test": None}
                    for split in meg_split:
                        storage_folder = f"data_files/{self.lock_event}/meg_data/source_space/{glaser_region}/{normalization}/subject_{self.subject_id}/session_{session_id}/{split}"  
                        storage_file = "meg_data.npy"
                        storage_path = os.path.join(storage_folder, storage_file)

                        meg_split[split] = np.load(storage_path)
                    all_session_split_data_combined.append((meg_split["train"], meg_split["test"]))

                    # Store number of elements
                    n_train_session = len(meg_split["train"])
                    n_test_session = len(meg_split["test"])
                    n_elements_by_session_by_split["session"][session_id]["train"] = n_train_session
                    n_elements_by_session_by_split["session"][session_id]["test"] = n_test_session

                # Combine all features into a single array to apply z-score to and fit PCA on
                all_session_split_data_combined = np.concatenate([np.concatenate((meg_train, meg_test)) for meg_train, meg_test in all_session_split_data_combined], axis=0)

                # Fit and apply PCA on combined sessions and splits (seperately on each timepoint)
                n_epochs, n_voxels, n_timepoints = all_session_split_data_combined.shape
                #print(f"n_epochs, n_voxels, n_timepoints: {n_epochs, n_voxels, n_timepoints}")
                if source_pca_type == "voxels":
                    pca_folder = "/voxels_pca_reduced"
                    n_pca_comps = int(n_voxels*0.05) if int(n_voxels*0.05) > 1 else 1  # keep 5% of components. V1: 4 components (out of 80 voxels) explain ~70% variance
                    all_session_split_data_combined_pca = np.zeros(shape=(n_epochs, n_pca_comps, n_timepoints))
                    for timepoint_idx in range(n_timepoints):
                        # Fit PCA
                        timepoint_data = all_session_split_data_combined[:,:,timepoint_idx]
                        pca = PCA(n_components=n_pca_comps, whiten=whiten)
                        pca.fit(timepoint_data)

                        # Investigate variance explained by n components
                        explained_var_per_component = pca.explained_variance_ratio_
                        explained_var = 0
                        for explained_var_component in explained_var_per_component:
                            explained_var += explained_var_component
                        logger.custom_info(f"\n Explained Variance {glaser_region}: {explained_var} \n")

                        # Apply pca
                        all_session_split_data_combined_pca[:,:,timepoint_idx] = pca.transform(timepoint_data)
                else:
                    pca_folder = "/voxels_and_timepoints_pca_reduced"
                    n_og_dims = n_voxels * n_timepoints
                    n_pca_comps = 4
                    meg_data_flattened = all_session_split_data_combined.reshape(n_epochs, -1)
                    pca = PCA(n_components=n_pca_comps, whiten=whiten)
                    pca.fit(meg_data_flattened)

                    explained_var_per_component = pca.explained_variance_ratio_
                    explained_var = 0
                    for explained_var_component in explained_var_per_component:
                        explained_var += explained_var_component
                    logger.custom_info(f'''\n Explained Variance {glaser_region} with {n_pca_comps} PCs: {explained_var} \n''')

                    all_session_split_data_combined_pca = pca.transform(meg_data_flattened)

                # Seperate all_session_split_data_combined_pca again and store results
                starting_meg_index = 0  # Updated with n elements belonging to each session to extract data belonging to each session
                for session_idx, session_id in enumerate(self.session_ids_num):
                    # Get splits for this session from combined meg data by indices
                    n_train_session = n_elements_by_session_by_split["session"][session_id]["train"]
                    n_test_session = n_elements_by_session_by_split["session"][session_id]["test"]
                    n_total_session = n_train_session + n_test_session

                    meg_data_train = all_session_split_data_combined_pca[starting_meg_index:starting_meg_index + n_train_session]
                    meg_data_test = all_session_split_data_combined_pca[starting_meg_index + n_train_session:starting_meg_index + n_train_session + n_test_session]
                    
                    session_meg_splits = {"train": meg_data_train, "test": meg_data_test}

                    for split in ["train", "test"]:
                        save_folder = f"data_files/{self.lock_event}/meg_data/source_space{pca_folder}{whiten_folder}/{glaser_region}/{normalization}/subject_{self.subject_id}/session_{session_id}/{split}"  
                        save_file = "meg_data.npy"
                        os.makedirs(save_folder, exist_ok=True)
                        save_path = os.path.join(save_folder, save_file)
                        np.save(save_path, session_meg_splits[split])

                    # update starting index for next session
                    starting_meg_index += n_total_session

            
    def create_source_meg_dataset(self, regions_of_interest:list, q_bottom:float, q_top:float, clip_outliers=True) -> None:
        """
        Creates one meg dataset per source region of interest (per session) (per normalization).
        Seperate from 'create_meg_dataset' for reasons of readability.
        ! assumes identical meg metadata for source data and sensor data !
        """
        if self.lock_event == "saccade":
            raise ValueError("Only fixation-centered source reconstructed meg data available atm.")

        # Read metadata and meg data
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")
        meg_metadata = self.read_dict_from_json(type_of_content="meg_metadata")

        meg_data_folder = f'/share/klab/datasets/avs/population_codes/as{self.subject_id}/source_space/beamformer/glasser/ori_None/hem_lh/filter_0.2_200/ica'

        for session_id_char in self.session_ids_char:
            session_id = self.map_session_letter_id_to_num(session_id_char)

            # get session data
            meg_file_name = f"as{self.subject_id}{session_id_char}_population_codes_{self.lock_event}_500hz_masked_False.h5"
            meg_data_path = os.path.join(meg_data_folder, meg_file_name)

            with h5py.File(meg_data_path, "r") as meg_file:
                # create seperate dataset for each glaser region of interest
                for glaser_region in regions_of_interest:
                    meg_data = np.array(meg_file[glaser_region]['onset'])  #  shape: (n_epochs, n_voxels, n_timepoints)

                    n_voxels = meg_data.shape[1]

                    # Filter considered meg data based on relevant timepoints
                    meg_data = meg_data[:,:,self.timepoint_min:self.timepoint_max+1]

                    for normalization in self.normalizations:
                        # Set normalization that is to be performed per session; some normalizations require additional global computations across all sessions
                        if normalization == "mean_centered_voxel_then_global_robust_scaling":
                            # Treat voxels as channels in normalization
                            normalization_stage = "mean_centered_vox"
                        elif normalization == "global_robust_scaling":
                            normalization_stage = "no_norm"
                        else:
                            normalization_stage = normalization

                        # Apply normalization step
                        meg_data_norm = self.normalize_array(meg_data, normalization=normalization_stage, session_id=session_id, n_channels=n_voxels)

                        # Split into train/test
                        trials_split_dict = self.load_split_data_from_file(session_id_num=session_id, type_of_content="trial_splits")  # Get train/test split based on trials (based on scenes)

                        # Iterate over train/test split by trials (not over meg_metadata as before)
                        # In this fashion, the first element in the crop and meg dataset of each split type will surely be the first element in the array of trials for that split
                        meg_split = {"train": [], "test": []}
                        for split in meg_split:
                            for trial_id in trials_split_dict[split]:
                                # Get timepoints from combined_metadata
                                for timepoint_id in combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]:
                                    meg_index = combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["meg_index"]
                                    meg_datapoint = meg_data_norm[meg_index]

                                    meg_split[split].append(meg_datapoint)

                            meg_split[split] = np.array(meg_split[split])

                        # Clip out outliers based percentile (except for two step norms, here it will be done later)
                        if clip_outliers and "global" not in normalization:
                            cutoff_low, cutoff_high = np.percentile(np.concatenate((meg_split["train"], meg_split["test"])), [q_bottom, q_top], axis=None)
                            for split in meg_split:
                                meg_split[split] = np.clip(meg_split[split], a_min=cutoff_low, a_max=cutoff_high)

                        # Export meg dataset arrays to .npz
                        save_folder = f"data_files/{self.lock_event}/meg_data/source_space/{glaser_region}/{normalization_stage}/subject_{self.subject_id}/session_{session_id}"  
                        os.makedirs(save_folder, exist_ok=True)

                        for split in meg_split:
                            save_file = f"meg_data_{split}.npy"
                            save_path = os.path.join(save_folder, save_file)

                            np.save(save_path, meg_split[split])
        
        for normalization in self.normalizations:
            if "global_robust_scaling" in normalization:
                for glaser_region in regions_of_interest:
                    self._apply_global_robust_scaling_across_all_sessions(current_norm=normalization, clip_outliers=clip_outliers, interpolate_outliers=False, glaser_region=glaser_region, q_bottom=q_bottom, q_top=q_top)


    def create_train_test_split(self, debugging=False):
        """
        Creates train/test split of trials based on scene_ids.
        """
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")

        # Prepare splits for all sessions: count scenes
        scene_ids = {}
        trial_ids = {}
        num_datapoints_dict = {}
        for session_id in combined_metadata["sessions"]:
            scene_ids[session_id] = {}
            trial_ids[session_id] = []
            num_datapoints_dict[session_id] = 0
        for session_id in combined_metadata["sessions"]:
            for trial_id in combined_metadata["sessions"][session_id]["trials"]:
                # Store trials in session for comparison with scenes
                trial_ids[session_id].append(trial_id)
                for timepoint in combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]:
                    # get sceneID of current timepoint
                    scene_id_current = combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint]["sceneID"]
                    # First occurance of the scene? Store its occurange with the corresponding trial
                    if scene_id_current not in scene_ids[session_id]:
                        scene_ids[session_id][scene_id_current] = {"trials": [trial_id]}
                    # SceneID previously occured in another trial, but this trial is not stored yet: store it under the scene_id aswell
                    elif trial_id not in scene_ids[session_id][scene_id_current]["trials"]:
                        scene_ids[session_id][scene_id_current]["trials"].append(trial_id)
                        logger.warning(f"Sceneid {scene_id_current} occurs in multiple trials: {scene_ids[session_id][scene_id_current]['trials']}")

                    # Count total datapoints/timepoints/fixations in combined metadata in session
                    num_datapoints_dict[session_id] += 1

        if debugging:
            total_num_datapoints = 0
            for session_id in num_datapoints_dict:
                total_num_datapoints += num_datapoints_dict[session_id]
            logger.custom_debug(f"Total number of datapoints: {total_num_datapoints}")

        # Create splits for each session
        for session_id in combined_metadata["sessions"]:
            # Debugging: Identical number of trials and scenes in this session?
            num_scenes = len(scene_ids[session_id])  # subject 2, session a: 300 
            num_trials = len(trial_ids[session_id])
            assert num_scenes == num_trials, f"Session {session_id}: Number of trials and number of scenes is not identical. Doubled scenes need to be considered"
            assert num_trials == len(combined_metadata["sessions"][session_id]["trials"].keys()), f"Registered number of trials not identical to number of trials in combined metadata for session {session_id}."

            # Choose split size (80/20)
            num_trials_train = int(num_trials*0.8)
            num_trials_test = num_trials - num_trials_train
            assert num_trials == (num_trials_train + num_trials_test), "Sanity check, lost trials to split"

            # Split based on scene ids, but trial information is sufficient to identify datapoint
            train_split_trials = []
            test_split_trials = []
            # Create split sceneID (by ordering the trials by scenes, trials that belong to the same scene will most likely be in the same split)
            index = 0        
            # Shuffle the scenes in the session, so that the test split does not always contain the latest scenes (I iterated by trials and timepoints above)
            session_scene_ids = list(scene_ids[session_id].keys())
            random.shuffle(session_scene_ids)
            for scene_id in session_scene_ids:
                for trial_id in scene_ids[session_id][scene_id]["trials"]:
                    # Each unique combination is one point in the dataset
                    if index < num_trials_train:
                        train_split_trials.append(trial_id)
                    else:
                        test_split_trials.append(trial_id)
                    index += 1

            # Export trial_split arrays to .npz
            split_dict = {"train": train_split_trials, "test": test_split_trials}

            self.export_split_data_as_file(session_id=session_id, 
                                        type_of_content="trial_splits",
                                        array_dict=split_dict)

            # TODO, maybe (most likely not): Make sure that scenes only double in the same split

            assert num_trials_train == len(train_split_trials), "Lost trials in train split"
            assert num_trials_test == len(test_split_trials), "Lost trials in train split"
            for split in split_dict:
                assert len(split_dict[split]) == len(set(split_dict[split])), f"Session {session_id}: Split {split} contains doubled trials."
                for trial_id in split_dict[split]:
                    other_split = "train" if split == "test" else "test"
                    assert trial_id not in split_dict[other_split], f"Session {session_id}: Trial {trial_id} occurs in both train and test split."

            logger.custom_debug(f"[Session {session_id}]: len train_split: {len(train_split_trials)}, len test_split: {len(test_split_trials)}, combined size: {len(train_split_trials) + len(test_split_trials)} \n")


    def investigate_session_cluster_distribution(self, debugging=False):
        """
        Creates train/test split of trials based on scene_ids and even distribution of semantic cluster.
        Each subject sees each scene only once over all sessions.
        """
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")

        # Collect all scenes shown in each session and their trial_id
        scene_trial_tuple_list_by_session_id = {"sessions": {session_id: [] for session_id in self.session_ids_num}}
        for session_id, session_dict in combined_metadata["sessions"].items():
            for trial_id, trial_dict in session_dict["trials"].items():
                # Get sceneID of trial based on first timepoint and store it in the list of the current session
                first_timepoint = list(trial_dict["timepoints"].keys())[0]
                scene_id = trial_dict["timepoints"][first_timepoint]["sceneID"]
                
                scene_trial_tuple_list_by_session_id["sessions"][session_id].append((scene_id, trial_id))

        # Collect semantic cluster distribution of scenes among sessions
        semantic_clusters_path = "/share/klab/datasets/avs/input/scene_sampling_MEG/scenes-per-sub-active-visual-semantics-MEG.csv"
        semantic_clusters_df = pd.read_csv(semantic_clusters_path, delimiter='|', on_bad_lines='warn')

        def extract_cluster_from_scene_id(scene_id:float):
            df_scene_id_idx = semantic_clusters_df.index[semantic_clusters_df['cocoID'] == int(scene_id)].tolist()[0]  # Takes one df index out of all occurances of the scene_id in the dataset, based on which corresponding cluster and other info can be extracted 
            cluster_id = semantic_clusters_df['cluster'][df_scene_id_idx]  # int between 0 and 59

            return cluster_id

        # Count occurances of scenes belonging to each cluster in each session
        
        cluster_of_interest_idx = [cluster_idx for cluster_idx in range(60)]
        n_scenes_per_cluster_sessions = []
        for session_id, session_list in scene_trial_tuple_list_by_session_id["sessions"].items():
            #print(f"Session {session_id}: {len(session_list)} scenes.")
            n_scenes_per_cluster = np.zeros(shape=(60))
            for scene_id, trial_id in session_list:
                cluster_id_trial = extract_cluster_from_scene_id(scene_id)
                n_scenes_per_cluster[cluster_id_trial] += 1
            n_scenes_per_cluster_sessions.append(n_scenes_per_cluster)
            print(f"Session {session_id}: \n {n_scenes_per_cluster[cluster_of_interest_idx]}")
        n_scenes_per_cluster_sessions = np.sum(np.array(n_scenes_per_cluster_sessions), axis=0)
        print(f"n_scenes_per_cluster all sessions added: \n {n_scenes_per_cluster_sessions[cluster_of_interest_idx]} \n")
        ### Clusters are evently distributed within subjects (64 to 68 scenes per cluster per subject), but unevenly within sessions! ###

    def create_pytorch_dataset(self, debugging=False):
        """
        Creates pytorch datasets from numpy image datasets.
        """
        
        for session_id in self.session_ids_num:
            # Load numpy datasets for session
            crop_ds = self.load_split_data_from_file(session_id_num=session_id, type_of_content="crop_data")

            # Convert arrays to PyTorch tensors and create Dataset
            train_tensors = torch.tensor(crop_ds['train'], dtype=torch.float32)
            test_tensors = torch.tensor(crop_ds['test'], dtype=torch.float32)
            tensor_dict = {"train": train_tensors, "test": test_tensors}

            if debugging:
                for split, tensor in tensor_dict.items():
                    logger.custom_debug(f"[Session {session_id}][Split {split}][Content TorchDataset]: Contains tensor of shape {tensor.shape}")
            
            torch_dataset = {}
            for split in tensor_dict:
                torch_dataset[split] = TensorDataset(tensor_dict[split])

            # Store datasets
            self.export_split_data_as_file(session_id=session_id, type_of_content="torch_dataset", array_dict=torch_dataset)

            # Store some tensors for debugging of pipeline
            if debugging and session_id in ["2", "5", "8"]:
                for split in tensor_dict:
                    for crop_split_index in [0, 10, 100, 1000]:
                        if len(tensor_dict[split]) >= crop_split_index:
                            save_folder = f"data_files/{self.lock_event}/debugging/crop_data/pytorch_dataset/session_{session_id}/{split}/crop_split_index_{crop_split_index}"
                            os.makedirs(save_folder, exist_ok=True)
                            save_path = os.path.join(save_folder, "crop_image_pytorch.pt")
                            torch.save(tensor_dict[split][crop_split_index], save_path)


    class TorchDatasetHelper(Dataset):
        """
        Inner class to create Pytorch dataset from numpy arrays (images)
        DEPRECATED FOR THE MOMENT, APPROACH UNINTUITIVE.
        """
        def __init__(self, numpy_array, transform=None):
            """
            Args:
                numpy_array (numpy.ndarray): A Numpy array of images (should be in CHW format if channels are present)
                transform (callable, optional): Optional transform to be applied on a sample.
            """
            self.numpy_array = numpy_array
            if transform is None:
                self.transform = transforms.Compose([transforms.ToTensor(),])  # Convert numpy arrays to torch tensors
            else:
                self.transform = transform

        def __len__(self):
            return len(self.numpy_array)

        def __getitem__(self, idx):
            image = self.transform(self.numpy_array[idx])

            return image



class ExtractionHelper(DatasetHelper):  # previously inherited only from BasicOperationsHelper
    def __init__(self, ann_model:str, module_name:str, batch_size:int, pca_components:int, **kwargs):
        super().__init__(**kwargs)

        self.ann_model = ann_model
        self.module_name = module_name  # Name of Layer to extract features from
        self.batch_size = batch_size
        self.pca_components = pca_components


    def extract_features_simulation_scene_dataset(self):
        """
        Extracts features from the scenes selected for the generation of simulated responses.
        """
        # Load simulation scene dataset
        simulation_scenes_folder = f"data_files/{self.lock_event}/simulation_scenes/scenes_numpy/subject_{self.subject_id}"  
        simulation_scenes_file = f"simulation_scenes_by_clusters.npy"
        simulation_scenes_path = os.path.join(simulation_scenes_folder, simulation_scenes_file)
        simulation_scenes_arr = np.load(simulation_scenes_path)  # shape: (60, 5, 224, 224, 3)

        # Load CNN model
        model_name = f'{self.ann_model}_ecoset'
        source = 'custom'
        device = 'cuda'
        extractor = get_extractor(
            model_name=model_name,
            source=source,
            device=device,
            pretrained=True
        )

        simulation_scenes_tensors = torch.tensor(simulation_scenes_arr, dtype=torch.float32)

        # Transpose dimensions to match (clusters, scenes_per_cluster, channels, height, width) instead of (clusters, scenes_per_cluster, height, width, channels) as before
        simulation_scenes_tensors = simulation_scenes_tensors.permute(0, 1, 4, 2, 3)
        simulation_scenes_tensors_shape = simulation_scenes_tensors.shape  # (60, 5, 3, 224, 224)

        # Extract features cluster-wise
        simulation_scenes_features = []
        for cluster_idx in range(simulation_scenes_tensors_shape[0]):
            # Create a DataLoader to handle batching
            cluster_input_tensors = DataLoader(simulation_scenes_tensors[cluster_idx], batch_size=self.batch_size, shuffle=False)

            cluster_features = extractor.extract_features(
                batches=cluster_input_tensors,
                module_name=self.module_name,
                flatten_acts=True  # flatten 2D feature maps from convolutional layer
            )

            simulation_scenes_features.append(cluster_features)
        simulation_scenes_features = np.array(simulation_scenes_features)  # shape: (60, 5, 9216)

        save_folder = f"data_files/{self.lock_event}/simulation_scenes/scenes_features/subject_{self.subject_id}"  
        save_file = f"simulation_scenes_features_by_clusters.npy"
        save_path = os.path.join(save_folder, save_file)
        os.makedirs(save_folder, exist_ok=True)
        np.save(save_path, simulation_scenes_features)


    def reduce_feature_dimensionality_simulation_scene_dataset(self, z_score_features_before_pca:bool):
        """
        Reduces dimensionality of features extracted from the scenes selected for the generation of simulated responses.
        """
        # Load features of simulated scene dataset
        simulation_features_folder = f"data_files/{self.lock_event}/simulation_scenes/scenes_features/subject_{self.subject_id}"  
        simulation_features_file = f"simulation_scenes_features_by_clusters.npy"
        simulation_features_path = os.path.join(simulation_features_folder, simulation_features_file)
        simulation_features = np.load(simulation_features_path)  # shape: (60, 5, 9216)

        n_clusters, n_scenes_per_cluster, n_cnn_dims = simulation_features.shape

        # Fit PCA on and apply PCA to all features of all clusters combined (reevaluate method!)
        simulation_features_clusters_combined = simulation_features.reshape(-1, simulation_features.shape[2])  # shape (60*clusters_per_image, ann_feature_dims)

        if z_score_features_before_pca:
            simulation_features_clusters_combined = self.normalize_array(data=simulation_features_clusters_combined, normalization="z_score")

        pca = PCA(n_components=self.pca_components)
        pca.fit(simulation_features_clusters_combined)
        simulation_features_clusters_combined_reduced_dim = pca.transform(simulation_features_clusters_combined)

        simulation_features_reduced_dim = simulation_features_clusters_combined_reduced_dim.reshape(n_clusters, n_scenes_per_cluster, self.pca_components)  # shape: (n_clusters, n_scenes_per_cluster, self.pca_components)

        print(f"simulation_features_reduced_dim.shape: {simulation_features_reduced_dim.shape}")

        # Store features with reduced dim
        save_folder = f"data_files/{self.lock_event}/simulation_scenes/scenes_features_pca/subject_{self.subject_id}"  
        save_file = f"simulation_scenes_features_pca_by_clusters.npy"
        save_path = os.path.join(save_folder, save_file)
        os.makedirs(save_folder, exist_ok=True)
        np.save(save_path, simulation_features_reduced_dim) 


    def extract_features_from_all_crops(self):
        """
        Extracts features from crop datasets over all sessions for a subject.
        """
        # Load model
        model_name = f'{self.ann_model}_ecoset'
        source = 'custom'
        device = 'cuda'

        extractor = get_extractor(
            model_name=model_name,
            source=source,
            device=device,
            pretrained=True
        )
        #print(F"extractor.show_model(): {extractor.show_model()}")
        transform_resize_normalize = extractor.get_transformations()

        logger.custom_info(f"transform_resize_normalize: {transform_resize_normalize}")

        logger.custom_info(f"model_name: {model_name}")
        logger.custom_info(f"self.module_name: {self.module_name} \n")

        for session_id in self.session_ids_num:
            # Load numpy datasets for session
            crop_ds = self.load_split_data_from_file(session_id_num=session_id, type_of_content="crop_data") # shape (n_images, crop_size,crop_size,3)

            # Perform preprocessing
            # Conver to PIL IMG 
            train_images = [Image.fromarray(img_arr.astype('uint8')) for img_arr in crop_ds["train"]]
            test_images = [Image.fromarray(img_arr.astype('uint8')) for img_arr in crop_ds["test"]]
            # Perform dimension permutation, rescaling and normalization
            train_images_resized = [transform_resize_normalize(img_arr) for img_arr in train_images]
            test_images_resized = [transform_resize_normalize(img_arr) for img_arr in test_images]

            train_tensors = torch.stack(train_images_resized).type(torch.float32)
            test_tensors = torch.stack(test_images_resized).type(torch.float32)
            
            # Debugging: Store example resized images
            if session_id == "1":
                # Unnormalize before visualizing, based on norm performed in ecoset pretrained alexnet from thingsvision
                invert_norm = transforms.Normalize(
                        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                    )
                for sample_index in range(100, 110):
                    img_tensor = invert_norm(test_tensors[sample_index])
                    save_folder = f"data_files/{self.lock_event}/visualizations/test_image_pre_extract/test_image_{sample_index}.png"
                    save_image(img_tensor, save_folder)

            logger.custom_debug(f"Shape of train tensors: {train_tensors.shape}")
            logger.custom_debug(f"Shape of test tensors: {test_tensors.shape}")

            # Create a DataLoader to handle batching
            model_input = {}
            model_input["train"] =  DataLoader(train_tensors, batch_size=self.batch_size, shuffle=False)
            model_input["test"] = DataLoader(test_tensors, batch_size=self.batch_size, shuffle=False)
    
            features_split = {}
            for split in ["train", "test"]:
                # Extract features
                features_split[split] = extractor.extract_features(
                    batches=model_input[split],
                    module_name=self.module_name,
                    flatten_acts=True  # flatten 2D feature maps from convolutional layer
                )

                # Debugging
                logger.custom_info(f"Session {session_id}: {split}_features.shape: {features_split[split].shape}")

            # Export numpy array to .npz
            self.export_split_data_as_file(session_id=session_id, type_of_content="ann_features", array_dict=features_split, ann_model=self.ann_model, module=self.module_name)

    def reduce_feature_dimensionality_all_crops(self, z_score_features_before_pca:bool = True, all_sessions_combined:bool = False):
        """
        Reduces dimensionality of extracted features using PCA. This seems to be necessary to avoid overfit in the ridge Regression.
        """
        if not all_sessions_combined:
            # Collect features of all sessions and splits 
            n_elements_by_session_by_split = self.recursive_defaultdict()  # Store number of elements to seperate combined features into sessions and splits again
            all_session_split_features_combined = []  # collect (train, test) tuples for all sessions  
            for session_id in self.session_ids_num:
                ann_features = self.load_split_data_from_file(session_id_num=session_id, type_of_content="ann_features", ann_model=self.ann_model, module=self.module_name)
                all_session_split_features_combined.append((ann_features["train"], ann_features["test"]))

                # Store number of elements
                n_train_session = len(ann_features["train"])
                n_test_session = len(ann_features["test"])
                n_elements_by_session_by_split["session"][session_id]["train"] = n_train_session
                n_elements_by_session_by_split["session"][session_id]["test"] = n_test_session
                
            # Combine all features into a single array to apply z-score to and fit PCA on
            all_session_split_features_combined = np.concatenate([np.concatenate((features_train, features_test)) for features_train, features_test in all_session_split_features_combined], axis=0)

            print(f"all_session_split_features_combined.shape: {all_session_split_features_combined.shape}")
            print(f"n PCA components: {self.pca_components}")

            # Apply z-scoring and fit PCA on combined sessions and splits
            if z_score_features_before_pca:
                raise ValueError("do NOT z-score features for the moment.")
                all_session_split_features_combined = self.normalize_array(data=all_session_split_features_combined, normalization="z_score")
            pca = PCA(n_components=self.pca_components)
            pca.fit(all_session_split_features_combined)

            explained_var_per_component = pca.explained_variance_ratio_
            explained_var = 0
            for explained_var_component in explained_var_per_component:
                explained_var += explained_var_component
            logger.custom_info(f"\n Explained Variance: {explained_var} \n")

            # Apply PCA to each session and each split and store results, seperating all_session_split_features_combined again (necessary due to z-scoring)
            starting_feature_index = 0  # Updated with n elements belonging to each session to extract features belonging to each session
            for session_idx, session_id in enumerate(self.session_ids_num):
                # Get splits for this session from combined features by indices
                n_train_session = n_elements_by_session_by_split["session"][session_id]["train"]
                n_test_session = n_elements_by_session_by_split["session"][session_id]["test"]
                n_total_session = n_train_session + n_test_session

                ann_features_train = all_session_split_features_combined[starting_feature_index:starting_feature_index + n_train_session]
                ann_features_test = all_session_split_features_combined[starting_feature_index + n_train_session:starting_feature_index + n_train_session + n_test_session]
                
                session_features_splits = {"train": ann_features_train, "test": ann_features_test}

                for split in ["train", "test"]:
                    session_features_splits[split] = pca.transform(session_features_splits[split])

                    save_folder = f"data_files/{self.lock_event}/ann_features_pca/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/session_{session_id}/{split}"
                    save_file = "ann_features_pca.npy"
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, save_file)
                    np.save(save_path, session_features_splits[split])

                # update starting index for next session
                starting_feature_index += n_total_session
        else:
            # Concat features over all sessions, only then apply pca

            def apply_pca_to_features(ann_features):
                """
                Fits pca on train and test features  combined. Use fix amount of components to allow cross-session predictions
                """
                ann_features_combined = np.concatenate((ann_features["train"], ann_features["test"]))
                if z_score_features_before_pca:
                    ann_features_combined = self.normalize_array(data=ann_features_combined, normalization="z_score")
                    n_train = len(ann_features["train"])
                    ann_features["train"] = ann_features_combined[:n_train,:]
                    ann_features["test"] = ann_features_combined[n_train:,:]

                pca = PCA(n_components=self.pca_components)
                pca.fit(ann_features_combined)
                explained_var_per_component = pca.explained_variance_ratio_
                explained_var = 0
                for explained_var_component in explained_var_per_component:
                    explained_var += explained_var_component

                # Transform splits
                for split in ann_features:
                    ann_features[split] = pca.transform(ann_features[split])

                return ann_features, explained_var

            pca_features = {"train": None, "test": None}
            for session_id in self.session_ids_num:
                # Get ANN features for session
                ann_features = self.load_split_data_from_file(session_id_num=session_id, type_of_content="ann_features", ann_model=self.ann_model, module=self.module_name)
                for split in pca_features:
                    if pca_features[split] is None:
                        pca_features[split] = ann_features[split]
                    else:
                        pca_features[split] = np.concatenate((pca_features[split], ann_features[split]))
            ann_features_pca, explained_var = apply_pca_to_features(pca_features)
            logger.custom_debug(f"Explained Variance: {explained_var}")

            for split in ann_features_pca:
                logger.custom_debug(f"Session {session_id}: {split}_features.shape: {ann_features_pca[split].shape}")

            self.export_split_data_as_file(session_id=None, type_of_content="ann_features_pca_all_sessions_combined", array_dict=ann_features_pca, ann_model=self.ann_model, module=self.module_name)



class GLMHelper(ExtractionHelper):
    def __init__(self, fractional_grid:list, alphas:list, pca_features:bool, fractional_ridge:bool = True, **kwargs):
        super().__init__(**kwargs)

        self.fractional_ridge = fractional_ridge
        self.fractional_grid = fractional_grid
        self.alphas = alphas
        self.ann_features_type = "ann_features_pca" if pca_features else "ann_features"


    def train_mapping(self, all_sessions_combined:bool=False, shuffle_train_labels:bool=False, downscale_features:bool=False, regions_of_interest:list=None, source_pca_type:str=None, whiten:bool=False):
        """
        Trains a mapping from ANN features to MEG data over all sessions.
        """
        if regions_of_interest is not None:
            assert not all_sessions_combined, "[train_mapping]: Invalid argument combination."
        whiten_folder = "/whiten" if whiten else ""
        if source_pca_type == 'voxels':
            pca_folder = f"/voxels_pca_reduced{whiten_folder}"
            timepoints_pca = False
        elif source_pca_type == 'voxels_and_timepoints':
            pca_folder = f"/voxels_and_timepoints_pca_reduced{whiten_folder}"
            timepoints_pca = True
        else:
            pca_folder = ""
            timepoints_pca = False

        def train_model(X_train:np.ndarray, Y_train: np.ndarray, normalization:str, all_sessions_combined:bool, session_id_num:str=None, glaser_region:str=None, pca_folder:str="", timepoints_pca:bool=False):
            # Initialize Helper class
            ridge_model = GLMHelper.MultiDimensionalRegression(self, timepoints_pca=timepoints_pca) 

            if downscale_features:
                X_train = self.normalize_array(data=X_train, normalization="range_-1_to_1")

            # Fit model on train data
            ridge_model.fit(X_train, Y_train)

            # Get alphas selected in RidgeCV
            selected_alphas = ridge_model.selected_alphas

            all_session_folder = f"/all_sessions_combined" if all_sessions_combined else ""
            session_addition = f"/session_{session_id_num}" if not all_sessions_combined else ""
            source_folder = f"/source_space{pca_folder}/{glaser_region}" if glaser_region is not None else ""

            # Store trained models as pickle
            save_folder = f"data_files/{self.lock_event}/GLM_models{source_folder}/{self.ann_model}/{self.module_name}/subject_{self.subject_id}{all_session_folder}/norm_{normalization}{session_addition}"  
            save_file = "GLM_models.pkl"
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, save_file)

            with open(save_path, 'wb') as file:
                pickle.dump(ridge_model.models, file)
            
            return selected_alphas

        if not all_sessions_combined:
            for normalization in self.normalizations:
                logger.custom_info(f"Training mapping for normalization {normalization}")
                session_alphas = {}
                if regions_of_interest is None:
                    for session_id_num in self.session_ids_num:
                        # Get ANN features and meg data for session
                        ann_features = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content=self.ann_features_type, ann_model=self.ann_model, module=self.module_name)
                        meg_data = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="meg_data", type_of_norm=normalization)

                        X_train, Y_train = ann_features['train'], meg_data['train']

                        if shuffle_train_labels:
                            np.random.shuffle(Y_train)

                        selected_alphas = train_model(X_train=X_train, Y_train=Y_train, normalization=normalization, all_sessions_combined=all_sessions_combined, session_id_num=session_id_num)
                        session_alphas[session_id_num] = selected_alphas
                    #self.save_dict_as_json(type_of_content="selected_alphas_by_session", dict_to_store=session_alphas, type_of_norm=normalization, predict_train_data=predict_train_data)
                else:
                    # seperately for each source/glaser region
                    for session_id in self.session_ids_num:
                        ann_features = self.load_split_data_from_file(session_id_num=session_id, type_of_content=self.ann_features_type, ann_model=self.ann_model, module=self.module_name)
                        X_train = ann_features['train']
                        for glaser_region in regions_of_interest:
                            # load train meg data
                            storage_folder = f"data_files/{self.lock_event}/meg_data/source_space{pca_folder}/{glaser_region}/{normalization}/subject_{self.subject_id}/session_{session_id}/train"  
                            storage_file = "meg_data.npy"
                            storage_path = os.path.join(storage_folder, storage_file)

                            Y_train = np.load(storage_path)

                            if shuffle_train_labels:
                                np.random.shuffle(Y_train)

                            selected_alphas = train_model(X_train=X_train, Y_train=Y_train, normalization=normalization, all_sessions_combined=all_sessions_combined, session_id_num=session_id, glaser_region=glaser_region, pca_folder=pca_folder, timepoints_pca=timepoints_pca)
        # all sessions combined
        else:
            for normalization in self.normalizations:
                meg_data_train_combined = None
                # Collect ANN features and MEG data over sessions
                for session_id_num in self.session_ids_num:
                    meg_data_train = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="meg_data", type_of_norm=normalization)['train']

                    if session_id_num == "1":
                        meg_data_train_combined = meg_data_train
                    else:
                        meg_data_train_combined = np.concatenate([meg_data_train_combined, meg_data_train], axis=0)

                ann_features_train_combined = self.load_split_data_from_file(session_id_num=None, type_of_content=self.ann_features_type+"_all_sessions_combined" , ann_model=self.ann_model, module=self.module_name)['train']

                X_train, Y_train = ann_features_train_combined, meg_data_train_combined

                assert X_train.shape[0] == Y_train.shape[0], "Different number of samples for features and meg data."

                if shuffle_train_labels:
                    np.random.shuffle(Y_train)

                logger.custom_debug(f"Train_mapping: X_train.shape: {X_train.shape}")
                logger.custom_debug(f"Train_mapping: Y_train.shape: {Y_train.shape}")

                 # Debugging
                #logger.custom_debug(f"X_train: {X_train}")
                #logger.custom_debug(f"Y_train: {Y_train}")
                logger.custom_debug(f"max X_train: {np.max(X_train)}, min X_train: {np.min(X_train)}")
                logger.custom_debug(f"max Y_train: {np.max(Y_train)}, min Y_train: {np.min(Y_train)}")
                logger.custom_debug(f"mean over epochs (each pca component) X_train: {np.mean(X_train, axis=(0))}")
                logger.custom_debug(f"mean over epochs and timepoints (each sensor) Y_train: {np.mean(Y_train, axis=(0,2))}")
                logger.custom_debug(f"std over epochs (each pca component) X_train: {np.std(X_train, axis=(0))}")
                logger.custom_debug(f"std over epochs and timepoints (each sensor) Y_train: {np.std(Y_train, axis=(0,2))}")

                selected_alphas = train_model(X_train=X_train, Y_train=Y_train, normalization=normalization, all_sessions_combined=all_sessions_combined)
                # For continuity with session alphas store combined alphas as dict aswell
                #selected_alphas_dict = {"all_sessions": selected_alphas}
                #self.save_dict_as_json(type_of_content="selected_alphas_all_sessions_combined", dict_to_store=selected_alphas_dict, type_of_norm=normalization, predict_train_data=predict_train_data)


        
    def predict_from_mapping_all_sessions(self, fit_measure_storage_distinction:str="session_level", predict_train_data:bool=False, all_sessions_combined:bool=False, shuffle_test_labels:bool=False, downscale_features:bool=False):
        """
        Based on the trained mapping for each session, predicts MEG data over all sessions from their respective test features.
        If predict_train_data is True, predicts the train data of each session as a sanity check of the complete pipeline. Expect strong overfit.
        """
        assert fit_measure_storage_distinction in ["session_level", "timepoint_level", "timepoint_sensor_level"], "[predict_from_mapping_all_sessions] Invalid argument for parameter fit_measure_storage_distinction"

        if not all_sessions_combined:
            for normalization in self.normalizations:
                logger.custom_info(f"Predicting from mapping for normalization {normalization}")
                variance_explained_dict = self.recursive_defaultdict()
                correlation_dict = self.recursive_defaultdict()
                predicted_responses_dict = self.recursive_defaultdict()
                mse_session_losses = {"session_mapping": {}}
                for session_id_model in self.session_ids_num:
                    mse_session_losses["session_mapping"][session_id_model] = {"session_pred": {}}
                    # Get trained ridge regression model for this session
                    # Load ridge model
                    storage_folder = f"data_files/{self.lock_event}/GLM_models/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/session_{session_id_model}"  
                    storage_file = "GLM_models.pkl"
                    storage_path = os.path.join(storage_folder, storage_file)
                    with open(storage_path, 'rb') as file:
                        ridge_models = pickle.load(file)

                    # Initialize MultiDim GLM class with stored models
                    ridge_model = GLMHelper.MultiDimensionalRegression(self, models=ridge_models)

                    # Generate predictions for test features over all sessions and evaluate them 
                    for session_id_pred in self.session_ids_num:
                        # Get ANN features and MEG data for session where predictions are to be evaluated
                        ann_features = self.load_split_data_from_file(session_id_num=session_id_pred, type_of_content=self.ann_features_type, ann_model=self.ann_model, module=self.module_name)
                        meg_data = self.load_split_data_from_file(session_id_num=session_id_pred, type_of_content="meg_data", type_of_norm=normalization)
                        
                        if predict_train_data:
                            X_test, Y_test = ann_features['train'], meg_data['train']
                        else:
                            X_test, Y_test = ann_features['test'], meg_data['test']

                        #logger.custom_info(f"predict_from_mapping_all_sessions: X_test.shape: {X_test.shape}")
                        #logger.custom_info(f"predict_from_mapping_all_sessions: Y_test.shape: {Y_test.shape}")

                        if shuffle_test_labels:
                            np.random.shuffle(Y_test)

                        # Generate and store predictions
                        predictions = ridge_model.predict(X_test, downscale_features=downscale_features)
                        predicted_responses_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred] = predictions

                        if fit_measure_storage_distinction == "timepoint_level":
                            # Store fit measures seperately for each timepoint/model
                            n_timepoints = predictions.shape[2]
                            n_sensors = predictions.shape[1]
                            for t in range(n_timepoints):
                                var_explained_timepoint_sum = 0
                                r_pearson_timepoint_sum = 0  
                                for s in range(n_sensors):
                                    var_explained_timepoint_sum += r2_score(Y_test[:,s,t], predictions[:,s,t])
                                    r_pearson_timepoint_sensor, _ = pearsonr(Y_test[:,s,t], predictions[:,s,t])
                                    r_pearson_timepoint_sum += r_pearson_timepoint_sensor

                                var_explained_timepoint = var_explained_timepoint_sum / n_sensors
                                r_pearson_timepoint = r_pearson_timepoint_sum / n_sensors
                                    
                                # Save fit measures
                                variance_explained_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred]["timepoint"][str(t)] = var_explained_timepoint
                                correlation_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred]["timepoint"][str(t)] = r_pearson_timepoint
                        elif fit_measure_storage_distinction == "timepoint_sensor_level":
                            # Calculate fit measure seperately for each sensor and timepoint
                            # prediction shape example: (502, 5, 101) (epochs, sensors, timepoints)
                            n_sensors = predictions.shape[1]
                            n_timepoints = predictions.shape[2]
                            for sensor_idx in range(n_sensors):
                                for timepoint_idx in range(n_timepoints):
                                    var_explained_sensor_timepoint = r2_score(Y_test[:,sensor_idx,timepoint_idx], predictions[:,sensor_idx,timepoint_idx])  # .reshape(-1)
                                
                                    # Save fit measure for selected sensor and timepoint
                                    variance_explained_dict["sensor"][str(sensor_idx)]["session_mapping"][session_id_model]["session_pred"][session_id_pred]["timepoint"][str(timepoint_idx)] = var_explained_sensor_timepoint
                        else:
                            raise ValueError(f"fit_measure_storage_distinction {fit_measure_storage_distinction} deprecated. Flattening biases variance explained.")

                            mse = mean_squared_error(Y_test.reshape(-1), predictions.reshape(-1))
                            var_explained = r2_score(Y_test.reshape(-1), predictions.reshape(-1))  
                            r_pearson, _ = pearsonr(Y_test.reshape(-1), predictions.reshape(-1))

                            # Save loss and variance explained
                            mse_session_losses["session_mapping"][session_id_model]["session_pred"][session_id_pred] = mse
                            correlation_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred] = r_pearson
                            variance_explained_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred] = var_explained

                # Store predictions dict
                storage_folder = f"data_files/{self.lock_event}/predicted_meg_responses/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                os.makedirs(storage_folder, exist_ok=True)
                storage_file = f"predicted_responses_dict.pkl"
                storage_path = os.path.join(storage_folder, storage_file)

                with open(storage_path, 'wb') as file:
                    pickle.dump(predicted_responses_dict, file)

                # Store loss dict
                if fit_measure_storage_distinction == "session_level":
                    self.save_dict_as_json(type_of_content="mse_losses", dict_to_store=mse_session_losses, type_of_norm=normalization)
                    self.save_dict_as_json(type_of_content="var_explained", dict_to_store=variance_explained_dict, type_of_norm=normalization, predict_train_data=predict_train_data)
                else:
                    if fit_measure_storage_distinction == "timepoint_level":
                        storage_dicts_by_folders = {"var_explained_timepoints": variance_explained_dict, "pearson_r_timepoints": correlation_dict}
                    elif fit_measure_storage_distinction == "timepoint_sensor_level":
                        storage_dicts_by_folders = {"var_explained_sensors_timepoints": variance_explained_dict}
                    else:
                        raise ValueError("Invalid value for fit_measure_storage_distinction.")

                    for main_folder, fit_measure_dict in storage_dicts_by_folders.items():
                        storage_folder = f"data_files/{self.lock_event}/{main_folder}/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                        os.makedirs(storage_folder, exist_ok=True)
                        json_storage_file = f"{main_folder}_dict.json"
                        json_storage_path = os.path.join(storage_folder, json_storage_file)

                        with open(json_storage_path, 'w') as file:
                            logger.custom_debug(f"Storing timepoint dict to {json_storage_path}")
                            # Serialize and save the dictionary to the file
                            json.dump(fit_measure_dict, file, indent=4)

                # Debugging
                if fit_measure_storage_distinction == "session_level":
                    for session_id in variance_explained_dict["session_mapping"]:
                        session_explained_var = variance_explained_dict['session_mapping'][session_id]['session_pred'][session_id]
                        logger.custom_info(f"[Session {session_id}]: Variance_explained_dict: {session_explained_var}")
                else:
                    first_timepoint_value = variance_explained_dict["session_mapping"]["1"]["session_pred"]["1"]["timepoint"]["0"]
                    logger.custom_debug(f"first_timepoint_value: {first_timepoint_value}")
    
        else:
            for normalization in self.normalizations:
                logger.custom_info(f"Predicting from mapping for normalization {normalization}")
                mse_dict = {}
                var_explained_dict = {}
                correlation_dict = {}
                # Get trained ridge regression models 
                storage_folder = f"data_files/{self.lock_event}/GLM_models/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/all_sessions_combined/norm_{normalization}"  
                storage_file = "GLM_models.pkl"
                storage_path = os.path.join(storage_folder, storage_file)
                with open(storage_path, 'rb') as file:
                    ridge_models = pickle.load(file)

                # Initialize MultiDim GLM class with stored models
                ridge_model = GLMHelper.MultiDimensionalRegression(self, models=ridge_models)

                # Generate predictions for test features evaluate them (or for train features to evaluate overfit)
                # Collect ANN features and MEG data over sessions
                meg_data_pred_combined = None
                pred_type = "train" if predict_train_data else "test"
                for session_id_num in self.session_ids_num:
                    meg_data_pred = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="meg_data", type_of_norm=normalization)[pred_type]

                    if session_id_num == "1":
                        meg_data_pred_combined = meg_data_pred
                    else:
                        meg_data_pred_combined = np.concatenate([meg_data_pred_combined, meg_data_pred], axis=0)

                ann_features_pred_combined = self.load_split_data_from_file(session_id_num=None, type_of_content=self.ann_features_type+"_all_sessions_combined" , ann_model=self.ann_model, module=self.module_name)[pred_type]

                X_test, Y_test = ann_features_pred_combined, meg_data_pred_combined

                logger.custom_debug(f"predict_from_mapping_all_sessions: X_test.shape: {X_test.shape}")
                logger.custom_debug(f"predict_from_mapping_all_sessions: Y_test.shape: {Y_test.shape}")

                if shuffle_test_labels:
                    np.random.shuffle(Y_test)

                # Generate predictions
                predictions = ridge_model.predict(X_test, downscale_features=downscale_features)

                # Calculate the mean squared error across all flattened features and timepoints
                mse = mean_squared_error(Y_test.reshape(-1), predictions.reshape(-1))
                # Calculate variance explained 
                var_explained = r2_score(Y_test.reshape(-1), predictions.reshape(-1))

                r_pearson, _ = pearsonr(Y_test.reshape(-1), predictions.reshape(-1))

                # Save results in dict
                mse_dict = {"mse_losses": mse}
                var_explained_dict = {"var_explained": var_explained}
                correlation_dict = {"correlation": r_pearson}

                for fit_measure in ["var_explained", "mse_losses", "correlation"]:
                    storage_folder = f"data_files/{self.lock_event}/{fit_measure}/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/all_sessions_combined/norm_{normalization}/predict_train_data_{predict_train_data}"
                    os.makedirs(storage_folder, exist_ok=True)
                    json_storage_file = f"{fit_measure}_all_sessions_combined_dict.json"
                    json_storage_path = os.path.join(storage_folder, json_storage_file)

                    dict_to_store = var_explained_dict if fit_measure == "var_explained" else mse_dict if fit_measure == "mse_losses" else correlation_dict
                    with open(json_storage_path, 'w') as file:
                        logger.custom_debug(f"Storing dict {fit_measure} to {json_storage_path}")
                        # Serialize and save the dictionary to the file
                        json.dump(dict_to_store, file, indent=4)


    def predict_from_mapping_source_all_sessions(self, regions_of_interest:list, source_pca_type:str, store_result_by_pc:bool, whiten:bool, predict_train_data:bool=False, shuffle_test_labels:bool=False, downscale_features:bool=False):
        """
        For each source region: based on the trained mapping for each session, predicts MEG data over all sessions from their respective test features.
        For readability reasons seperated from predict_from_mapping_all_sessions.
        If predict_train_data is True, predicts the train data of each session as a sanity check of the complete pipeline. Expect strong overfit.
        """
        whiten_folder = "/whiten" if whiten else ""
        if source_pca_type == 'voxels':
            pca_folder = f"/voxels_pca_reduced{whiten_folder}"
            timepoints_pca = False
        elif source_pca_type == 'voxels_and_timepoints':
            pca_folder = f"/voxels_and_timepoints_pca_reduced{whiten_folder}"
            timepoints_pca = True
        else:
            pca_folder = ""
            timepoints_pca = False

        for glaser_region in regions_of_interest:
            for normalization in self.normalizations:
                variance_explained_dict = self.recursive_defaultdict()
                correlation_dict = self.recursive_defaultdict()
                predicted_responses_dict = self.recursive_defaultdict()
                for session_id_model in self.session_ids_num:
                    # Get trained ridge regression model for this session
                    # Load ridge model
                    storage_folder = f"data_files/{self.lock_event}/GLM_models/source_space{pca_folder}/{glaser_region}/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/session_{session_id_model}"  
                    storage_file = "GLM_models.pkl"
                    storage_path = os.path.join(storage_folder, storage_file)
                    with open(storage_path, 'rb') as file:
                        ridge_models = pickle.load(file)

                    # Initialize MultiDim GLM class with stored models
                    ridge_model = GLMHelper.MultiDimensionalRegression(self, models=ridge_models, timepoints_pca=timepoints_pca)

                    # Generate predictions for test features over all sessions and evaluate them 
                    for session_id_pred in self.session_ids_num:
                        # Get ANN features and MEG data for session where predictions are to be evaluated
                        ann_features = self.load_split_data_from_file(session_id_num=session_id_pred, type_of_content=self.ann_features_type, ann_model=self.ann_model, module=self.module_name)
                        
                        meg_data = {"train": None, "test": None}
                        for split in meg_data:
                            storage_folder = f"data_files/{self.lock_event}/meg_data/source_space{pca_folder}/{glaser_region}/{normalization}/subject_{self.subject_id}/session_{session_id_pred}/{split}"  
                            storage_file = "meg_data.npy"
                            storage_path = os.path.join(storage_folder, storage_file)

                            meg_data[split] = np.load(storage_path)

                        if predict_train_data:
                            X_test, Y_test = ann_features['train'], meg_data['train']
                        else:
                            X_test, Y_test = ann_features['test'], meg_data['test']

                        if shuffle_test_labels:
                            np.random.shuffle(Y_test)

                        # Generate and store predictions
                        predictions = ridge_model.predict(X_test, downscale_features=downscale_features)
                        predicted_responses_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred] = predictions

                        # Store fit measures seperately for each timepoint/model
                        if not timepoints_pca:  # second dim are voxels or voxel pcs, third time are timepoints
                            _, n_voxels, n_timepoints = predictions.shape
                            if not store_result_by_pc:
                                for t in range(n_timepoints):
                                    var_explained_timepoint_sum = 0
                                    r_pearson_timepoint_sum = 0  
                                    for v in range(n_voxels):
                                        var_explained_timepoint_sum += r2_score(Y_test[:,v,t], predictions[:,v,t])
                                        r_pearson_timepoint_sensor, _ = pearsonr(Y_test[:,v,t], predictions[:,v,t])
                                        r_pearson_timepoint_sum += r_pearson_timepoint_sensor

                                    var_explained_timepoint = var_explained_timepoint_sum / n_voxels
                                    r_pearson_timepoint = r_pearson_timepoint_sum / n_voxels
                                        
                                    # Save fit measures
                                    variance_explained_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred]["timepoint"][str(t)] = var_explained_timepoint
                                    correlation_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred]["timepoint"][str(t)] = r_pearson_timepoint
                            else:
                                for t in range(n_timepoints):
                                    for pc in range(n_voxels):
                                        var_explained_timepoint_pc = r2_score(Y_test[:,pc,t], predictions[:,pc,t])
                                        r_pearson_timepoint_pc = pearsonr(Y_test[:,pc,t], predictions[:,pc,t])[0]

                                        # Save fit measures
                                        variance_explained_dict["pcs"][pc]["session_mapping"][session_id_model]["session_pred"][session_id_pred]["timepoint"][str(t)] = var_explained_timepoint_pc
                                        correlation_dict["pcs"][pc]["session_mapping"][session_id_model]["session_pred"][session_id_pred]["timepoint"][str(t)] = r_pearson_timepoint_pc
                        else:
                            _, n_pcs = predictions.shape
                            if not store_result_by_pc:
                                var_explained_sum = 0
                                r_pearson_sum = 0  
                                for pc in range(n_pcs):
                                    var_explained_sum += r2_score(Y_test[:,pc], predictions[:,pc])
                                    r_pearson_sum += pearsonr(Y_test[:,pc], predictions[:,pc])[0]

                                variance_explained_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred] = var_explained_sum
                                correlation_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred] = r_pearson_sum
                            else:
                                for pc in range(n_pcs):
                                    var_explained_pc = r2_score(Y_test[:,pc], predictions[:,pc])
                                    r_pearson_pc = pearsonr(Y_test[:,pc], predictions[:,pc])[0]

                                    variance_explained_dict["pcs"][pc]["session_mapping"][session_id_model]["session_pred"][session_id_pred] = var_explained_pc
                                    correlation_dict["pcs"][pc]["session_mapping"][session_id_model]["session_pred"][session_id_pred] = r_pearson_pc


                # Store predictions dict
                storage_folder = f"data_files/{self.lock_event}/predicted_meg_responses/source_space{pca_folder}/{glaser_region}/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                os.makedirs(storage_folder, exist_ok=True)
                storage_file = f"predicted_responses_dict.pkl"
                storage_path = os.path.join(storage_folder, storage_file)

                with open(storage_path, 'wb') as file:
                    pickle.dump(predicted_responses_dict, file)

                # Store loss dict
                if not timepoints_pca:
                    storage_dicts_by_folders = {"var_explained_timepoints": variance_explained_dict, "pearson_r_timepoints": correlation_dict}
                else:
                    storage_dicts_by_folders = {"var_explained": variance_explained_dict, "pearson_r": correlation_dict}

                pc_storage_folder = "/pcs_separate" if store_result_by_pc else ""
                for main_folder, fit_measure_dict in storage_dicts_by_folders.items():
                    storage_folder = f"data_files/{self.lock_event}/{main_folder}/source_space{pca_folder}{pc_storage_folder}/{glaser_region}/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                    os.makedirs(storage_folder, exist_ok=True)
                    json_storage_file = f"{main_folder}_dict.json"
                    json_storage_path = os.path.join(storage_folder, json_storage_file)

                    with open(json_storage_path, 'w') as file:
                        json.dump(fit_measure_dict, file, indent=4)


    def predict_from_mapping_simulation_scene_dataset(self):
        """
        Predicts MEG activity from PCA reduced features extracted from the scenes selected for the generation of simulated responses.
        """
        # Load features with reduced dim
        scene_pca_features_folder = f"data_files/{self.lock_event}/simulation_scenes/scenes_features_pca/subject_{self.subject_id}"  
        scene_pca_features_file = f"simulation_scenes_features_pca_by_clusters.npy"
        scene_pca_features_path = os.path.join(scene_pca_features_folder, scene_pca_features_file)
        scene_pca_features = np.load(scene_pca_features_path)  # shape: (n_clusters, n_scenes_per_cluster, self.pca_components)

        # For each session (and each selected norm): Predict/simulate MEG responses for all features/scenes (and thus all clusters)
        for normalization in self.normalizations:
            for session_id in self.session_ids_num:
                # Load and initialize ridge model
                storage_folder = f"data_files/{self.lock_event}/GLM_models/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/session_{session_id}"  
                storage_file = "GLM_models.pkl"
                storage_path = os.path.join(storage_folder, storage_file)
                with open(storage_path, 'rb') as file:
                    ridge_models = pickle.load(file)

                ridge_model = GLMHelper.MultiDimensionalRegression(self, models=ridge_models)

                # Generate predictions for all clusters
                predictions_all_clusters = []
                for cluster_idx in range(scene_pca_features.shape[0]):
                    predictions_cluster = ridge_model.predict(scene_pca_features[cluster_idx], downscale_features=False)
                    predictions_all_clusters.append(predictions_cluster)
                predictions_all_clusters = np.array(predictions_all_clusters)  # shape: (n_clusters, n_scenes_per_cluster, n_sensors, n_timepoints)

                # Store simulated responses
                save_folder = f"data_files/{self.lock_event}/simulation_scenes/simulated_meg_responses/subject_{self.subject_id}/{normalization}"  
                save_file = f"session_{session_id}_simulated_meg_responses_by_clusters.npy"
                save_path = os.path.join(save_folder, save_file)
                os.makedirs(save_folder, exist_ok=True)
                np.save(save_path, predictions_all_clusters)  


    class MultiDimensionalRegression:
        """
        Inner class to apply (fractional) Ridge Regression over all timepoints. Enables training and prediction, as well as initialization of random weights for baseline comparison.
        """
        def __init__(self, GLM_helper_instance, models:list=[], random_weights:bool=False, timepoints_pca:bool=False):
            self.GLM_helper_instance = GLM_helper_instance
            self.random_weights = random_weights
            self.models = models  # Standardly initialized as empty list, otherwise with passed, previously trained models
            self.alphas = self.GLM_helper_instance.alphas
            self.selected_alphas = None
            self.timepoints_pca = timepoints_pca

            if random_weights and timepoints_pca:
                raise NotImplementedError("MultiDimensionalRegression not yet intended for random weights and timepoint-pca data.")

        def fit(self, X=None, Y=None):
            if not self.timepoints_pca:
                n_features = X.shape[1]
                n_sensors = Y.shape[1]
                n_timepoints = Y.shape[2]
                if self.GLM_helper_instance.fractional_ridge:
                    self.models = [FracRidgeRegressorCV() for _ in range(n_timepoints)]
                else:
                    self.models = [RidgeCV(alphas=self.GLM_helper_instance.alphas) for _ in range(n_timepoints)]
                
                logger.custom_debug(f"Fit model with alphas {self.GLM_helper_instance.alphas}")
                if self.random_weights:
                    # Randomly initialize weights and intercepts
                    # Careful, in the current implementation the random model does not use an alpha
                    for model in self.models:
                        model.coef_ = np.random.rand(n_sensors, n_features) - 0.5  # Random weights centered around 0
                        model.intercept_ = np.random.rand(n_sensors) - 0.5  # Random intercepts centered around 0
                else:
                    for t in range(n_timepoints):
                        Y_t = Y[:, :, t]
                        if self.GLM_helper_instance.fractional_ridge:
                            self.models[t].fit(X, Y_t, frac_grid=self.GLM_helper_instance.fractional_grid)
                            if self.models[t].best_frac_ <= 0.000_000_000_000_000_1:  # log if smallest fraction has been chosen
                                logger.custom_debug(f"\n Timepoint {self.GLM_helper_instance.timepoint_min+t}: frac = {self.models[t].best_frac_}, alpha = {self.models[t].alpha_}") 
                        else:
                            self.models[t].fit(X, Y_t)

                # Debugging: For each model (aka for each timepoint) store the alpha/fraction that was selected as best fit in RidgeCV/FracRidgeRegressorCV
                if not self.GLM_helper_instance.fractional_ridge:
                    selected_regularize_param = [timepoint_model.alpha_ for timepoint_model in self.models]
                    param_type = "alphas"
                    self.selected_alphas = selected_regularize_param
                else:
                    selected_regularize_param = [timepoint_model.best_frac_ for timepoint_model in self.models]
                    param_type = "fractions"

                counts_regularize_params = Counter(selected_regularize_param)
                sorted_counts_regularize_params = sorted(counts_regularize_params.items(), key=lambda x: x[1], reverse=True)
                logger.custom_debug(f"selected {param_type}: {sorted_counts_regularize_params}")

            else: # If PCA was performed over timepoints, we only fit a single model
                
                if self.GLM_helper_instance.fractional_ridge:
                    self.models = [FracRidgeRegressorCV()]
                    self.models[0].fit(X, Y, frac_grid=self.GLM_helper_instance.fractional_grid)
                else:
                    self.models = [RidgeCV(alphas=self.GLM_helper_instance.alphas)]
                    self.models[0].fit(X, Y)

        def predict(self, X, downscale_features:bool=False):
            if downscale_features:
                X = self.GLM_helper_instance.normalize_array(data=X, normalization="range_-1_to_1")

            n_samples = X.shape[0]
            if not self.timepoints_pca:
                n_sensors = self.models[0].coef_.shape[0] if not self.GLM_helper_instance.fractional_ridge else self.models[0].coef_.shape[1]
                n_timepoints = len(self.models)
                predictions = np.zeros((n_samples, n_sensors, n_timepoints))

                for t, model in enumerate(self.models):
                    if self.random_weights:
                        # Use the random weights and intercept to predict; we are missing configurations implicitly achieved when calling .fit()
                        predictions[:, :, t] = X @ model.coef_.T + model.intercept_
                    else:
                        predictions[:, :, t] = model.predict(X)
            else:
                predictions = self.models[0].predict(X)

            return predictions    



class VisualizationHelper(GLMHelper):
    def __init__(self, n_grad: int, n_mag: int, time_window_n_indices:int, **kwargs):
        super().__init__(**kwargs)

        self.time_window_n_indices = time_window_n_indices
        self.n_grad = n_grad
        self.n_mag = n_mag

        ### Calculate sessions which should be omitted from drift analysis ###
        # TODO: Adjust session omittance in RDM drift etc based on this result
        self.omit_sessions = self.determine_excluded_drift_sessions()
        

        # investigate session day differences
        session_diffs = self.get_session_date_differences()
        session_diff_1_10 = session_diffs["1"]["10"]

        logger.custom_info(f"Subject {self.subject_id}: Differences in days between recording of sessions 1 and 10: {session_diff_1_10}")


    def visualize_selected_sensor_positions(self):
        """
        Visualizes the positions of the selected MEG sensors on the scalp.
        """
        fif_file_path = f'/share/klab/datasets/avs/population_codes/as{self.subject_id}/sensor/filter_0.2_200/saccade_evoked_02_05_.fif'
        og_evoked = mne.read_evokeds(fif_file_path)[0]
        fig = og_evoked.plot_sensors(show_names=True, ch_type="mag")

        chosen_sensor_names = [f"MEG{channel_id}" for channel_id in self.chosen_channels]
        sensor_points = fig.axes[0].collections[0]
        n_sensors = len(fig.axes[0].texts)
        text_positions_chosen_chs_by_name = {text.get_text(): text.get_position() for text in fig.axes[0].texts if text.get_text() in chosen_sensor_names}

        # Because of missing allignment: Find indices for dots/collection based on closest text values
        sensor_positions_from_plot = sensor_points.get_offsets()
        dot_indices_selected_chs = []
        for sensor_name, text_pos in text_positions_chosen_chs_by_name.items():
            closest_distance_dot_idx = (None, None)
            for dot_idx, dot_pos in enumerate(sensor_positions_from_plot):
                distance_text_dot = np.linalg.norm(dot_pos - text_pos)
                if closest_distance_dot_idx[0] is None or distance_text_dot < closest_distance_dot_idx[0]:
                    closest_distance_dot_idx = distance_text_dot, dot_idx
            dot_indices_selected_chs.append(closest_distance_dot_idx[1])

        # Change color and size of selected dots
        dot_colors = sensor_points.get_facecolor()
        dot_colors[dot_indices_selected_chs] = [1, 0, 0, 1]  # Change to red (RGBA)
        sensor_points.set_facecolor(dot_colors)

        dot_size = sensor_points.get_sizes()
        dot_sizes = np.full(shape=n_sensors, fill_value=dot_size)  # previously stored as single number because identical for all, reformat to array
        dot_sizes[dot_indices_selected_chs] = 100  
        sensor_points.set_sizes(dot_sizes)
   
        # Only keep and modify text for selected sensors
        for sensor_text in fig.axes[0].texts:
            sensor_name = sensor_text.get_text()
            if sensor_name not in chosen_sensor_names:
                sensor_text.set_text("")  # dont show names of non-selected sensors
            else:
                # Adjust font, size and position so that text is below dots
                sensor_id = sensor_text.get_text()[3:]  # exclude 'MEG'
                sensor_text.set_text(sensor_id)
                sensor_text.set_fontsize(11)  
                sensor_text.set_fontweight('demibold')
                x, y = sensor_text.get_position()
                x -= 0.014
                y -= 0.01
                sensor_text.set_position((x,y))

        plot_folder =  f"data_files/{self.lock_event}/visualizations/sensor_locs/"
        plot_file = "sensor_locations.png"
        self.save_plot_as_file(plt=fig, plot_folder=plot_folder, plot_file=plot_file)


    def visualize_session_distances(self):
        """
        Visualizes distance session recordings from recording of first session in days for all participants.
        """
        session_diffs_by_subj = {}
        for curr_subject_id in ["01", "02", "03", "04", "05"]:
            session_diffs_subj = list(self.get_session_date_differences(curr_subject_id)["1"].values())  # load distances from first session
            curr_subject_id = int(curr_subject_id[1])  # "02" -> 2
            session_diffs_by_subj[curr_subject_id] = session_diffs_subj

        plt.figure(figsize=(8, 4))
        colormap = cm.viridis(np.linspace(0, 1, 10))  # colormap for sessions 1-10
        for subject_idx, (curr_subject_id, session_diffs_subj) in enumerate(session_diffs_by_subj.items()):
            curr_subject_id_arr = np.full(shape=[len(session_diffs_subj)], fill_value=curr_subject_id)
            plt.scatter(session_diffs_subj, curr_subject_id_arr, color=colormap, s=80)

        plt.xlabel('Days from first session')
        plt.ylabel('Participant')
        plt.yticks(ticks=list(session_diffs_by_subj.keys()), labels=list(session_diffs_by_subj.keys()))
        plt.tick_params(left=False, bottom=False) 

        # Add colorbar legend for session gradient
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=1, vmax=10))
        cbar = plt.colorbar(sm, orientation="vertical", ticks=[1, 10])
        cbar.set_ticklabels(['1', '10'])  
        cbar.set_label("Sessions 1–10", fontsize=10, labelpad=0, rotation=270)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        save_folder = f"data_files/{self.lock_event}/visualizations/session_distances"  
        save_file = f"session_distances_all_subjects.png"
        self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)
        plt.close()


    def determine_excluded_drift_sessions(self):
        """
        Calculates which sessions should be omitted from drift analysis for the current subject based on self-session encoding performances for sessions.
        Sessions which show a negative deviation from the mean of more than 1.5 stds will be excluded.
        """
        assert "global_robust_scaling" in self.normalizations, "Session omittance calculation will always be performed based on global_robust_scaling normalization (and timepoint variance)."
        
        storage_folder = f"data_files/{self.lock_event}/var_explained_timepoints/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_global_robust_scaling/"
        json_storage_file = f"var_explained_timepoints_dict.json"
        json_storage_path = os.path.join(storage_folder, json_storage_file)
        with open(json_storage_path, 'r') as file:
            session_fit_measures = json.load(file)

        # Extract self session encoding results and calculate respective average for each session
        fit_measures_by_session_by_timepoint = {"session": {}}
        avg_self_session_fit_measure_by_session = {}
        num_timepoints = (self.timepoint_max - self.timepoint_min) + 1  # +1 because of index 0 ofc
        for session_id in session_fit_measures['session_mapping']:
            fit_measures_by_session_by_timepoint["session"][session_id] = {"timepoint":{}}
            session_fit_measure_sum = 0
            n_timepoints = len(list(session_fit_measures['session_mapping'][session_id]["session_pred"][session_id]["timepoint"].keys()))
            for timepoint, fit_measure_timepoint in session_fit_measures['session_mapping'][session_id]["session_pred"][session_id]["timepoint"].items():
                fit_measures_by_session_by_timepoint["session"][session_id]["timepoint"][timepoint] = fit_measure_timepoint
                session_fit_measure_sum += fit_measure_timepoint
            avg_self_session_fit_measure_by_session[session_id] = session_fit_measure_sum / num_timepoints

        all_session_encoding_vals = list(avg_self_session_fit_measure_by_session.values())
        avg_all_sessions = np.sum(all_session_encoding_vals) / len(all_session_encoding_vals)
        std_all_sessions = np.std(all_session_encoding_vals)
        cutoff = avg_all_sessions - (std_all_sessions*1.5)
        excluded_sessions = [session_id for session_id, session_encoding_val in avg_self_session_fit_measure_by_session.items() if session_encoding_val < cutoff or session_encoding_val < 0]

        logger.custom_info(f"Subject {self.subject_id}: omitting sessions {excluded_sessions}")
        logger.custom_info(f"avg_self_session_fit_measure_by_session: {avg_self_session_fit_measure_by_session}")
        logger.custom_info(f"avg_all_sessions: {avg_all_sessions}")
        logger.custom_info(f"std_all_sessions: {std_all_sessions}")
        logger.custom_info(f"cutoff: {cutoff} \n")
        

        return excluded_sessions


    def investigate_source_df(self):
        """
        source_file_path = f'/share/klab/datasets/avs/population_codes/as{self.subject_id}/source_space/dSPM/glasser/ori_normal/hem_lh/filter_0.2_200/ica/as{self.subject_id}a_et_epochs_info_saccade.fif' #_population_codes_saccade_500hz_masked_False.h5

        # Load the source estimate (dSPM results) from the .fif file
        stc = mne.read_source_estimate(source_file_path)

        # Check the data shape and time range
        print(f"Shape of source estimate data: {stc.data.shape}")
        print(f"Time points (in seconds): {stc.times}")
        print(f"Number of vertices in each hemisphere: {len(stc.vertices[0])}, {len(stc.vertices[1])}")

        # You can access the data array (source estimates)
        # stc.data gives a 2D array: [n_sources x n_times]
        print(f"First few data points:\n{stc.data[:5, :5]}")

        # Get the time step (difference between consecutive time points)
        print(f"Time step: {stc.tstep}")

        # Plot the source estimate for inspection (using the brain surface)
        brain_plot = stc.plot(subject='sample', hemi='both', subjects_dir='path_to_your_subjects_dir')

        plot_folder =  f"data_files/{self.lock_event}/visualizations/source_space/visual_inspection"
        plot_file = f"source_space_estimation.png"
        os.makedirs(plot_folder, exist_ok=True)
        self.save_plot_as_file(plt=brain_plot, plot_folder=plot_folder, plot_file=plot_file)

        # Plot time series for the first few sources (vertices)
        #stc.plot_time_series(hemi='both')
        """
        source_file_path = f'/share/klab/datasets/avs/population_codes/as02/source_space/beamformer/glasser/ori_None/hem_lh/filter_0.2_200/ica/as{self.subject_id}a_population_codes_fixation_500hz_masked_False.h5'
        with h5py.File(source_file_path, "r") as f:
            print(f"f.keys(): {f.keys()}")
            print(f"len f.keys(): {len(f.keys())}")
            print(f"H5 f.attrs.keys(): {f.attrs.keys()}")

            print(f"f['V3']['onset'].shape: {f['V3']['onset'].shape}")
            print(f"f['V3A']['onset'].shape: {f['V3A']['onset'].shape}")
            print(f"f['V3B']['onset'].shape: {f['V3B']['onset'].shape}")

            print(f"H5 f.attrs['rois'].shape: {f.attrs['rois'].shape}")
            print(f"H5 f.attrs['rois']: {f.attrs['rois']}")



    def visualize_arousal_mean_over_sessions(self):
        """
        Visualizes the average pupil dialation for each session as recorded by the eye-tracking software.
        """
        mean_pupil_dilations_by_day = {}
        session_date_differences = self.get_session_date_differences()
        for session_id in self.session_ids_num:
            session_double_char_id = f"0{session_id}" if session_id != "10" else session_id
            events_eyetracking_path = f"/share/klab/datasets/avs/results/as{self.subject_id}_{session_double_char_id}/preprocessed/as_s{self.subject_id[1]}_el_events.csv"
            samples_eyetracking_path = f"/share/klab/datasets/avs/results/as{self.subject_id}_{session_double_char_id}/preprocessed/as_s{self.subject_id[1]}_el_samples.csv"
           
            events_df = pd.read_csv(events_eyetracking_path)
            samples_df = pd.read_csv(samples_eyetracking_path)

            #print(f"samples_eyetracking_path: {samples_eyetracking_path}")
            #events_df.info()
            #samples_df.info()            

            # Filter events for only fixation events and reset index
            events_df = events_df[events_df["type"] == "fixation"].reset_index(drop=True)
            samples_df = samples_df[samples_df["type"] == "fixation"].reset_index(drop=True)

            # Debugging
            #print(f"samples_df['smpl_time'].to_numpy()[:100]: {samples_df['smpl_time'].to_numpy()[:100]}")
            #print(f"events_df['start_time'].to_numpy()[:100]: {events_df['start_time'].to_numpy()[:100]}")
            #sys.exit()

            events_start_ordered = events_df['start_time'].is_monotonic_increasing
            events_end_ordered = events_df['end_time'].is_monotonic_increasing
            samples_ordered = samples_df['smpl_time'].is_monotonic_increasing
            assert events_start_ordered and events_end_ordered and samples_ordered, "Timepoint values for at least one df are not sorted."

            # Extract values from samples only for fixations in the events_df. smpl_time needs to be between a given start_time and end_time belonging to a fixation.
            pupil_vals_by_fixation = defaultdict(list)
            start_sample_idx = 0
            for fixation_index, fixation_df_row in events_df.iterrows():
                start_time = fixation_df_row['start_time']
                end_time = fixation_df_row['end_time']
                for _, simpl_time in samples_df['smpl_time'].iloc[start_sample_idx:].items():
                    if start_time > simpl_time:
                        # We are not far enough in the samples dataframe
                        start_sample_idx += 1
                    elif simpl_time <= end_time:
                        # simpl_time is greater than start_time and smaller than or equal to end_time of current fixation --> match
                        pupil_vals_by_fixation[fixation_index].append(simpl_time)
                        start_sample_idx += 1
                    else:
                        # simpl_time is greater than start_time AND greater than end_time: We are not far enough in the events dataframe
                        break
            # Compute mean per fixation and per session
            mean_pupil_vals_by_fixation = {fixation_index: np.mean(fixation_values) for fixation_index, fixation_values in pupil_vals_by_fixation.items()}
            mean_val_session = np.mean(list(mean_pupil_vals_by_fixation.values()))
            median_val_session = np.median(list(mean_pupil_vals_by_fixation.values()))
            std_val_session = np.std([pupil_val for fixation_values in pupil_vals_by_fixation.values() for pupil_val in fixation_values])

            n_total_pupil_vals_session = 0
            for pupil_vals_fixation in pupil_vals_by_fixation.values():
                n_total_pupil_vals_session += len(pupil_vals_fixation)
    
            print(f"session [{session_id}]: Extracted pupil values for {len(pupil_vals_by_fixation.keys())} fixations. Total pupil values: {n_total_pupil_vals_session}")
            print(f"session [{session_id}]: Standard deviation of pupil dilation is {std_val_session}.")
            print(f"session [{session_id}]: Median pupil dilation is {median_val_session}.")
            print(f"session [{session_id}]: Mean pupil dilation is {mean_val_session}. \n")
           

            # aggregate values by distance in days from first session
            days_from_session_1 = session_date_differences["1"][session_id]
            mean_pupil_dilations_by_day[days_from_session_1] = list(mean_pupil_vals_by_fixation.values())

        global_mean_pupil_dilations_for_sessions = [np.mean(mean_pupil_dilations_by_day[session_id_int]) for session_id_int in sorted(mean_pupil_dilations_by_day, key=int)]

        # Plot average pupil dilation across sessions
        plt.figure(figsize=(8, 6))
        x_values, y_values = self.extract_x_y_arrays(mean_pupil_dilations_by_day, losses_averaged_within_distances=False)
        x_values_set = np.array(list(set(x_values)))
        
        slope, intercept, r_value, _, std_err = linregress(x=x_values, y=y_values)
        trend_line = slope * x_values_set + intercept
        r_value = "{:.3f}".format(r_value) 
        #p_value = "{:.4f}".format(self.perform_permutation_test(x_values, y_values, direction="both"))
        
        plt.plot(x_values_set, trend_line, color='green' , label=f"r={r_value}", linestyle='-', linewidth=3)
        plt.plot(x_values_set, global_mean_pupil_dilations_for_sessions, marker='o', linestyle='', markersize=3)
        plt.title(f'Pupil dilation across sessions subject {self.subject_id}. \n Averaged across measurements within each fixation in each session.')
        plt.xlabel('Days since Session 1')
        plt.ylabel('Pupil Dilation')
        plt.legend()
        #plt.xticks(x_values) 
        plt.grid(True)

        save_folder = f"data_files/{self.lock_event}/visualizations/pupil_dilation"  
        save_file = f"pupil_dilations_across_sessions_subject_{self.subject_id}.png"
        self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)
        plt.close()
     

    def calculate_RSM_test_set_drift(self, calculate_anew:bool):
        """
        Investigate RSM drift based on test sets of all sessions: Calculate distance between true RSM and RSM of predicted values.
        """
        session_ids = np.array([session_id for session_id in self.session_ids_num if session_id not in self.omit_sessions])
        session_day_differences = self.get_session_date_differences()

        for normalization in self.normalizations:
            if calculate_anew:
                ### Calculate (true) RSM of responses (for each sessions test set) ###
                true_rsms_by_session = {"sessions": {}}
                for session_id in session_ids:
                    session_test_split_path = f"data_files/{self.lock_event}/meg_data/norm_{normalization}/subject_{self.subject_id}/session_{session_id}/test/meg_data.npy"  
                    session_test_data = np.load(session_test_split_path)  # shape: (n_epochs, n_channels, n_timepoints)
                    n_epochs, n_channels, n_timepoints = session_test_data.shape  # epochs = crops

                    rsm_matrix_session = np.zeros(shape=(n_epochs, n_epochs))
                    for epoch1_idx in range(n_epochs):
                        for epoch2_idx in range(epoch1_idx, n_epochs):
                            # ! Flattening is only valid without consideration for correlation, not variance explained etc. !
                            epoch_corr = pearsonr(session_test_data[epoch1_idx].flatten(),session_test_data[epoch2_idx].flatten())[0]
                            rsm_matrix_session[epoch1_idx][epoch2_idx] = epoch_corr
                    
                    # Store upper triangular part without diagonal (only part with values)
                    rsm_matrix_session = rsm_matrix_session[np.triu_indices_from(rsm_matrix_session, k=1)]
                    true_rsms_by_session["sessions"][session_id] = rsm_matrix_session

                    if 0 in rsm_matrix_session:
                        raise ValueError("Error in upper tri extraction.")

                ### Calculate RSMs of predicted responses and compare to true RSMs ###

                # Load predicted responses
                storage_folder = f"data_files/{self.lock_event}/predicted_meg_responses/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                storage_file = f"predicted_responses_dict.pkl"
                storage_path = os.path.join(storage_folder, storage_file)
                with open(storage_path, 'rb') as file:
                    predicted_responses_dict = pickle.load(file)

                rsm_corr_by_distance = {}
                for session_id_test_set in session_ids:
                    true_rsm_matrix_session = true_rsms_by_session["sessions"][session_id_test_set]
                    for session_id_model in session_ids:
                        predicted_responses = predicted_responses_dict["session_mapping"][session_id_model]["session_pred"][session_id_test_set]
                        n_epochs, n_channels, n_timepoints = predicted_responses.shape

                        # Calculate RSM of predicted responses                        
                        predicted_rsm_matrix = np.zeros(shape=(n_epochs, n_epochs))
                        for epoch1_idx in range(n_epochs):
                            for epoch2_idx in range(epoch1_idx, n_epochs):
                                # ! Flattening is only valid without consideration for correlation, not variance explained etc. !
                                epoch_corr = pearsonr(predicted_responses[epoch1_idx].flatten(),predicted_responses[epoch2_idx].flatten())[0]
                                predicted_rsm_matrix[epoch1_idx][epoch2_idx] = epoch_corr
                        # Extract upper triangular part without diagonal (only part with values)
                        predicted_rsm_matrix = predicted_rsm_matrix[np.triu_indices_from(predicted_rsm_matrix, k=1)]

                        # Calculate correlation of predicted and true RSM, stored as fit measure by distance
                        distance = session_day_differences[session_id_model][session_id_test_set]
                        rsm_spearman_r, _ = spearmanr(true_rsm_matrix_session, predicted_rsm_matrix)

                        if distance not in rsm_corr_by_distance.keys():
                            rsm_corr_by_distance[distance] = [rsm_spearman_r]
                        else:
                            rsm_corr_by_distance[distance].append(rsm_spearman_r)

                # Store computation results
                save_folder = f"data_files/{self.lock_event}/rsm_corr_by_distance/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                os.makedirs(save_folder, exist_ok=True)
                save_file = f"rsm_corr_by_distance.json"
                save_path = os.path.join(save_folder, save_file)
                with open(save_path, 'w') as file:
                    json.dump(rsm_corr_by_distance, file, indent=4)
            else:
                # Load from previous calculation (takes ~2hours)
                storage_folder = f"data_files/{self.lock_event}/rsm_corr_by_distance/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                storage_file = f"rsm_corr_by_distance.json"
                storage_path = os.path.join(storage_folder, storage_file)
                with open(storage_path, 'r') as file:
                    rsm_corr_by_distance = json.load(file)

            # DEPRECATED: shortcut to exclude session 4 for subject 2 based on distance
            #if self.subject_id == "02":
            #    rsm_corr_by_distance = {distance: rsm_corr for distance, rsm_corr in rsm_corr_by_distance.items() if int(distance) < 60}


            # Plot drift
            plt.figure(figsize=(8, 6))
            x_values, y_values = self.extract_x_y_arrays(fit_measures_by_distances=rsm_corr_by_distance, losses_averaged_within_distances=False)
            filtered_x_values_set = np.array(list(set(x_values)))

            slope, intercept, r_value, p_value, std_err = linregress(x=x_values, y=y_values)
            trend_line = slope * filtered_x_values_set + intercept
            r_value = "{:.3f}".format(r_value)  # limit to three decimals
            p_value = self.perform_permutation_test(x_values, y_values, direction="both")
            plt.plot(x_values, y_values, marker='o', linestyle='none')
            plt.plot(filtered_x_values_set, trend_line, color='green' , label=f"r={r_value}, p={p_value}", linestyle='-', linewidth=3)
            plt.title(f'Correlations of (upper triangular) true RSMs and predicted RSMs by distance. \n Subject {self.subject_id}. Omitted Sessions: {self.omit_sessions}')
            plt.xlabel('Distance in days between test set session and model session')
            plt.ylabel('Spearman Correlation')
            plt.legend()
            plt.grid(True)

            save_folder = f"data_files/{self.lock_event}/visualizations/RSM_Test_Set_Drift/Distance_Plots/{normalization}"  
            save_file = f"RSM_Test_Set_Drift_subject_{self.subject_id}.png"
            self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)
            plt.close()

    def calculate_and_visualize_cluster_geometry_RSMs_simulated_responses(self, image_level:bool, omit_sessions_from_corr:list):
        """
        Calculates response simlarity matrices for the simulated responses for the 60 semantic clusters, or on image level. For each session, the similarity of all clusters/images is calculated. 
        Comparison of RSMs will show if the 'geometry' of the representations of the clusters/images (i.e. the cluster/image representations relative to each other, operationalized by the RSA)) changes over sessions.
        """
        comparison_level = "image_level" if image_level else "cluster_level"

        session_day_differences = self.get_session_date_differences()
        session_ids = np.array([session_id for session_id in self.session_ids_num if session_id not in omit_sessions_from_corr])
        n_sessions = len(session_ids)
        for normalization in self.normalizations:
            upper_tri_rsm_by_session = {"session": {}}  # Placeholder to store upper triangular part of RSM matrices of sessions for later correlations of RSMs
            for session_id in session_ids:
                # Load simulated responses of session's model
                simulated_responses_folder = f"data_files/{self.lock_event}/simulation_scenes/simulated_meg_responses/subject_{self.subject_id}/{normalization}"  
                simulated_responses_file = f"session_{session_id}_simulated_meg_responses_by_clusters.npy"
                simulated_responses_path = os.path.join(simulated_responses_folder, simulated_responses_file)
                simulated_responses = np.load(simulated_responses_path)  # shape: (n_clusters, n_scenes_per_cluster, n_sensors, n_timepoints) f.e. (60, 5, 5, 31)
                #print(f"pre reshape simulated_responses.shape: {simulated_responses.shape}")
                # Compare responses on image level
                if image_level:
                    # Reshape: eliminate cluster dimension
                    simulated_responses = simulated_responses.reshape(-1, simulated_responses.shape[2], simulated_responses.shape[3])  # shape: (n_clusters x n_scenes_per_cluster, n_sensors, n_timepoints)
                
                    # Correlate simulated responses for all images
                    n_images, n_sensors, n_timepoints = simulated_responses.shape
                    rsm_matrix_session = np.zeros(shape=(n_images, n_images))

                    #print(f"post reshape simulated_responses.shape: {simulated_responses.shape}")

                    for image1_idx in range(n_images):
                        for image2_idx in range(image1_idx, n_images):
                            # ! Flattening is only valid without consideration for correlation, not variance explained etc. !
                            image_corr = pearsonr(simulated_responses[image1_idx].flatten(),simulated_responses[image2_idx].flatten())[0]
                            rsm_matrix_session[image1_idx][image2_idx] = image_corr

                            if image1_idx != image2_idx:  # Fill in lower triangular part of matrix
                                rsm_matrix_session[image2_idx, image1_idx] = image_corr
                # Compare responses on cluster level
                else:
                    # Correlate the simulated responses for all clusters
                    n_clusters, _, n_sensors, n_timepoints = simulated_responses.shape
                    rsm_matrix_session = np.zeros(shape=(n_clusters, n_clusters))
                    for cluster1_idx in range(n_clusters):
                        for cluster2_idx in range(cluster1_idx, n_clusters):
                            # ! Flattening is only valid without consideration for correlation, not variance explained etc. !
                            cluster_corr = pearsonr(simulated_responses[cluster1_idx].flatten(),simulated_responses[cluster2_idx].flatten())[0]
                            rsm_matrix_session[cluster1_idx][cluster2_idx] = cluster_corr

                            if cluster1_idx != cluster2_idx:  # Fill in lower triangular part of matrix
                                rsm_matrix_session[cluster2_idx, cluster1_idx] = cluster_corr

                # Extract upper triangular part for correlation between sessions (exluding doubled values and diagonal)
                upper_tri_rsm_by_session["session"][session_id] = rsm_matrix_session[np.triu_indices_from(rsm_matrix_session, k=1)]

                # Store RSM
                save_folder = f"data_files/{self.lock_event}/simulation_scenes/RSMs/{comparison_level}/subject_{self.subject_id}/{normalization}"  
                save_file = f"session_{session_id}_RSM.npy"
                save_path = os.path.join(save_folder, save_file)
                os.makedirs(save_folder, exist_ok=True)
                np.save(save_path, rsm_matrix_session)  # shape: (n_clusters, n_clusters)

                # Visualize RSM
                plt.figure(figsize=(10, 8))
                plt.imshow(rsm_matrix_session, cmap='viridis', aspect='auto')
                plt.colorbar(label='Pearson Correlation')
                plt.title(f'{comparison_level} Geometry RSM for Subject {self.subject_id}, Session {session_id}')
                x_y_label = 'Cluster Index' if not image_level else 'Image Index'
                plt.xlabel(x_y_label)
                plt.ylabel(x_y_label)

                save_folder = f"data_files/{self.lock_event}/visualizations/simulation_scenes/RSMs/{comparison_level}/Geometry_Singular_Session/subject_{self.subject_id}/{normalization}"  
                save_file = f"session_{session_id}_{comparison_level}_geometry_RSM.png"
                self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)
                plt.close()
            

            # Plot spearman correlation of session RSMs as a function of the distance between the sessions which the models simulating the responses, based on which the RSMs were calculated, were trained on. (Phew!)
            # Also visualize values of this plot as RSM where each value is the correlation of the cluster/image geometry RSM between two sessions
            geometry_session_comp_matrix = np.zeros(shape=(n_sessions, n_sessions))
            fig, ax1 = plt.subplots(figsize=(12, 8))
            # Loop over session combinations (no doubles and some sessions may be omitted), obtaining correlation by distances in days
            session_pairs = list(itertools.combinations([int(session_id) for session_id in session_ids], 2))  # all combinations to be considered
            rsm_corr_by_distance = {}
            for session_id_1, session_id_2 in session_pairs:
                session_id_1_str, session_id_2_str = str(session_id_1), str(session_id_2)
                # Extract matrix indices. not necessarily = session_id because of omitted sessions
                session_matrix_index_1 = np.where(session_ids == session_id_1_str)[0][0] 
                session_matrix_index_2 = np.where(session_ids == session_id_2_str)[0][0] 
                # Extract upper tri of session RSMs (cluster geometry)
                upper_tri_rsm_session_1, upper_tri_rsm_session_2 = upper_tri_rsm_by_session["session"][session_id_1_str], upper_tri_rsm_by_session["session"][session_id_2_str]
                distance = session_day_differences[session_id_1_str][session_id_2_str]
                #rsm_corr = pearsonr(upper_tri_rsm_session_1, upper_tri_rsm_session_2)[0]
                rsm_spearman_r, _ = spearmanr(upper_tri_rsm_session_1, upper_tri_rsm_session_2)
                if distance not in rsm_corr_by_distance.keys():
                    rsm_corr_by_distance[distance] = [rsm_spearman_r]
                else:
                    rsm_corr_by_distance[distance].append(rsm_spearman_r)
                # Also store corr for RSM visualization
                geometry_session_comp_matrix[session_matrix_index_1][session_matrix_index_2] = rsm_spearman_r
                geometry_session_comp_matrix[session_matrix_index_2][session_matrix_index_1] = rsm_spearman_r
            # Fill in diagonal values (always 1)
            for session_idx in range(n_sessions):
                geometry_session_comp_matrix[session_idx][session_idx] = 1

            # Plot correlations of session cluster geometry RSMs by session distance
            plt.figure(figsize=(8, 6))
            x_values, y_values = self.extract_x_y_arrays(fit_measures_by_distances=rsm_corr_by_distance, losses_averaged_within_distances=False)
            filtered_x_values_set = np.array(list(set(x_values)))
            slope, intercept, r_value, p_value, std_err = linregress(x=x_values, y=y_values)
            trend_line = slope * filtered_x_values_set + intercept
            r_value = "{:.3f}".format(r_value)  # limit to three decimals
            plt.plot(filtered_x_values_set, trend_line, color='green' , label=f"r={r_value}", linestyle='-', linewidth=3)
            plt.plot(x_values, y_values, marker='o', linestyle='')
            plt.title(f'Correlations of (upper triangular) {comparison_level} geometry RSMs between sessions by distance.')
            plt.xlabel('Distance in days between RSM Sessions')
            plt.ylabel('Spearman Correlation')
            plt.legend()
            plt.xticks(x_values) 
            plt.grid(True)

            save_folder = f"data_files/{self.lock_event}/visualizations/simulation_scenes/Distance_Plots/RSM_between_session_corrs/{comparison_level}/subject_{self.subject_id}/{normalization}"  
            save_file = f"session_{comparison_level}_geometry_RSM_corrs_by_distance.png"
            self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)
            plt.close()

            # Visualize same values as RSM
            plt.figure(figsize=(10, 8))
            plt.imshow(geometry_session_comp_matrix, cmap='viridis', aspect='auto')
            plt.colorbar(label='Spearman Correlation')
            plt.title(f'RSM for Subject {self.subject_id}, Correlations of cluster geometry RSMs between Sessions.')
            plt.xlabel('Session Index')
            plt.ylabel('Session Index')

            save_folder = f"data_files/{self.lock_event}/visualizations/simulation_scenes/RSMs/Geometry_Session_Similarities/{comparison_level}/subject_{self.subject_id}/{normalization}"  
            save_file = f"{comparison_level}_Geometry_Session_Similarities.png"
            self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)
            plt.close()


    def calculate_and_visualize_between_sessions_RSMs_simulated_responses(self, image_level:bool, omit_sessions_from_corr:list):
        """
        Calculates response simlarity matrices for the simulated responses for the sessions. For each cluster/image, the similarity of all sessions is calculated.
        RSM will show if the representation of a given cluster/image changes over sessions.
        """
        comparison_level = "image_level" if image_level else "cluster_level"

        session_day_differences = self.get_session_date_differences()
        session_ids = np.array([session_id for session_id in self.session_ids_num if session_id not in omit_sessions_from_corr])
        session_pairs = list(itertools.combinations([int(session_id) for session_id in session_ids], 2)) # all session combinations to be considered
        n_sessions = len(session_ids)
        for normalization in self.normalizations:
            simulated_responses_by_sessions = {"sessions": {}}
            # Load simulated responses of all sessions
            for session_id in session_ids:
                simulated_responses_folder = f"data_files/{self.lock_event}/simulation_scenes/simulated_meg_responses/subject_{self.subject_id}/{normalization}"  
                simulated_responses_file = f"session_{session_id}_simulated_meg_responses_by_clusters.npy"
                simulated_responses_path = os.path.join(simulated_responses_folder, simulated_responses_file)
                simulated_responses = np.load(simulated_responses_path)  # shape: (n_clusters, n_scenes_per_cluster, n_sensors, n_timepoints) f.e. (60, 5, 5, 31)
                if image_level:
                    # Reshape: eliminate cluster dimension
                    simulated_responses = simulated_responses.reshape(-1, simulated_responses.shape[2], simulated_responses.shape[3])  # shape: (n_clusters x n_scenes_per_cluster, n_sensors, n_timepoints)
                simulated_responses_by_sessions["sessions"][session_id] = simulated_responses
            
            # For each image, correlate simulated responses for all session combinations with spearmans r
            if image_level:
                n_images, n_sensors, n_timepoints = simulated_responses.shape
                rsm_matrices_images = []
                for image_idx in range(n_images):
                    rsm_matrix_image = np.zeros(shape=(n_sessions, n_sessions))
                    for session_id_1, session_id_2 in session_pairs:
                        session_id_1_str, session_id_2_str = str(session_id_1), str(session_id_2)  # use as dict keys
                        # Extract matrix indices. not necessarily = session_id because of omitted sessions
                        session_matrix_index_1 = np.where(session_ids == session_id_1_str)[0][0] 
                        session_matrix_index_2 = np.where(session_ids == session_id_2_str)[0][0] 
                        # ! Flattening is only valid without consideration for correlation, not variance explained etc. !
                        image_sim_responses_sess_1 = simulated_responses_by_sessions["sessions"][session_id_1_str][image_idx]
                        image_sim_responses_sess_2 = simulated_responses_by_sessions["sessions"][session_id_2_str][image_idx]
                        session_corr = pearsonr(image_sim_responses_sess_1.flatten(),image_sim_responses_sess_2.flatten())[0]
                        #session_corr_spearman_r, _ = spearmanr(image_sim_responses_sess_1.flatten(),image_sim_responses_sess_2.flatten())
                        rsm_matrix_image[session_matrix_index_1][session_matrix_index_2] = session_corr

                        if session_id_1 != session_id_2:  # Fill in lower triangular part of matrix
                            rsm_matrix_image[session_matrix_index_2][session_matrix_index_1] = session_corr
                    rsm_matrices_images.append(rsm_matrix_image)

            # For each Cluster, correlate simulated responses for all session combinations with spearmans r
            else:
                n_clusters, n_scenes_per_cluster, n_sensors, n_timepoints = simulated_responses.shape
                rsm_matrices_clusters = []
                for cluster_idx in range(n_clusters):
                    rsm_matrix_cluster = np.zeros(shape=(n_sessions, n_sessions))
                    for session_id_1, session_id_2 in session_pairs:
                        session_id_1_str, session_id_2_str = str(session_id_1), str(session_id_2)  # use as dict keys
                        # Extract matrix indices. not necessarily = session_id because of omitted sessions
                        session_matrix_index_1 = np.where(session_ids == session_id_1_str)[0][0] 
                        session_matrix_index_2 = np.where(session_ids == session_id_2_str)[0][0] 
                        # ! Flattening is only valid without consideration for correlation, not variance explained etc. !
                        cluster_sim_responses_sess_1 = simulated_responses_by_sessions["sessions"][session_id_1_str][cluster_idx]
                        cluster_sim_responses_sess_2 = simulated_responses_by_sessions["sessions"][session_id_2_str][cluster_idx]
                        session_corr = pearsonr(cluster_sim_responses_sess_1.flatten(),cluster_sim_responses_sess_2.flatten())[0]
                        rsm_matrix_cluster[session_matrix_index_1][session_matrix_index_2] = session_corr

                        if session_id_1 != session_id_2:  # Fill in lower triangular part of matrix
                            rsm_matrix_cluster[session_matrix_index_2][session_matrix_index_1] = session_corr
                    rsm_matrices_clusters.append(rsm_matrix_cluster)

                    # Visualize cluster RSM
                    plt.figure(figsize=(10, 8))
                    plt.imshow(rsm_matrix_cluster, cmap='viridis', aspect='auto')
                    plt.colorbar(label='Pearson Correlation')
                    plt.title(f'RSM for Subject {self.subject_id}, Cluser_idx {cluster_idx} \n Correlation of simulated reponses for this cluster between sessions.')
                    plt.xlabel('Session Index')
                    plt.ylabel('Session Index')

                    save_folder = f"data_files/{self.lock_event}/visualizations/simulation_scenes/RSMs/Session_Similarities_Singular_Clusters/subject_{self.subject_id}/{normalization}"  
                    save_file = f"Cluster_idx_{cluster_idx}_Session_Similarities.png"
                    self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)
                    plt.close()

                    # Create distance plot for current cluster
                    # Extract correlation between distance from RSM for current cluster
                    rsm_cluster_corr_by_distance = {}
                    for session_id_1, session_id_2 in session_pairs:
                        # Extract matrix indices. not necessarily = session_id because of omitted sessions
                        session_id_1_str, session_id_2_str = str(session_id_1), str(session_id_2)  # use as dict keys
                        session_matrix_index_1 = np.where(session_ids == session_id_1_str)[0][0] 
                        session_matrix_index_2 = np.where(session_ids == session_id_2_str)[0][0] 

                        # Get distance between sessions and store corresponding corr
                        session_distance = session_day_differences[session_id_1_str][session_id_2_str]
                        session_corr_cluster = rsm_matrix_cluster[session_matrix_index_1][session_matrix_index_2]
                        if session_distance not in rsm_cluster_corr_by_distance.keys():
                            rsm_cluster_corr_by_distance[session_distance] = [session_corr_cluster]
                        else:
                            rsm_cluster_corr_by_distance[session_distance].append(session_corr_cluster)

                    # Plot correlations of simulated current cluster responses across sessions by session distance
                    plt.figure(figsize=(8, 6))
                    x_values, y_values = self.extract_x_y_arrays(fit_measures_by_distances=rsm_cluster_corr_by_distance, losses_averaged_within_distances=False)
                    filtered_x_values_set = np.array(list(set(x_values)))
                    slope, intercept, r_value, p_value, std_err = linregress(x=x_values, y=y_values)
                    trend_line = slope * filtered_x_values_set + intercept
                    r_value = "{:.3f}".format(r_value)  # limit to three decimals
                    plt.plot(filtered_x_values_set, trend_line, color='green' , label=f"r={r_value}", linestyle='-', linewidth=3)
                    plt.plot(x_values, y_values, marker='o', linestyle='')
                    plt.title(f'Correlations of simulated responses cluster_idx {cluster_idx} responses by session distance.')
                    plt.xlabel('Distance in days between RSM Sessions')
                    plt.ylabel('Pearson Correlation')
                    plt.legend()
                    plt.xticks(x_values) 
                    plt.grid(True)

                    save_folder = f"data_files/{self.lock_event}/visualizations/simulation_scenes/Distance_Plots/Session_Similarities_Singular_Clusters/subject_{self.subject_id}/{normalization}"  
                    save_file = f"Cluster_idx_{cluster_idx}_session_corrs_by_distance.png"
                    self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)
                    plt.close()

            # Calculate average across RSMs of all clusters/images 
            if image_level:
                rsm_matrices_list = rsm_matrices_images
            else:
                rsm_matrices_list = rsm_matrices_clusters
            matrix_type = "Images" if image_level else "Clusters"
            
            rsm_matrices_arr = np.array(rsm_matrices_list) # clusters: shape: (60, 9, 9)
            rsm_matrix_all_avg = np.mean(rsm_matrices_arr, axis=0) # clusters: shape: (9, 9)

            # Visualize average cluster/image RSM
            plt.figure(figsize=(10, 8))
            plt.imshow(rsm_matrix_all_avg, cmap='viridis', aspect='auto')
            plt.colorbar(label='Pearson Correlation')
            plt.title(f'RSM for Subject {self.subject_id}, session differences averaged across all {matrix_type}.')
            plt.xlabel('Session Index')
            plt.ylabel('Session Index')

            save_folder = f"data_files/{self.lock_event}/visualizations/simulation_scenes/RSMs/All_{matrix_type}_Averaged_Session_Similarities/subject_{self.subject_id}/{normalization}"  
            save_file = f"All_{matrix_type}_Averaged_Session_Similarities.png"
            self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)
            plt.close()
            
            # Extract correlation between distance from RSM averaged over clusters
            rsm_averaged_corr_by_distance = {}
            for session_id_1, session_id_2 in session_pairs:
                # Extract matrix indices. not necessarily = session_id because of omitted sessions
                session_id_1_str, session_id_2_str = str(session_id_1), str(session_id_2)  # use as dict keys
                session_matrix_index_1 = np.where(session_ids == session_id_1_str)[0][0] 
                session_matrix_index_2 = np.where(session_ids == session_id_2_str)[0][0] 

                # Get distance between sessions and store corresponding corr
                session_distance = session_day_differences[session_id_1_str][session_id_2_str]
                session_corr_avg = rsm_matrix_all_avg[session_matrix_index_1][session_matrix_index_2]
                if session_distance not in rsm_averaged_corr_by_distance.keys():
                    rsm_averaged_corr_by_distance[session_distance] = [session_corr_avg]
                else:
                    rsm_averaged_corr_by_distance[session_distance].append(session_corr_avg)


            # Plot correlations of simulated cluster responses across sessions averaged across all clusters by session distance
            plt.figure(figsize=(8, 6))
            x_values, y_values = self.extract_x_y_arrays(fit_measures_by_distances=rsm_averaged_corr_by_distance, losses_averaged_within_distances=False)
            filtered_x_values_set = np.array(list(set(x_values)))
            slope, intercept, r_value, p_value, std_err = linregress(x=x_values, y=y_values)
            trend_line = slope * filtered_x_values_set + intercept
            r_value = "{:.3f}".format(r_value)  # limit to three decimals
            plt.plot(filtered_x_values_set, trend_line, color='green' , label=f"r={r_value}", linestyle='-', linewidth=3)
            plt.plot(x_values, y_values, marker='o', linestyle='')
            plt.title(f'Correlations of simulated {matrix_type} responses by session distance, averaged across all {matrix_type}.')
            plt.xlabel('Distance in days between RSM Sessions')
            plt.ylabel('Pearson Correlation')
            plt.legend()
            plt.xticks(x_values) 
            plt.grid(True)

            save_folder = f"data_files/{self.lock_event}/visualizations/simulation_scenes/Distance_Plots/RSM_between_session_corrs/{comparison_level}/subject_{self.subject_id}/{normalization}"  
            save_file = f"all_{matrix_type}_avg_session_corrs_by_distance.png"
            self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)
            plt.close()


    def calc_drift_corr_with_fit_measures_by_distances(self, fit_measures_by_distances:dict):
        """
        Calculates the drift correlation (corr between distance and fit measures). 
        Expects dict with fit measures in format dict["session_mapping"][session_train_id]["session_pred"][session_pred_id]
        """
        x_values, y_values = self.extract_x_y_arrays(fit_measures_by_distances, losses_averaged_within_distances=False)
        x_values_set = np.array(list(set(x_values)))  # Required/Useful for non-averaged fit measures within distances: x values occur multiple times
        slope, intercept, r_value, p_value, std_err = linregress(x=x_values, y=y_values)
        r_value = "{:.3f}".format(r_value)  # limit to three decimals

        return r_value

    
    def extract_x_y_arrays(self, fit_measures_by_distances:dict, losses_averaged_within_distances:bool):
        """
        Helper function to extract two arrays 'x_values' and 'y_values' from fit_measures_by_distances
        """
        # Sort by distance for readable plot
        fit_measures_by_distances = {distance: fit_measures_by_distances[distance] for distance in sorted(fit_measures_by_distances, key=int)}

        if losses_averaged_within_distances:
            x_values = np.array(list(fit_measures_by_distances.keys())).astype(int)
            y_values = np.array([data_by_distance["fit_measure"] for data_by_distance in fit_measures_by_distances.values()])

            num_measures_values = np.array([data_by_distance["num_measures"] for data_by_distance in fit_measures_by_distances.values()])
        else:
            x_values = []
            y_values = []
            for distance, fit_measure_list in fit_measures_by_distances.items():
                for _ in range(len(fit_measure_list)):
                    x_values.append(int(distance))
                y_values.extend(fit_measure_list)
            x_values = np.array(x_values)
            y_values = np.array(y_values)
        
        return x_values, y_values

    
    def _plot_scattered_fit_measures_by_distance_with_trend(self, fit_measures_by_distances:dict, ax1, color:str, include_0_distance:bool, timepoint_window_start_idx:str = None, plot_permutation_test:bool=False, curr_norm:str=None, subject_id:str = None):
        """
        Helper function that insert scattered fit measures (mostly variance explained) over distances into a given plot
        """
        x_values, y_values = self.extract_x_y_arrays(fit_measures_by_distances, losses_averaged_within_distances=False)

        # Calculate trend line 
        if include_0_distance:
            # Don't include the self predictions in the trend line calculation (i.e. filter x and y values where x = 0)
            filtered_tuple_values = [x_y_pair for x_y_pair in zip(x_values, y_values) if x_y_pair[0] != 0]
            filtered_x_values, filtered_y_values = zip(*filtered_tuple_values)
            filtered_x_values = list(filtered_x_values)
            filtered_y_values = list(filtered_y_values)
        else:
            filtered_x_values = x_values
            filtered_y_values = y_values

        filtered_x_values_set = np.array(list(set(filtered_x_values)))  # Required/Useful for non-averaged fit measures within distances: x values occur multiple times
        slope, intercept, r_value, p_value, std_err = linregress(x=filtered_x_values, y=filtered_y_values)
        trend_line = slope * filtered_x_values_set + intercept
        r_value = "{:.3f}".format(r_value)  # limit to three decimals

        if not plot_permutation_test:
            permutation_p_value = self.perform_permutation_test(filtered_x_values, filtered_y_values, n_permutations=10000, generate_plot=plot_permutation_test)
        else:
            assert curr_norm is not None, "Atemmpting to store permutation test but norm is not specified."
            # Also store a plot for the permutation test and the null distribution
            permutation_p_value, permutation_fig, permutation_ax = self.perform_permutation_test(filtered_x_values, filtered_y_values, n_permutations=10000, generate_plot=plot_permutation_test)
            permutation_ax.set_title(f'Permutation Test for Subject {self.subject_id} \n r={r_value}, p={permutation_p_value}')
            permutation_plot_folder = f"data_files/{self.lock_event}/visualizations/only_distance/timepoint_windows/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{curr_norm}"
            
            

        if timepoint_window_start_idx not in [None, 999]:
            # A timepoint window was specified (regular case)
            timepoint_start_ms = self.map_timepoint_idx_to_ms(timepoint_window_start_idx)
            timepoint_end_ms = timepoint_start_ms + ((self.time_window_n_indices - 1) * 2)  # each timepoint is equivalent to 2ms, 5 timepoints result in an 8ms range
            label = f"{timepoint_start_ms} to {timepoint_end_ms}ms, (r={r_value}, p={permutation_p_value})" 

            if plot_permutation_test:
                permutation_filename = f"permutation_test_mswindow_{timepoint_start_ms}_{timepoint_end_ms}.png"
                self.save_plot_as_file(plt=permutation_fig, plot_folder=permutation_plot_folder, plot_file=permutation_filename)

        elif timepoint_window_start_idx == 999:
            # In this case we are plotting all timepoints for one subject
            min_index = 0  # 0 because timepoint_min is added in the mapping
            max_index = self.timepoint_max - self.timepoint_min
            #logger.custom_info(f"max_index: {max_index}")
            #label = f"{self.map_timepoint_idx_to_ms(min_index)} to {self.map_timepoint_idx_to_ms(max_index)}ms, (r={r_value}, p={permutation_p_value})"   
            label = f"r={r_value}, p={permutation_p_value}"

            if plot_permutation_test:
                permutation_filename = f"permutation_test_all_timepoints.png"
                self.save_plot_as_file(plt=permutation_fig, plot_folder=permutation_plot_folder, plot_file=permutation_filename)
        elif subject_id != None:
            # Part of a plot for all participants (timepoints not mentioned)
            label = f"Subject {subject_id}: r={r_value}, p={permutation_p_value}"

            if plot_permutation_test:
                raise NotImplementedError("Invalid parameter configuration.")
        else:
            label = None

            if plot_permutation_test:
                raise NotImplementedError("Invalid parameter configuration.")

        if subject_id != "All Subjects":
            ax1.plot(x_values, y_values, color=color, marker='o', linestyle='none', markersize=4)

        # If only a single window is plotted, we want different colors for scatter points and trend. If all windows or all subjects are plotted this gets too confusing. In this the first case, 'color' will be an array
        trend_color = color if not isinstance(color,str) or subject_id == "All Subjects" else 'green' 
        ax1.plot(filtered_x_values_set, trend_line, color=trend_color, label=label, linestyle='-', linewidth=3)

    
    def _plot_drift_distance_based(self, fit_measures_by_distances:dict, self_pred_normalized:bool, losses_averaged_within_distances:bool, all_windows_one_plot:bool, timepoint_window_start_idx:int = None, include_0_distance:bool = False, distance_type:str="days", curr_norm:str=None):
        """
        Creates drift plot based on fit measures data by distance. Expects keys of distance in days as string number, each key containing keys "fit_measure" and "num_measures"
        """
        plot_permutation_test = True if curr_norm is not None else False  # Permutation plot should only be generated if norm is given

        # Plot loss as a function of distance of predicted session from "training" session
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Insert scatter values based on distance into plot
        if not all_windows_one_plot:
            self._plot_scattered_fit_measures_by_distance_with_trend(fit_measures_by_distances, ax1, color='C0', include_0_distance=include_0_distance, timepoint_window_start_idx=timepoint_window_start_idx, plot_permutation_test=plot_permutation_test, curr_norm=curr_norm)  # using default blue color
        else:
            # Create colormap to differentiate time-windows
            n_windows = len(fit_measures_by_distances["timewindow_start"].keys())
            colormap = cm.viridis  
            colors = colormap(np.linspace(0, 1, num=n_windows))

            for timepoint_idx, timepoint_window_start_idx in enumerate(fit_measures_by_distances["timewindow_start"]):
                fit_measures_by_distances_window = fit_measures_by_distances["timewindow_start"][timepoint_window_start_idx]["fit_measures_by_distances"]
                color = colors[timepoint_idx]
                self._plot_scattered_fit_measures_by_distance_with_trend(fit_measures_by_distances_window, ax1, color=color, include_0_distance=include_0_distance, timepoint_window_start_idx=timepoint_window_start_idx, plot_permutation_test=plot_permutation_test, curr_norm=curr_norm) 

        ax1.set_xlabel(f'Distance in {distance_type} between "train" and "test" Session')
        ax1.set_ylabel(f'Variance Explained')
        #plt.legend(loc="upper right")  # , bbox_to_anchor=(1.5, 1))
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)

        lines, labels = ax1.get_legend_handles_labels()
        if losses_averaged_within_distances:
            # Add secondary y-axis for num of datapoints in each average
            ax2 = ax1.twinx()
            ax2.plot(x_values, num_measures_values, 'r--', label='Number of datapoints averaged')
            ax2.set_ylabel('Number of Datapoints', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            # Create legend labels for secondary axis aswell
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines = lines + lines2
            labels = labels + labels2

        ax1.legend(lines, labels, loc='upper right')
        ax1.set_title(f'Averaged Normalized Variance Explained vs Distance in {distance_type}. \n Self-Pred Normalized?: {self_pred_normalized} \n, omitted_sessions: {self.omit_sessions}, {date.today()}')
        plt.grid(True)
        
        # plt.show required?
        return fig

    def visualize_self_prediction(self, fit_measure_type:str, pred_splits:list=["train","test"], all_sessions_combined:bool=False, all_subjects:bool=False):
        assert fit_measure_type in ["var_explained_timepoint", "pearson_r_timepoint"]
        if all_subjects:
            assert fit_measure_type == "var_explained_timepoint" and pred_splits == ["test"]

        if fit_measure_type == "var_explained_timepoint":
            fit_measure_name = "Variance Explained"
        else:
            fit_measure_name = "Pearson correlation"

        color_and_marker_by_split = {"train": {"markertype": '*', "color": '#1f77b4'}, "test": {"markertype": 'o', "color": '#FF8C00'}}
        
        if not all_sessions_combined:
            if all_subjects:
                # Generate a single plot, combining the results of all subjects
                all_subject_ids = ["01", "02", "03", "04", "05"]
                colormap = cm.viridis  
                colors = colormap(np.linspace(0, 1, num=len(all_subject_ids)))
                for normalization in self.normalizations:
                    plt.figure(figsize=(10, 6))
                    for subject_idx, subject_id in enumerate(["01", "02", "03", "04", "05"]):
                        self_pred_measures = {"sessions": {}}
                        storage_folder = f"data_files/{self.lock_event}/var_explained_timepoints/{self.ann_model}/{self.module_name}/subject_{subject_id}/norm_{normalization}/"
                        json_storage_file = f"var_explained_timepoints_dict.json"
                        json_storage_path = os.path.join(storage_folder, json_storage_file)
                        with open(json_storage_path, 'r') as file:
                            session_fit_measures_timepoints = json.load(file)
                        session_fit_measures = self.average_timepoint_data_per_session(session_fit_measures_timepoints)

                        for session_id in session_fit_measures['session_mapping']:
                            fit_measure = session_fit_measures['session_mapping'][session_id]['session_pred'][session_id]
                            self_pred_measures["sessions"][session_id] = fit_measure
                        mean_across_sessions = np.mean(list(self_pred_measures["sessions"].values()))
                        
                        # Add values of current subject to the plot
                        color = colors[subject_idx]
                        plt.plot(self_pred_measures["sessions"].keys(), self_pred_measures["sessions"].values(), color=color, label=f'Subject {subject_id}', linewidth=3)
                        plt.axhline(mean_across_sessions, color=color, linestyle='--')

                    plt.xlabel('Session ID')
                    plt.ylabel('Variance Explained')
                    plt.grid(True)
                    plt.legend(loc='upper right')

                    plt.title(f'Encoding performance for all sessions and subjects, averaged across timepoints. \n Norm {normalization}, {date.today()}')
                    plt.grid(True)

                    # Save the plot to a file
                    plot_folder = f"data_files/{self.lock_event}/visualizations/encoding_performance/all_subjects/norm_{normalization}"
                    plot_file = f"var_explained_encoding_all_subjects_all_sessions_{normalization}.png"
                    self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
            else:
                for normalization in self.normalizations:
                    self_pred_measures = {}
                    for pred_split in pred_splits:
                        predict_train_data = True if pred_split == "train" else False
                        self_pred_measures[pred_split] = {"sessions": {}}
                        storage_folder = f"data_files/{self.lock_event}/{fit_measure_type}s/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                        json_storage_file = f"{fit_measure_type}s_dict.json"
                        json_storage_path = os.path.join(storage_folder, json_storage_file)

                        with open(json_storage_path, 'r') as file:
                            session_fit_measures_timepoints = json.load(file)

                        #print(f"session_fit_measures_timepoints['session_mapping']['2']['session_pred']['2'].values(): {session_fit_measures_timepoints['session_mapping']['2']['session_pred']['2'].values()}")

                        session_fit_measures = self.average_timepoint_data_per_session(session_fit_measures_timepoints)

                        for session_id in session_fit_measures['session_mapping']:
                            fit_measure = session_fit_measures['session_mapping'][session_id]['session_pred'][session_id]
                            self_pred_measures[pred_split]["sessions"][session_id] = fit_measure
                    

                    # Plot values for test prediction
                    plt.figure(figsize=(10, 6))
                    for pred_split in self_pred_measures:
                        markertype = color_and_marker_by_split[pred_split]["markertype"]
                        color = color_and_marker_by_split[pred_split]["color"]
                        plt.plot(self_pred_measures[pred_split]["sessions"].keys(), self_pred_measures[pred_split]["sessions"].values(), marker=markertype, color=color, label=f'{pred_split} pred')
                
                    plt.xlabel(f'Number of Sesssion')
                    plt.ylabel(f'{fit_measure_name}')
                    plt.grid(True)
                    plt.legend(loc='upper right')

                    plt.title(f'{fit_measure_name} Session Self-prediction with Norm {normalization}, {date.today()}')
                    plt.grid(True)
                    #plt.show()

                    # Save the plot to a file
                    plot_folder = f"data_files/{self.lock_event}/visualizations/encoding_performance/subject_{self.subject_id}/norm_{normalization}"
                    pred_split_addition = pred_splits[0] if len(pred_splits) == 1 else "both"
                    plot_file = f"{fit_measure_name}_session_self_prediction_{normalization}_pred_splits_{pred_split_addition}.png"
                    self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
                    #plt.close()

                    logger.custom_debug(f"self_pred_measures: {self_pred_measures}")
        else:
            for normalization in self.normalizations:
                self_pred_measures = {}
                for pred_split in pred_splits:
                    self_pred_measures[pred_split] = {}
                for pred_type in self_pred_measures:
                    predict_train_data = True if pred_type == "train" else False
                    # Read fit measure combined over all sessions
                    storage_folder = f"data_files/{self.lock_event}/{type_of_content}/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/all_sessions_combined/norm_{normalization}/predict_train_data_{predict_train_data}"
                    json_storage_file = f"{type_of_content}_all_sessions_combined_dict.json"
                    json_storage_path = os.path.join(storage_folder, json_storage_file)

                    with open(json_storage_path, 'r') as file:
                        fit_measure = json.load(file)
                    self_pred_measures[pred_type] = {type_of_content: format(fit_measure[type_of_content], '.15f')}   # format to keep decimal notation in logs

                logger.custom_debug(f"self_pred_measures: {self_pred_measures}")

                logger.custom_debug(f"all_sessions_combined: self_pred_measures: {self_pred_measures}")
                
                # Plot values for test prediction
                plt.figure(figsize=(10, 6))
                self_pred_vals = [self_pred_measures[pred_type][type_of_content] for pred_type in self_pred_measures]
                markertype = color_and_marker_by_split[pred_type]["markertype"]
                color = color_and_marker_by_split[pred_type]["color"]
                plt.plot(self_pred_measures.keys(), self_pred_vals, marker=markertype, color=color, label=f'{pred_type} pred')

                plt.xlabel(f'Predicted Datasplit')
                plt.ylabel(f'{fit_measure_name}')
                plt.grid(True)
                plt.legend(loc='upper right')
                plt.title(f'{fit_measure_name} Self-prediction combined over all Sessions with Norm {normalization}, {date.today()}')
                plt.grid(True)
                #plt.show()

                # Save the plot to a file
                plot_folder = f"data_files/{self.lock_event}/visualizations/encoding_performance/subject_{self.subject_id}/all_sessions_combined/norm_{normalization}"
                pred_split_addition = pred_split[0] if len(pred_splits) == 1 else "both"
                plot_file = f"{fit_measure_name}_all_sessions_combined_self_prediction_{normalization}_pred_splits_{pred_split_addition}.png"
                self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
                

    def visualize_GLM_results(self, fit_measure_type:str, by_timepoints:bool=False, only_distance:bool=False, omit_sessions:list=[], separate_plots:bool=False, distance_in_days:bool=True, average_distance_vals:bool=False, regions_of_interest:list=None, source_pca_type:str=None, result_stored_by_pc:bool=False, whiten:bool=False):
        """
        Visualizes results from GLMHelper.predict_from_mapping_all_sessions
        """
        pc_storage_folder = "/pcs_separate" if result_stored_by_pc else ""
        whiten_folder = "/whiten" if whiten else ""
        if source_pca_type == 'voxels':
            pca_folder = f"/voxels_pca_reduced{whiten_folder}"
        else:
            pca_folder = ""
        if regions_of_interest is not None:
            assert by_timepoints, "[visualize_GLM_results] currently only used for self-prediction timepoint comparisons in the context of source space analysis."

        session_day_differences = self.get_session_date_differences()

        # TODO: Remove reprecated var_explained parameter; by_timepoints can now also be var_explained

        if fit_measure_type in ["var_explained_timepoint", "var_explained", "var_explained_sensors_timepoint"]:
            type_of_fit_measure = "Variance Explained"
        elif fit_measure_type == "pearson_r_timepoint":
            type_of_fit_measure = "Pearson correlation"
        else:
            type_of_fit_measure = "MSE"
        
        fit_measure_norms = {}
        for normalization in self.normalizations:
            if regions_of_interest is None:
                # Load loss/var explained dict
                if fit_measure_type not in ["var_explained_timepoint", "var_explained_sensors_timepoint", "pearson_r_timepoint"]:
                    session_fit_measures = self.read_dict_from_json(type_of_content=fit_measure_type, type_of_norm=normalization)
                else:
                    storage_folder = f"data_files/{self.lock_event}/{fit_measure_type}s/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                    json_storage_file = f"{fit_measure_type}s_dict.json"
                    json_storage_path = os.path.join(storage_folder, json_storage_file)

                    with open(json_storage_path, 'r') as file:
                        session_fit_measures = json.load(file)

                fit_measure_norms[normalization] = session_fit_measures
                

            if only_distance:
                # Plot loss as a function of distance of predicted session from "training" session
                fig, ax1 = plt.subplots(figsize=(12, 8))

                # Iterate over each training session
                losses_by_distances = {}
                for train_session, data in session_fit_measures['session_mapping'].items():
                    if train_session in omit_sessions:
                        continue
                    # Calculate distance and collect corresponding losses
                    for pred_session, fit_measure in data['session_pred'].items():
                        if pred_session in omit_sessions:
                            continue
                        if train_session != pred_session:
                            if not distance_in_days:
                                distance = abs(int(train_session) - int(pred_session))
                            else:
                                distance = session_day_differences[train_session][pred_session]
                            if average_distance_vals:
                                if distance not in losses_by_distances:
                                    losses_by_distances[distance] = {"fit_measure": fit_measure, "num_measures": 1}
                                else:
                                    losses_by_distances[distance]["fit_measure"] += fit_measure
                                    losses_by_distances[distance]["num_measures"] += 1
                            else:
                                if distance not in losses_by_distances:
                                    losses_by_distances[distance] = [fit_measure]
                                else:
                                    losses_by_distances[distance].append(fit_measure)
                                

                if average_distance_vals:
                    # Calculate average losses over distances
                    avg_losses = {}
                    num_datapoints = {}

                    for distance in losses_by_distances:
                        avg_losses[distance] = losses_by_distances[distance]["fit_measure"] / losses_by_distances[distance]["num_measures"]
                        num_datapoints[distance] = losses_by_distances[distance]["num_measures"]

                    # Sort by distance for plot lines
                    avg_losses = dict(sorted(avg_losses.items()))
                    num_datapoints = dict(sorted(num_datapoints.items()))

                    x_values = np.array(list(avg_losses.keys()))
                    y_values = np.array(list(avg_losses.values()))
                else:
                    # Extract values, sorted by x
                    sorted_losses_by_distances = {distance: losses_by_distances[distance] for distance in sorted(losses_by_distances, key=int)}
                    x_values = []
                    y_values = []
                    for distance in sorted_losses_by_distances:
                        for fit_measure in sorted_losses_by_distances[distance]:
                            x_values.append(distance)
                            y_values.append(fit_measure)
                    x_values = np.array(x_values)
                    y_values = np.array(y_values)
                            
                # Calculate trend line 
                x_values_set = np.array(list(set(x_values)))
                slope, intercept, r_value, p_value, std_err = linregress(x=x_values, y=y_values)
                trend_line = slope * x_values_set + intercept
                r_value = "{:.3f}".format(r_value)  # limit to three decimals

                # Plot
                title_addition = "in days" if distance_in_days else ""
                ax1.plot(x_values, y_values, marker='o', linestyle='none', label=f'Average {type_of_fit_measure}')
                ax1.plot(x_values_set, trend_line, color='green', linestyle='-', label=f'Trend line (r={r_value})', linewidth=3)
                ax1.set_xlabel(f'Distance {title_addition} between "train" and "test" Session')
                ax1.set_ylabel(f'{type_of_fit_measure}')
                ax1.tick_params(axis='y', labelcolor='b')
                ax1.grid(True)
        
                lines, labels = ax1.get_legend_handles_labels()
                if average_distance_vals:
                    # Add secondary y-axis for datapoints
                    ax2 = ax1.twinx()
                    ax2.plot(num_datapoints.keys(), num_datapoints.values(), 'r--', label='Number of datapoints averaged')
                    ax2.set_ylabel('Number of Datapoints', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')

                    # Create legend labels for secondary axis aswell
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    lines = lines + lines2
                    labels = labels + labels2

                ax1.legend(lines, labels, loc='upper right')
                ax1.set_title(f'{type_of_fit_measure} vs Distance for Predictions Averaged Across all Sessions. \n Norm {normalization}, sessions omitted: {omit_sessions}, {date.today()}')
                plt.grid(True)
                #plt.show()

                # Save the plot to a file
                plot_folder = f"data_files/{self.lock_event}/visualizations/only_distance/subject_{self.subject_id}/norm_{normalization}"
                plot_file = f"{type_of_fit_measure}_plot_over_distance_{normalization}.png"
                self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
                #plt.close()

            elif not by_timepoints:
                # Collect self-prediction {type_of_fit_measure}s for baseline and prepare average non-self-{type_of_fit_measure} calculation
                self_prediction_fit_measures = {}
                average_prediction_fit_measures = {}
                for session in session_fit_measures['session_mapping']:
                    self_prediction_fit_measures[session] = session_fit_measures['session_mapping'][session]['session_pred'][session]
                    average_prediction_fit_measures[session] = 0

                # Calculate average {type_of_fit_measure} for each session from all other training sessions
                for train_session, data in session_fit_measures['session_mapping'].items():
                    for pred_session, fit_measure in data['session_pred'].items():
                        if train_session != pred_session:
                            average_prediction_fit_measures[pred_session] += fit_measure
                num_other_sessions = len(self.session_ids_num) - 1
                for session in average_prediction_fit_measures:
                    average_prediction_fit_measures[session] /= num_other_sessions

                if separate_plots:
                    # Generate separate plots for each training session
                    for train_session, data in session_fit_measures['session_mapping'].items():
                        plt.figure(figsize=(10, 6))
                        sessions = list(data['session_pred'].keys())
                        fit_measures = list(data['session_pred'].values())
                        plt.plot(sessions, fit_measures, marker='o', linestyle='-', label=f'Training Session {train_session}')
                        plt.plot(self_prediction_fit_measures.keys(), self_prediction_fit_measures.values(), 'r--', label=f'Self-prediction {type_of_fit_measure}')
                        plt.plot(average_prediction_fit_measures.keys(), average_prediction_fit_measures.values(), 'g-.', label=f'Average Non-self-prediction {type_of_fit_measure}')

                        plt.title(f'{type_of_fit_measure} for Predictions from Training Session {train_session}')
                        plt.xlabel('Predicted Session')
                        plt.ylabel(f'{type_of_fit_measure}')
                        plt.legend()
                        plt.grid(True)

                        # Save the plot to a file
                        plot_folder = f"data_files/{self.lock_event}/visualizations/seperate_plots_{separate_plots}/subject_{self.subject_id}/norm_{normalization}"
                        plot_file = f"{type_of_fit_measure}_plot_session{train_session}.png"
                        self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
                else:
                    # Generate a single plot with all training sessions
                    plt.figure(figsize=(12, 8))
                    for train_session, data in session_fit_measures['session_mapping'].items():
                        sessions = list(data['session_pred'].keys())
                        fit_measures = list(data['session_pred'].values())
                        plt.plot(sessions, fit_measures, marker='o', linestyle='-', label=f'Trained on Session {train_session}')
                    plt.plot(self_prediction_fit_measures.keys(), self_prediction_fit_measures.values(), 'r--', label=f'Self-prediction {type_of_fit_measure}')
                    plt.plot(average_prediction_fit_measures.keys(), average_prediction_fit_measures.values(), 'g-.', label=f'Average Non-self-prediction {type_of_fit_measure}')
                    
                    plt.title(f'{type_of_fit_measure} for Predictions Across All Sessions')
                    plt.xlabel('Prediction Session')
                    plt.ylabel(f'{type_of_fit_measure}')
                    plt.legend()
                    plt.grid(True)

                    # Save the plot to a file
                    plot_folder = f"data_files/{self.lock_event}/visualizations/seperate_plots_{separate_plots}/subject_{self.subject_id}/norm_{normalization}"
                    plot_file = f"MSE_plot_all_sessions.png"
                    self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
            elif by_timepoints:
                assert fit_measure_type in ["var_explained_timepoint", "var_explained_sensors_timepoint", "pearson_r_timepoint"]

                def plot_timepoint_fit_measure(timepoint_loss_list:list, num_timepoints:int, session_id:str=None, sensor_name:str=None, glaser_region:str=None, pca_folder:str="", pc_num:str=None):
                    plt.figure(figsize=(10, 6))
                    timepoints_in_ms = [self.map_timepoint_idx_to_ms(timepoint_idx) for timepoint_idx in list(range(num_timepoints))]
                    #plt.bar(timepoints_in_ms, timepoint_loss_list, color='blue')
                    #timepoints_indices = [timepoint_idx for timepoint_idx in list(range(num_timepoints))]
                    plt.bar(timepoints_in_ms, timepoint_loss_list, color='blue', width=1.6)
                    #plt.bar(list(range(num_timepoints)), timepoint_loss_list, color='blue')
                    #logger.custom_info(f"list(range(num_timepoints): {list(range(num_timepoints))}")
                    session_subtitle = "Averaged across all Sessions, predicting themselves" if session_id is None else f"Session {session_id}, predicting iteself"
                    sensor_subtitle = "Averaged across all sensors" if sensor_name is None else f"Sensor {sensor_name}"
                    region_subtitle = "" if glaser_region is None else f"Glaser region {glaser_region}"
                    plt.title(f'{type_of_fit_measure} per Timepoint Model. \n {region_subtitle} {sensor_subtitle} {session_subtitle}.')
                    plt.xlabel(f'Time relative to {self.lock_event} onset in ms')
                    plt.ylabel(f'{type_of_fit_measure}')
                    plt.grid(True)

                    # Save the plot to a file
                    sensor_folder_addition = f"/sensor_level/{sensor_name}" if sensor_name is not None else ""
                    pc_storage_folder = "/pcs_separate" if pc_num is not None else ""
                    region_folder_addition = f"/source_space{pca_folder}{pc_storage_folder}/{glaser_region}" if glaser_region is not None else ""
                    plot_folder = f"data_files/{self.lock_event}/visualizations/timepoint_model_comparison/{self.ann_model}/{self.module_name}{region_folder_addition}{sensor_folder_addition}/subject_{self.subject_id}/norm_{normalization}"
                    if session_id is None:
                        plot_folder += "/all_sessions_combined"  
                        session_name_addition = "" if sensor_name is None else f"_sensor_{sensor_name}"
                    else:
                        plot_folder += f""
                        session_name_addition = f"_session{session_id}" if sensor_name is None else f"_sensor_{sensor_name}_session{session_id}"
                    pc_name_addition = f"pc_{pc_num}_" if pc_num is not None else ""
                    plot_file = f"{pc_name_addition}{fit_measure_type}_comparison_{normalization}{session_name_addition}.png"
                    self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
                    plt.close()

                def extract_self_session_pred_and_plot_by_timepoints(session_fit_measures:dict, sensor_name:str=None, glaser_region:str=None, pca_folder:str="", pc_num:str=None) -> None:
                    # Collect fit_measures for timepoint models on predictions on the own session
                    fit_measures_by_session_by_timepoint = {"session": {}}
                    for session_id in session_fit_measures['session_mapping']:
                        fit_measures_by_session_by_timepoint["session"][session_id] = {"timepoint":{}}
                        for timepoint, fit_measure_timepoint in session_fit_measures['session_mapping'][session_id]["session_pred"][session_id]["timepoint"].items():
                            fit_measures_by_session_by_timepoint["session"][session_id]["timepoint"][timepoint] = fit_measure_timepoint

                    num_timepoints = len([timepoint for timepoint in fit_measures_by_session_by_timepoint["session"]["3"]["timepoint"]])

                    logger.custom_info(f"num_timepoints: {num_timepoints}")

                    timepoint_average_fit_measure = {}
                    if not separate_plots:
                        # Plot results averaged over all sessions
                        for timepoint in range(num_timepoints):
                            # Calculate average over all within session predictions for this timepoint
                            fit_measures = []
                            for session in fit_measures_by_session_by_timepoint["session"]:
                                timepoint_session_loss = fit_measures_by_session_by_timepoint["session"][session]["timepoint"][str(timepoint)]
                                fit_measures.append(timepoint_session_loss)
                            avg_loss = np.sum(fit_measures) / len(fit_measures)
                            timepoint_average_fit_measure[timepoint] = avg_loss
                        timepoint_avg_loss_list = [timepoint_average_fit_measure[timepoint] for timepoint in timepoint_average_fit_measure]

                        plot_timepoint_fit_measure(timepoint_loss_list=timepoint_avg_loss_list, num_timepoints=num_timepoints, sensor_name=sensor_name, glaser_region=glaser_region, pca_folder=pca_folder, pc_num=pc_num)
                    else:
                        for session_id in fit_measures_by_session_by_timepoint["session"]:
                            timepoint_loss_list = [fit_measures_by_session_by_timepoint["session"][session_id]["timepoint"][timepoint] for timepoint in fit_measures_by_session_by_timepoint["session"][session_id]["timepoint"]]
                            plot_timepoint_fit_measure(timepoint_loss_list=timepoint_loss_list, num_timepoints=num_timepoints, session_id=session_id, sensor_name=sensor_name, glaser_region=glaser_region, pca_folder=pca_folder, pc_num=pc_num)
            
                if fit_measure_type != "var_explained_sensors_timepoint":
                    # All sensors combined
                    if regions_of_interest is None:
                        extract_self_session_pred_and_plot_by_timepoints(session_fit_measures=session_fit_measures)
                    else:
                        # Load timepoint-based variance explained and plot seperately for each glaser region of interest
                        for glaser_region in regions_of_interest:
                            storage_folder = f"data_files/{self.lock_event}/var_explained_timepoints/source_space{pca_folder}{pc_storage_folder}/{glaser_region}/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                            json_storage_file = f"var_explained_timepoints_dict.json"
                            json_storage_path = os.path.join(storage_folder, json_storage_file)
                            with open(json_storage_path, 'r') as file:
                                fit_measures_by_session_by_timepoint = json.load(file)
                            
                            if not result_stored_by_pc:
                                extract_self_session_pred_and_plot_by_timepoints(session_fit_measures=fit_measures_by_session_by_timepoint, glaser_region=glaser_region, pca_folder=pca_folder)
                            else:
                                for pc, pc_fit_measures_by_session_by_timepoint in fit_measures_by_session_by_timepoint["pcs"].items():
                                    extract_self_session_pred_and_plot_by_timepoints(session_fit_measures=pc_fit_measures_by_session_by_timepoint, glaser_region=glaser_region, pca_folder=pca_folder, pc_num=pc)
                else:
                    # Each sensor seperate
                    sensor_index_name_dict = self.get_relevant_meg_channels(self.chosen_channels)
                    if sensor_index_name_dict['grad']:
                        raise NotImplementedError("Plot not yet implemented for grad sensors")
                    sensor_names = [sensor_name for sensor_name in sensor_index_name_dict['mag']['sensor_index_within_type'].values()]

                    # TODO: Store and retrieve sensors by name out of dict instead of by index? Seems to be safest
                    # TODO: I adjusted the sensor timepoint dict so that the sensor is the outmost attribute. Assure that this does not cause issues elsewhere
                    
                    for sensor_idx, sensor_session_fit_measures in session_fit_measures["sensor"].items():
                        sensor_name = sensor_names[int(sensor_idx)]
                        extract_self_session_pred_and_plot_by_timepoints(session_fit_measures=sensor_session_fit_measures, sensor_name=sensor_name)

            else:
                raise ValueError("[ERROR][visualize_GLM_results] Function called with invalid parameter configuration.")

        if not by_timepoints:
            fit_measures_new = {}
            for norm, fit_measure in fit_measure_norms.items():
                if fit_measure in fit_measures_new.values():
                    # get key
                    for key, value in fit_measures_new.items():
                        if value == fit_measure:
                            norm_new = key
                    logger.warning(f"Same loss for norm {norm} and norm {norm_new}")  # raise ValueError()

                fit_measures_new[norm] = fit_measure

    def three_dim_timepoint_predictions(self, subtract_self_pred:bool):
        """
        Creates a 3D plot. Every singular position on the third axis is similar to the 'by_timepoints' plot in visualize_GLM_results.
        """
        def plot_timepoint_fit_measure_3d(fit_measures_by_session_by_timepoint, session_train_id):
            num_sessions = len(fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"])

            """
            timepoint_10 = fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"][session_train_id]["timepoint"][str(10)]
            timepoint_50 = fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"][session_train_id]["timepoint"][str(50)]
            logger.custom_info(f"timepoint_10: {timepoint_10}")
            logger.custom_info(f"timepoint_50: {timepoint_50}")
            """
        
            colormap = cm.viridis  
            colors = colormap(np.linspace(0, 1, num_sessions))

            #logger.custom_info(f"colors: {colors}")

            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(111, projection='3d')

            min_z = None
            for session_id_pred in fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"]:
                session_pred_data = fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"][session_id_pred]
                num_timepoints = len(session_pred_data["timepoint"])

                x = np.full(num_timepoints, session_id_pred, dtype=int)
                y = np.arange(0, num_timepoints)
                z = np.array([session_pred_data["timepoint"][str(timepoint)] for timepoint in session_pred_data["timepoint"]])

                if min_z is None or np.any(min(z) < min_z):
                    min_z = min(z)

                color = colors[int(session_id_pred)-1] if session_train_id != session_id_pred else "red"
                ax.plot(x, y, z, label=f'Session {session_id_pred}', linewidth=2.5, color=color)

                # Create vertices for the polygon
                # Add a zero line at the bottom
                x2 = np.append(x, x[::-1])
                y2 = np.append(y, y[::-1])
                z2 = np.append(z, np.zeros_like(z))
                verts = [list(zip(x2, y2, z2))]

                #verts = [list(zip(x, y, z))]

                # Create the polygon and add it to the plot
                poly = Poly3DCollection(verts, facecolor=color, alpha=0.3)
                ax.add_collection3d(poly)
            
            
            for session_id_pred in fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"]:
                session_pred_data = fit_measures_by_session_by_timepoint['session_mapping'][session_train_id]["session_pred"][session_id_pred]
                num_timepoints = len(session_pred_data["timepoint"])

                x = np.full(num_timepoints, session_id_pred, dtype=int)
                y = np.arange(0, num_timepoints)
                z_control = np.array([0 for timepoint in range(num_timepoints)])

                color = colors[int(session_id_pred)-1] if session_train_id != session_id_pred else "red"
                ax.plot(x, y, z_control, linewidth=2.5, color=color)
            

            ax.set_xlabel('X: Predicted Sessions')
            ax.set_ylabel('Y: Timepoints')
            ax.set_zlabel('Z: Variance Explained')
            ax.set_title(f'Variance Explained for Each Session and Timepoint for Model trained on Session {session_train_id}')
            plt.legend()

            # Set x-axis (session) ticks and labels
            ax.set_xticks(np.arange(1, num_sessions+1))
            ax.set_xticklabels([str(i) for i in range(1, num_sessions+1)])

            ax.set_yticks(np.arange(0, num_timepoints+1, 10))
            #logger.custom_info(f"yticks: {np.arange(0, num_timepoints+1, 10)}")
            

            #plt.show()
            
            return fig

        for normalization in self.normalizations:
            storage_folder = f"data_files/{self.lock_event}/var_explained_timepoints/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
            json_storage_file = f"var_explained_timepoints_dict.json"
            json_storage_path = os.path.join(storage_folder, json_storage_file)
            with open(json_storage_path, 'r') as file:
                fit_measures_by_session_by_timepoint = json.load(file)

            if subtract_self_pred:
                # Normalize with self-predictions
                fit_measures_by_session_by_timepoint = self.normalize_cross_session_preds_with_self_preds(fit_measures_by_session_by_timepoint=fit_measures_by_session_by_timepoint)

                # Also store new distance based measures
                fit_measures_by_distances = self.calculate_fit_by_distances(fit_measures_by_session=fit_measures_by_session_by_timepoint, timepoint_level_input=True)
                
                json_storage_folder = f"data_files/{self.lock_event}/var_explained/by_distance/self_pred_normalized/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                json_storage_file = f"fit_measures_by_distances_self_pred_normalized_dict.json"
                json_storage_path = os.path.join(json_storage_folder, json_storage_file)
                os.makedirs(json_storage_folder, exist_ok=True)

                with open(json_storage_path, 'w') as file:
                    # Serialize and save the dictionary to the file
                    json.dump(fit_measures_by_distances, file, indent=4)

            for session_id in self.session_ids_num:
                timepoints_sessions_plot = plot_timepoint_fit_measure_3d(fit_measures_by_session_by_timepoint, session_train_id=session_id)

                plot_folder = f"data_files/{self.lock_event}/visualizations/3D_Plots/cross_session_preds_timepoints/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                plot_file = f"cross_session_preds_timepoints_session_{session_id}.png"
                plot_dest = os.path.join(plot_folder, plot_file)
                
                self.save_plot_as_file(plt=timepoints_sessions_plot, plot_folder=plot_folder, plot_file=plot_file, plot_type="figure")

                # Save with pickle to keep plot interactive
                #with open(plot_dest, 'wb') as file: 
                #    pickle.dump(timepoints_sessions_plot, file)


    def drift_distance_all_subjects(self, subject_ids:list, fit_measure:str, omit_sessions_by_subject:dict, include_0_distance:bool):
        """Visualizes drift graph based on encoding performance by distance for all subjects, As well as average over all subjects."""
        assert fit_measure in ["var_explained", "pearson_r"]

        for normalization in self.normalizations:
            fit_measures_by_distances_all_subjects = {}
            fig, ax1 = plt.subplots(figsize=(12, 8))
            ax1.set_xlabel('Distance in days between "train" and "test" Session')
            ylabel = 'Variance Explained' if fit_measure == "var_explained" else 'Pearson Correlation'
            ax1.set_ylabel(ylabel)
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True)
            colormap_subjects = cm.viridis  
            colors = colormap_subjects(np.linspace(0, 1, num=len(subject_ids)))
            # Collect timepoint cross session fit measures for all considered subjects
            for subject_nr, subject_id in enumerate(subject_ids):
                color_subject = colors[subject_nr]
                if fit_measure == "var_explained":
                    # Load timepoint-based variance explained
                    storage_folder = f"data_files/{self.lock_event}/var_explained_timepoints/{self.ann_model}/{self.module_name}/subject_{subject_id}/norm_{normalization}/"
                    json_storage_file = f"var_explained_timepoints_dict.json"
                else:
                    # Load timepoint-based pearson r
                    storage_folder = f"data_files/{self.lock_event}/pearson_r_timepoints/{self.ann_model}/{self.module_name}/subject_{subject_id}/norm_{normalization}/"
                    json_storage_file = f"pearson_r_timepoints_dict.json"
                json_storage_path = os.path.join(storage_folder, json_storage_file)
                with open(json_storage_path, 'r') as file:
                    fit_measures_by_session_by_timepoint = json.load(file)
                
                # Filter data from sessions that are to be omitted and convert to fit by distances
                fit_measures_by_filtered_session_by_timepoint = self.omit_selected_sessions_from_fit_measures(fit_measures_by_session=fit_measures_by_session_by_timepoint, omitted_sessions=omit_sessions_by_subject[subject_id], sensors_seperated=False)
                fit_measures_by_distances_subject = self.calculate_fit_by_distances(fit_measures_by_session=fit_measures_by_filtered_session_by_timepoint, timepoint_level_input=True, average_within_distances=False, include_0_distance=include_0_distance, subject_id=subject_id)

                # Add a colored line for the subject with their drift data
                self._plot_scattered_fit_measures_by_distance_with_trend(fit_measures_by_distances=fit_measures_by_distances_subject, ax1=ax1, color=color_subject, include_0_distance=include_0_distance, subject_id=subject_id)

                # Plot a histogram capturing the results of the permutation test
                x_values, y_values = self.extract_x_y_arrays(fit_measures_by_distances_subject, losses_averaged_within_distances=False)

                if include_0_distance:
                    # Don't include the self predictions in the trend line calculation (i.e. filter x and y values where x = 0)
                    filtered_tuple_values = [x_y_pair for x_y_pair in zip(x_values, y_values) if x_y_pair[0] != 0]
                    filtered_x_values, filtered_y_values = zip(*filtered_tuple_values)
                    filtered_x_values = list(filtered_x_values)
                    filtered_y_values = list(filtered_y_values)
                else:
                    filtered_x_values = x_values
                    filtered_y_values = y_values

                # Calculate trend line 
                observed_corr = "{:.3f}".format(pearsonr(filtered_x_values,filtered_y_values)[0])

                permutation_p_value, permutation_plot, permutation_ax = self.perform_permutation_test(filtered_x_values, filtered_y_values, n_permutations=10000, generate_plot=True)
                permutation_ax.set_title(f'Permutation Test for Subject {subject_id} \n r={observed_corr}, p={permutation_p_value}')

                plot_folder = f"data_files/{self.lock_event}/visualizations/distance_drift/permutation_test/{fit_measure}/{normalization}"
                plot_file = f"permutation_test_results_{subject_id}.png"
                self.save_plot_as_file(plt=permutation_plot, plot_folder=plot_folder, plot_file=plot_file)

                # Add subject data to distance/drift data for all subjects combined
                for distance in fit_measures_by_distances_subject:
                    if distance not in fit_measures_by_distances_all_subjects.keys():
                        fit_measures_by_distances_all_subjects[distance] = fit_measures_by_distances_subject[distance]
                    else:
                        fit_measures_by_distances_all_subjects[distance].extend(fit_measures_by_distances_subject[distance])

            # Calculate and plot average drift over all considered subjects (based on combined distance scatter values)
            self._plot_scattered_fit_measures_by_distance_with_trend(fit_measures_by_distances=fit_measures_by_distances_all_subjects, ax1=ax1, color='black', include_0_distance=include_0_distance, subject_id='All Subjects')

            lines, labels = ax1.get_legend_handles_labels()
            ax1.legend(lines, labels, loc='upper right')

            plot_folder = f"data_files/{self.lock_event}/visualizations/distance_drift/all_subjects/{normalization}"
            plot_file = f"distance_drift_all_subjects_{fit_measure}.png"
            self.save_plot_as_file(fig, plot_folder=plot_folder, plot_file=plot_file)

    def timepoint_window_drift(self, all_windows_one_plot:bool, subtract_self_pred:bool, sensor_level:bool, include_0_distance:bool, debugging:bool=False, regions_of_interest:list=None, source_pca_type:str=None, whiten:bool=False):
        """
        Plots drift for seperate timepoint windows, as well as for all windows combined.
        """
        if regions_of_interest is not None:
            assert not sensor_level, "timepoint_window_drift was called on source data and sensor level."

        whiten_folder = "/whiten" if whiten else ""
        if source_pca_type == 'voxels':
            pca_folder = f"/voxels_pca_reduced{whiten_folder}"
        elif source_pca_type == 'voxels_and_timepoints':
            raise ValueError("[timepoint_window_drift] does not work for timepoint pca.")
        else:
            pca_folder = ""

        def filter_timepoint_dict_for_window(fit_measures_by_session_by_timepoint: dict, timepoint_window_start_idx:int):
            """
            Returns a copy of the input fit measure dict by timepoints that only contains the timepoints within the selected window
            """
            fit_measures_by_session_by_chosen_timepoints = self.recursive_defaultdict()
            for session_train_id, fit_measures_train_session in fit_measures_by_session_by_timepoint['session_mapping'].items():
                for session_pred_id, fit_measures_pred_session in fit_measures_train_session["session_pred"].items():
                    for timepoint_idx, timepoint_value in fit_measures_pred_session["timepoint"].items():
                        if int(timepoint_idx) >= timepoint_window_start_idx and int(timepoint_idx) < (timepoint_window_start_idx + self.time_window_n_indices):
                            fit_measures_by_session_by_chosen_timepoints['session_mapping'][session_train_id]["session_pred"][session_pred_id]["timepoint"][timepoint_idx] = timepoint_value
                    # Only consider full windows (i.e. cut off potential smaller last timepoint window)
                    current_window_size = len(fit_measures_by_session_by_chosen_timepoints['session_mapping'][session_train_id]["session_pred"][session_pred_id]["timepoint"])
                    if current_window_size < self.time_window_n_indices:
                        return None
                    assert current_window_size == self.time_window_n_indices, f"Time window size is incorrect. current_window_size: {current_window_size}, time_window_n_indices: {self.time_window_n_indices}"
            
            return fit_measures_by_session_by_chosen_timepoints

        def plot_timepoint_window_drift_for_timepoint_fit_measures(fit_measures_by_session_by_timepoint:dict, sensor_name:str=None, glaser_region:str=None, pca_folder:str="", curr_norm:str=None) -> None:
            sensor_filename_addition = f"sensor_{sensor_name}_" if sensor_name is not None else ""
            glaser_region_filename_addition = f"{glaser_region}" if glaser_region is not None else ""
            pca_filename_addition = "voxels_pca_reduced_" if pca_folder.startswith("/voxels_pca_reduced") else ""

            if subtract_self_pred:
                # Normalize with self-predictions
                fit_measures_by_session_by_timepoint = self.normalize_cross_session_preds_with_self_preds(fit_measures_by_session_by_timepoint=fit_measures_by_session_by_timepoint)
            if self.omit_sessions:
                fit_measures_by_session_by_timepoint = self.omit_selected_sessions_from_fit_measures(fit_measures_by_session=fit_measures_by_session_by_timepoint, omitted_sessions=self.omit_sessions, sensors_seperated=False)

            # Define storage folder for all plots in function
            sensor_folder_addition = "sensor_level/" if sensor_level else ""
            source_region_folder_addition = f"source_space{pca_folder}/{glaser_region}/" if glaser_region is not None else ""
            storage_folder = f"data_files/{self.lock_event}/visualizations/only_distance/timepoint_windows/{sensor_folder_addition}{source_region_folder_addition}{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}"
            
            # Calculate drift for various timewindows by slicing 
            if all_windows_one_plot:
                fit_measures_by_distance_by_time_window = {"timewindow_start": {}}

            num_timepoints = (self.timepoint_max - self.timepoint_min) + 1  # +1 because of index 0 ofc
            timepoint_window_start_idx = 0
            while timepoint_window_start_idx < num_timepoints:
                # Filter timepoint values for current window
                fit_measures_window_by_session = filter_timepoint_dict_for_window(fit_measures_by_session_by_timepoint=fit_measures_by_session_by_timepoint,
                                                                                    timepoint_window_start_idx=timepoint_window_start_idx)
                # Only consider full windows (i.e. don't use potential smaller last timepoint window)
                if fit_measures_window_by_session is not None:
                    # Calculate distance based variance explained for current window
                    fit_measures_by_distances_window = self.calculate_fit_by_distances(fit_measures_by_session=fit_measures_window_by_session, timepoint_level_input=True, average_within_distances=False, include_0_distance=include_0_distance)

                    if not all_windows_one_plot:
                        # Plot drift for current window
                        drift_plot_window = self._plot_drift_distance_based(fit_measures_by_distances=fit_measures_by_distances_window, self_pred_normalized=subtract_self_pred, omitted_sessions=omitted_sessions, losses_averaged_within_distances=False, all_windows_one_plot=False, timepoint_window_start_idx=timepoint_window_start_idx, include_0_distance=include_0_distance)

                        # Store plot for current window
                        window_end = timepoint_window_start_idx + self.time_window_n_indices
                        timepoint_window_description = f"window_{timepoint_window_start_idx}-{window_end}"
                        storage_filename = f"drift_plot_{pca_filename_addition}{sensor_filename_addition}{glaser_region_filename_addition}{timepoint_window_description}"
                        self.save_plot_as_file(plt=drift_plot_window, plot_folder=storage_folder, plot_file=storage_filename, plot_type="figure")
                    else:
                        fit_measures_by_distance_by_time_window["timewindow_start"][timepoint_window_start_idx] = {"fit_measures_by_distances": fit_measures_by_distances_window}

                timepoint_window_start_idx += self.time_window_n_indices

            # Plot all timewindows in the same plot (with different colors)
            if all_windows_one_plot:
                drift_plot_all_windows = self._plot_drift_distance_based(fit_measures_by_distances=fit_measures_by_distance_by_time_window, self_pred_normalized=subtract_self_pred, losses_averaged_within_distances=False, all_windows_one_plot=True, include_0_distance=include_0_distance, curr_norm=curr_norm)
                storage_filename = f"drift_plot_{pca_filename_addition}{sensor_filename_addition}all_windows_comparison"
                self.save_plot_as_file(plt=drift_plot_all_windows, plot_folder=storage_folder, plot_file=storage_filename, plot_type="figure")

            # For control/comparison, plot the drift for the all timepoint values combined/averaged aswell
            logger.custom_debug(f"fit_measures_by_session_by_timepoint: {fit_measures_by_session_by_timepoint}")
            fit_measures_by_distances_all_timepoints = self.calculate_fit_by_distances(fit_measures_by_session=fit_measures_by_session_by_timepoint, timepoint_level_input=True, average_within_distances=False, include_0_distance=include_0_distance)
            drift_plot_all_timepoints = self._plot_drift_distance_based(fit_measures_by_distances=fit_measures_by_distances_all_timepoints, self_pred_normalized=subtract_self_pred, losses_averaged_within_distances=False, all_windows_one_plot=False, timepoint_window_start_idx=999, include_0_distance=include_0_distance, curr_norm=curr_norm)  # 999 indicates that we are considering all timepoints

            storage_filename = f"drift_plot_{pca_filename_addition}{sensor_filename_addition}all_timepoints"
            self.save_plot_as_file(plt=drift_plot_all_timepoints, plot_folder=storage_folder, plot_file=storage_filename, plot_type="figure")

            
            def calculate_fit_by_session_distance(fit_measures_by_session_by_timepoint:dict, include_0_distance:bool):
                if self.subject_id == "02":
                    # Adjust for repeated session 04
                    new_session_ids = ["1", "2", "3", "10", "4", "5", "6", "7", "8", "9"]
                    session_mapping = dict(zip(self.session_ids_num, new_session_ids))

                fit_measures_by_session = self.average_timepoint_data_per_session(fit_measures_by_session_by_timepoint)

                # Calculate fit measures relative to distance in time between train and pred session
                fit_by_session_distances = {}
                for session_train_id, fit_measures_train_session in fit_measures_by_session['session_mapping'].items():
                    if self.subject_id == "02":
                        session_train_id = session_mapping[session_train_id]
                    for session_pred_id, fit_measure_pred_session in fit_measures_train_session["session_pred"].items():
                        if self.subject_id == "02":
                            session_pred_id = session_mapping[session_pred_id]
                        if include_0_distance or session_train_id != session_pred_id:
                            session_distance = abs(int(session_train_id) - int(session_pred_id))
                            if session_distance not in fit_by_session_distances:
                                fit_by_session_distances[session_distance] = [fit_measure_pred_session]
                            else:
                                fit_by_session_distances[session_distance].append(fit_measure_pred_session)

                return fit_by_session_distances

            # And plot by difference in session
            fit_measures_by_session_distances_all_timepoints = calculate_fit_by_session_distance(fit_measures_by_session_by_timepoint=fit_measures_by_session_by_timepoint, include_0_distance=include_0_distance)
            drift_plot_all_timepoints_session_dists = self._plot_drift_distance_based(fit_measures_by_distances=fit_measures_by_session_distances_all_timepoints, self_pred_normalized=subtract_self_pred, losses_averaged_within_distances=False, all_windows_one_plot=False, timepoint_window_start_idx=999, include_0_distance=include_0_distance, distance_type="sessions")  # 999 indicates that we are considering all timepoints

            storage_filename = f"drift_plot_by_session_distances_all_timepoints"
            self.save_plot_as_file(plt=drift_plot_all_timepoints_session_dists, plot_folder=storage_folder, plot_file=storage_filename, plot_type="figure")
            plt.close(drift_plot_all_timepoints_session_dists)

            # Debugging: average timepoint data to compare values to non-timepoint dict
            if debugging:
                fit_measures_by_distances_session_level = self.average_timepoint_data_per_session(fit_measures_by_session_by_timepoint)

                storage_filename = f"var_explained_session_level_used_for_timepoint_plots.json"
                json_storage_path = os.path.join(storage_folder, storage_filename)

                with open(json_storage_path, 'w') as file:
                    logger.custom_debug(f"Storing averaged timepoint data to {json_storage_path}")
                    # Serialize and save the dictionary to the file
                    json.dump(fit_measures_by_distances_session_level, file, indent=4)

                # And store timepoint values themselves again
                storage_filename = f"var_explained_timepoints_used_for_timepoint_plots.json"
                json_storage_path = os.path.join(storage_folder, storage_filename)

                with open(json_storage_path, 'w') as file:
                    logger.custom_debug(f"Storing timepoint func in plot data to {json_storage_path}")
                    # Serialize and save the dictionary to the file
                    json.dump(fit_measures_by_session_by_timepoint, file, indent=4)


        for normalization in self.normalizations:
            if not sensor_level:
                if regions_of_interest is None:
                    # Load timepoint-based variance explained
                    storage_folder = f"data_files/{self.lock_event}/var_explained_timepoints/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                    json_storage_file = f"var_explained_timepoints_dict.json"
                    json_storage_path = os.path.join(storage_folder, json_storage_file)
                    with open(json_storage_path, 'r') as file:
                        logger.custom_debug(f"var_explained_timepoints loaded from {json_storage_path}")
                        fit_measures_by_session_by_timepoint = json.load(file)

                    plot_timepoint_window_drift_for_timepoint_fit_measures(fit_measures_by_session_by_timepoint, curr_norm=normalization)
                else:
                    for glaser_region in regions_of_interest:
                        # Load timepoint-based variance explained
                        storage_folder = f"data_files/{self.lock_event}/var_explained_timepoints/source_space{pca_folder}/{glaser_region}/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                        json_storage_file = f"var_explained_timepoints_dict.json"
                        json_storage_path = os.path.join(storage_folder, json_storage_file)
                        with open(json_storage_path, 'r') as file:
                            fit_measures_by_session_by_timepoint = json.load(file)

                        plot_timepoint_window_drift_for_timepoint_fit_measures(fit_measures_by_session_by_timepoint, glaser_region=glaser_region, pca_folder=pca_folder)
            else:
                # Load sensor- and timepoint-based variance explained
                storage_folder = f"data_files/{self.lock_event}/var_explained_sensors_timepoints/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                json_storage_file = f"var_explained_sensors_timepoints_dict.json"
                json_storage_path = os.path.join(storage_folder, json_storage_file)
                with open(json_storage_path, 'r') as file:
                    fit_measures_by_sensor_by_session_by_timepoint = json.load(file)

                sensor_index_name_dict = self.get_relevant_meg_channels(self.chosen_channels)
                sensor_names = np.array([sensor_name for sensor_name in sensor_index_name_dict['mag']['sensor_index_within_type'].values()])
                n_sensors_selected = len(sensor_names)

                if sensor_index_name_dict['grad']:
                    raise NotImplementedError("Plot not yet implemented for grad sensors")

                for sensor_idx, sensor_name in enumerate(sensor_names):
                    sensor_fit_measures_by_session_by_timepoint = fit_measures_by_sensor_by_session_by_timepoint["sensor"][str(sensor_idx)]
                    plot_timepoint_window_drift_for_timepoint_fit_measures(sensor_fit_measures_by_session_by_timepoint, sensor_name=sensor_name)


    def mne_topo_plot_per_sensor(self, data_type:str, all_timepoints_combined:bool, sessions_separate:bool=False):
        if data_type not in ["self-pred", "drift"]:
            raise ValueError(f"visualize_topo_with_drift_per_sensor called with invalid argument for data_type {data_type}")
        if (sessions_separate and data_type == "drift") or (sessions_separate and all_timepoints_combined):
            raise NotImplementedError("Invalid parameter configuration.")

        for normalization in self.normalizations:
            # Load sensor- and timepoint-based variance explained
            storage_folder = f"data_files/{self.lock_event}/var_explained_sensors_timepoints/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
            json_storage_file = f"var_explained_sensors_timepoints_dict.json"
            json_storage_path = os.path.join(storage_folder, json_storage_file)
            with open(json_storage_path, 'r') as file:
                fit_measures_by_sensor_by_session_by_timepoint = json.load(file)

            if self.omit_sessions:
                fit_measures_by_sensor_by_session_by_timepoint = self.omit_selected_sessions_from_fit_measures(fit_measures_by_session=fit_measures_by_sensor_by_session_by_timepoint, omitted_sessions=self.omit_sessions, sensors_seperated=True)
           
            # Get sensor names (maybe not needed)
            sensor_index_name_dict = self.get_relevant_meg_channels(self.chosen_channels)
            if sensor_index_name_dict['grad']:
                raise NotImplementedError("Plot not yet implemented for grad sensors")
            sensor_names = np.array([sensor_name for sensor_name in sensor_index_name_dict['mag']['sensor_index_within_type'].values()])
            selected_sensors_indices_total = np.array([index for index in sensor_index_name_dict['mag']['sensor_index_total'].keys()])
            n_sensors_selected = len(sensor_names)

            timepoint_indices = np.array(list(range(1 + self.timepoint_max - self.timepoint_min)))
            
            if data_type == "drift":
                # Calculate drift for each sensor seperate (over all timepoints)
                drift_correlations_sensors = []
                for sensor_idx, sensor_name in enumerate(sensor_names):
                    sensor_fit_measures_by_session_by_timepoint = fit_measures_by_sensor_by_session_by_timepoint["sensor"][str(sensor_idx)]
                    logger.custom_debug(f"Processing sensor {sensor_name}")
                    
                    if all_timepoints_combined:
                        # Calculate drift correlation (correlation between distance and fit measure)
                        sensor_fit_measures_by_distances = self.calculate_fit_by_distances(fit_measures_by_session=sensor_fit_measures_by_session_by_timepoint, timepoint_level_input=True, average_within_distances=False)
                        x_values, y_values = self.extract_x_y_arrays(sensor_fit_measures_by_distances, losses_averaged_within_distances=False)
                        x_values_set = np.array(list(set(x_values)))  # Required/Useful for non-averaged fit measures within distances: x values occur multiple times
                        slope, intercept, r_value, p_value, std_err = linregress(x=x_values, y=y_values)
                        r_value = "{:.3f}".format(r_value)  # limit to three decimals

                        drift_correlations_sensors.append(float(r_value))
                    else:
                        sensor_drift_correlations_timepoints = []
                        for timepoint_idx in timepoint_indices:
                            # Filter dict for current timepoint
                            sensor_timepoint_fit_measures_by_session = self.recursive_defaultdict()
                            for session_train_id, fit_measures_train_session in sensor_fit_measures_by_session_by_timepoint['session_mapping'].items():
                                for session_pred_id, fit_measures_pred_session in fit_measures_train_session["session_pred"].items():
                                    sensor_timepoint_fit_measures_by_session["session_mapping"][session_train_id]["session_pred"][session_pred_id] = fit_measures_pred_session["timepoint"][str(timepoint_idx)]

                            sensor_fit_measures_by_distances = self.calculate_fit_by_distances(fit_measures_by_session=sensor_timepoint_fit_measures_by_session, timepoint_level_input=False, average_within_distances=False)
                            r_value = float(self.calc_drift_corr_with_fit_measures_by_distances(sensor_fit_measures_by_distances))

                            sensor_drift_correlations_timepoints.append(r_value)
                        drift_correlations_sensors.append(np.array(sensor_drift_correlations_timepoints))

                drift_correlations_sensors = np.array(drift_correlations_sensors)
                min_corr = "{:.4f}".format(np.min(drift_correlations_sensors))
                max_corr = "{:.4f}".format(np.max(drift_correlations_sensors))

                # plot_topomap always expects shape (n_sensors, n_timepoints)
                if all_timepoints_combined:
                    drift_correlations_sensors = drift_correlations_sensors.reshape(n_sensors_selected, 1)
            else:
                if not sessions_separate:
                    # Aggregate self-pred values on sensor (and timepoint) level, averaged over sessions. I need one value per sensor, per timepoint
                    self_pred_values_sensors = []
                    n_considered_sessions = len(list(fit_measures_by_sensor_by_session_by_timepoint["sensor"]["0"]["session_mapping"].keys()))
                    for sensor_idx, sensor_name in enumerate(sensor_names):
                        sensor_fit_measures_by_session_by_timepoint = fit_measures_by_sensor_by_session_by_timepoint["sensor"][str(sensor_idx)]
                        sensor_self_preds_timepoints = []
                        # Average self-preds for each timepoint over all considered sessions
                        for timepoint_idx in timepoint_indices:
                            sensor_timepoint_sum_over_sessions = 0
                            for session_id in sensor_fit_measures_by_session_by_timepoint["session_mapping"]:
                                sensor_timepoint_sum_over_sessions += sensor_fit_measures_by_session_by_timepoint["session_mapping"][str(session_id)]["session_pred"][str(session_id)]["timepoint"][str(timepoint_idx)]

                            sensor_timepoint_self_pred = sensor_timepoint_sum_over_sessions / n_considered_sessions
                            sensor_self_preds_timepoints.append(sensor_timepoint_self_pred)
                        self_pred_values_sensors.append(np.array(sensor_self_preds_timepoints))
                    self_pred_values_sensors = np.array(self_pred_values_sensors)
                        
                    min_var_explained = "{:.4f}".format(np.min(self_pred_values_sensors))
                    max_var_explained = "{:.4f}".format(np.max(self_pred_values_sensors))

                    if all_timepoints_combined:
                        # Average across all timepoint values
                        self_pred_values_sensors = np.array([np.mean(sensor_self_preds_timepoints) for sensor_self_preds_timepoints in self_pred_values_sensors]).reshape(n_sensors_selected, 1)
                else:
                    self_pred_values_sensors_by_session = {}
                    min_var_explained_by_session = {}
                    max_var_explained_by_session = {}
                    # Seperate plot for each session
                    for session_id in fit_measures_by_sensor_by_session_by_timepoint["sensor"]["0"]["session_mapping"]:
                        self_pred_values_sensors = []
                        for sensor_idx, sensor_name in enumerate(sensor_names):
                            sensor_fit_measures_by_session_by_timepoint = fit_measures_by_sensor_by_session_by_timepoint["sensor"][str(sensor_idx)]
                            sensor_self_preds_timepoints = []
                            for timepoint_idx in timepoint_indices:
                                sensor_timepoint_val = sensor_fit_measures_by_session_by_timepoint["session_mapping"][str(session_id)]["session_pred"][str(session_id)]["timepoint"][str(timepoint_idx)]
                                sensor_self_preds_timepoints.append(sensor_timepoint_val)
                            self_pred_values_sensors.append(np.array(sensor_self_preds_timepoints))
                        self_pred_values_sensors = np.array(self_pred_values_sensors)
                        self_pred_values_sensors_by_session[session_id] = self_pred_values_sensors
                        min_var_explained_by_session[session_id] = "{:.4f}".format(np.min(self_pred_values_sensors))
                        max_var_explained_by_session[session_id] = "{:.4f}".format(np.max(self_pred_values_sensors))

            # Create mne info object
            fif_file_path = f'/share/klab/datasets/avs/population_codes/as{self.subject_id}/sensor/filter_0.2_200/saccade_evoked_{self.subject_id}_01_.fif'
            og_evoked = mne.read_evokeds(fif_file_path)[0]
            mne_info = og_evoked.info
            selected_mag_info = mne.pick_info(mne_info, selected_sensors_indices_total)

            # Get arr of timepoints to plot
            if all_timepoints_combined:
                # These are dummy values, we average across all timepoints but mne requires a timepoint value
                t_start = 0
                timepoints = [0]
            else:
                # Timepoints need to be converted from indices to seconds. map_timepoint_idx_to_ms maps only maps to ms, /1000 converts to seconds
                timepoints = np.array([float(self.map_timepoint_idx_to_ms(timepoint_idx)/1000) for timepoint_idx in timepoint_indices])
                t_start = timepoints[0]

            if not sessions_separate:
                if data_type == "drift":
                    data_array_to_plot = drift_correlations_sensors
                else:
                    data_array_to_plot = self_pred_values_sensors

                # Create new Evoked object with correct drift data and timepoint metadata
                evoked = mne.EvokedArray(data_array_to_plot, selected_mag_info, tmin=t_start)
                logger.custom_info(f"Topo plot time range: {evoked.times[0]} - {evoked.times[-1]}")

                fig = evoked.plot_topomap(timepoints, ch_type="mag", colorbar=False)  # , axes=axes)
                if data_type == "drift":
                    fig.suptitle(f'Drift on Sensor Level. Negative correlations (drift) are in blue, positive correlations are in red. \n Min r: {min_corr}. Max r: {max_corr}', fontsize=18)
                else:
                    fig.suptitle(f'Self-Pred Variance Explained on Sensor Level. Negative values are in blue, positive values are in red. \n Min var_explained: {min_var_explained}. Max var_explained: {max_var_explained}', fontsize=18)

                plot_folder =  f"data_files/{self.lock_event}/visualizations/topographic_plots/{data_type}/subject_{self.subject_id}/"
                all_timepoints_name_addition = "_all_timepoints" if all_timepoints_combined else ""
                plot_file = f"{data_type}_topo_plot{all_timepoints_name_addition}.png"
                self.save_plot_as_file(plt=fig, plot_folder=plot_folder, plot_file=plot_file)
            else:
                for session_id, self_pred_values_sensors in self_pred_values_sensors_by_session.items():                    
                    # Create new Evoked object with correct drift data and timepoint metadata
                    evoked = mne.EvokedArray(self_pred_values_sensors, selected_mag_info, tmin=t_start)
                    fig = evoked.plot_topomap(timepoints, ch_type="mag", colorbar=False)  # , axes=axes)
                    fig.suptitle(f'Session {session_id} Self-Pred Variance Explained on Sensor Level. Negative values are in blue, positive values are in red. \n Min var_explained: {min_var_explained_by_session[session_id]}. Max var_explained: {max_var_explained_by_session[session_id]}', fontsize=18)

                    plot_folder =  f"data_files/{self.lock_event}/visualizations/topographic_plots/{data_type}/sessions_separate/subject_{self.subject_id}/"
                    all_timepoints_name_addition = "_all_timepoints" if all_timepoints_combined else ""
                    plot_file = f"Session_{session_id}_{data_type}_topo_plot{all_timepoints_name_addition}.png"
                    self.save_plot_as_file(plt=fig, plot_folder=plot_folder, plot_file=plot_file)


            # Plot channel locations
            #fig = og_evoked.plot_sensors(show_names=True, ch_type="mag")
            # Reduce font size (otherwise non-readable due to overlap)
            #for text in fig.axes[0].texts:
            #    text.set_fontsize(5)  

            #plot_folder =  f"data_files/{self.lock_event}/visualizations/topographic_plots/subject_{self.subject_id}/"
            #plot_file = "sensor_locations.png"
            #self.save_plot_as_file(plt=fig, plot_folder=plot_folder, plot_file=plot_file)


    def visualize_meg_epochs_mne(self):
        """
        Visualizes meg data at various processing steps
        """
        for session_id_num in self.session_ids_num:
            # Load meg data and split into grad and mag
            meg_data = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="meg_data")
            meg_data = meg_data["test"]
            meg_dict = {"grad": {"meg": meg_data[:,:204,:], "n_sensors": 204}, "mag": {"meg": meg_data[:,204:,:], "n_sensors": 102}}
            

            logger.custom_debug(f"meg_data.shape: {meg_data.shape}")
            logger.custom_debug(f"meg_dict['grad']['meg'].shape: {meg_dict['grad']['meg'].shape}")
            logger.custom_debug(f"meg_dict['mag']['meg'].shape: {meg_dict['mag']['meg'].shape}")

            for sensor_type in meg_dict:
                # Read in with mne
                meg_info = mne.create_info(ch_names=[str(sensor_nr) for sensor_nr in range(meg_dict[sensor_type]["n_sensors"])], sfreq=500, ch_types=sensor_type)  # Create minimal Info objects with default values
                epochs = mne.EpochsArray(meg_dict[sensor_type]["meg"], meg_info)

                # Plot
                epochs_plot = epochs.plot()
                plot_folder = f"data_files/{self.lock_event}/visualizations/meg_data/subject_{self.subject_id}/session_{session_id_num}/{sensor_type}"
                plot_file = f"{sensor_type}_plot.png"
                self.save_plot_as_file(plt=epochs_plot, plot_folder=plot_folder, plot_file=plot_file, plot_type="mne")

    
    def plot_ERPs(self, plot_norms:list, regions_of_interest:list, non_preprocessed:bool=False):
        """
        Visualizes meg data in ERP fashion, averaged over sessions and channels.
        """
        if not non_preprocessed:
            if regions_of_interest is not None:
                for glaser_region in regions_of_interest:
                    for session_id in self.session_ids_num:
                        # Use defaultdict to automatically create missing keys
                        session_by_norms_dict = self.recursive_defaultdict()
                        n_timepoints = None
                        for normalization in plot_norms:
                            # Load meg data for region, session and norm
                            meg_data = {"train": None, "test": None}
                            for split in meg_data:
                                storage_folder = f"data_files/{self.lock_event}/meg_data/source_space/{glaser_region}/{normalization}/subject_{self.subject_id}/session_{session_id}/{split}"  
                                storage_file = "meg_data.npy"
                                storage_path = os.path.join(storage_folder, storage_file)

                                meg_data[split] = np.load(storage_path)
                            meg_data_complete = np.concatenate((meg_data["train"], meg_data["test"])) 

                            # Calculate the mean over the epochs and voxels (not timepoints)
                            mean_timepoint_values = np.mean(meg_data_complete, axis=(0, 1))
                            session_by_norms_dict[normalization] = mean_timepoint_values

                            if n_timepoints is None:
                                n_timepoints = len(mean_timepoint_values)

                        timepoints = np.array(list(range(n_timepoints)))

                        # Plotting
                        plt.figure(figsize=(10, 6))
                        for normalisation in plot_norms:
                            plt.plot(timepoints, session_by_norms_dict[normalisation], label=f'{normalisation}')

                        plt.xlabel('Timepoints)')
                        plt.ylabel('MEG Value')
                        plt.title(f'ERP-like Average MEG Signal, averaged over Epochs and Voxels. Session {session_id} Region {glaser_region}')
                        plt.legend()

                        # Save plot
                        plot_folder = f"data_files/{self.lock_event}/visualizations/meg_data/ERP_like/source_space/{glaser_region}"
                        plot_file = f"Region-{glaser_region}_Session-{session_id}_plot.png"
                        self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
            else:
                for session_id in self.session_ids_num:
                    # Use defaultdict to automatically create missing keys
                    session_by_norms_dict = self.recursive_defaultdict()
                    n_timepoints = None
                    for normalization in plot_norms:
                        # Load meg data for region, session and norm
                        meg_data = self.load_split_data_from_file(session_id_num=session_id, type_of_content="meg_data", type_of_norm=normalization)
                        meg_data_complete = np.concatenate((meg_data["train"], meg_data["test"])) 

                        # Calculate the mean over the epochs and channels (not timepoints)
                        mean_timepoint_values = np.mean(meg_data_complete, axis=(0, 1))
                        session_by_norms_dict[normalization] = mean_timepoint_values

                        if n_timepoints is None:
                            n_timepoints = len(mean_timepoint_values)

                    timepoints = np.array(list(range(n_timepoints)))

                    # Plotting
                    plt.figure(figsize=(10, 6))
                    for normalisation in plot_norms:
                        plt.plot(timepoints, session_by_norms_dict[normalisation], label=f'{normalisation}')

                    plt.xlabel('Timepoints)')
                    plt.ylabel('MEG Value')
                    plt.title(f'ERP-like Average MEG Signal, averaged over Epochs and Channels. Session {session_id}')
                    plt.legend()

                    # Save plot
                    plot_folder = f"data_files/{self.lock_event}/visualizations/meg_data/ERP_like"
                    plot_file = f"Session-{session_id}_plot.png"
                    self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
        else:
            if regions_of_interest is not None:
                meg_data_folder = f'/share/klab/datasets/avs/population_codes/as{self.subject_id}/source_space/beamformer/glasser/ori_None/hem_lh/filter_0.2_200/ica'
                for session_id_char in self.session_ids_char:
                    session_id = self.map_session_letter_id_to_num(session_id_char)
                    meg_file_name = f"as{self.subject_id}{session_id_char}_population_codes_{self.lock_event}_500hz_masked_False.h5"
                    meg_data_path = os.path.join(meg_data_folder, meg_file_name)

                    with h5py.File(meg_data_path, "r") as meg_file:
                        for glaser_region in regions_of_interest:
                            # Load meg data for region and session 
                            meg_data_reg_sess = np.array(meg_file[glaser_region]['onset'])

                            # Calculate the mean over the epochs and voxels (not timepoints)
                            mean_timepoint_values = np.mean(meg_data_reg_sess, axis=(0, 1))

                            # Plotting
                            plt.figure(figsize=(10, 6))
                            timepoints = np.array(list(range(len(mean_timepoint_values))))
                            plt.plot(timepoints, mean_timepoint_values)

                            plt.xlabel('Timepoints)')
                            plt.ylabel('MEG Value')
                            plt.title(f'ERP-like Average MEG Signal (non-preprocessed), averaged over Epochs and Voxels. Session {session_id} Region {glaser_region}')

                            # Save plot
                            plot_folder = f"data_files/{self.lock_event}/visualizations/meg_data/ERP_like/non_preprocessed/source_space/{glaser_region}"
                            plot_file = f"Non_preprocessed_Region-{glaser_region}_Session-{session_id}_plot.png"
                            self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
            else:
                raise NotImplementedError("plot_ERPs not yet implemented to visualize non-preproccessed non-source meg data.")


    def visualize_meg_ERP_style(self, plot_norms: list):
        """
        Visualizes meg data in ERP fashion, averaged over sessions and channels.
        """
        # To-do: Combine data from normalizations for the same session and sensor type into one plot
        # hopefully this is at least somewhat reasonable with the different scales
        for session_id_num in self.session_ids_num:
            # Use defaultdict to automatically create missing keys
            session_dict = self.recursive_defaultdict()
            for normalization in plot_norms:
                # Load meg data and split into grad and mag
                meg_data = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="meg_data", type_of_norm=normalization)
                meg_data_complete = np.concatenate((meg_data["train"], meg_data["test"]))
                meg_dict = {"grad": {"meg": meg_data_complete[:,:204,:], "n_sensors": 204}, "mag": {"meg": meg_data_complete[:,204:,:], "n_sensors": 102}}
                # Store number of timepoints in current processing for plots

                for sensor_type in meg_dict:
                    data = meg_dict[sensor_type]["meg"]

                    # Calculate the mean over the epochs and sensors
                    averaged_data = np.mean(data, axis=(0, 1))
                    #logger.custom_debug(f"averaged_data.shape: {averaged_data.shape}")
                    # Store data in session dict
                    session_dict["norm"][normalization]["sensor_type"][sensor_type] = averaged_data
                    #logger.custom_debug(f"session_dict['norm'][norm]['sensor_type'][sensor_type].shape: {session_dict['norm'][normalization]['sensor_type'][sensor_type].shape}")

            for sensor_type in ["grad", "mag"]:
                timepoints = np.array(list(range(session_dict["norm"][norm]["sensor_type"][sensor_type].shape[0])))
                raise ValueError(f"timepoints is {timepoints}, shape is {session_dict['norm'][norm]['sensor_type'][sensor_type].shape} debug this if necessary, else delete this Error raise.")

                # Plotting
                plt.figure(figsize=(10, 6))
                for norm in plot_norms:
                    if sensor_type in session_dict["norm"][norm]["sensor_type"]:
                        plt.plot(timepoints, session_dict["norm"][norm]["sensor_type"][sensor_type], label=f'{norm}')
                        #logger.custom_debug(f"session_dict['norm'][norm]['sensor_type'][sensor_type].shape: {session_dict['norm'][norm]['sensor_type'][sensor_type].shape}")
                    else:
                        logger.warning(f"Wrong key combination: {norm} and {sensor_type}")

                plt.xlabel('Timepoints)')
                plt.ylabel('Average MEG Value')
                plt.title(f'ERP-like Average MEG Signal over Epochs and Sensors. Session {session_id_num} Sensor {sensor_type}')
                plt.legend()

                # Save plot
                plot_folder = f"data_files/{self.lock_event}/visualizations/meg_data/ERP_like/{sensor_type}_combined-norms"
                plot_file = f"Session-{session_id_num}_Sensor-{sensor_type}_plot.png"
                self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)

    def visualize_model_perspective(self, plot_norms: list, seperate_plots=False):
        """
        DEPRECATED. Replaced by new_visualize_model_perspective
        Visualizes meg data from the regression models perspective. This means, we plot the values over the epochs for each timepoint, averaged over the sensors.
        """
        for session_id_num in self.session_ids_num:
            # Use defaultdict to automatically create missing keys
            session_dict = self.recursive_defaultdict()
            for normalization in plot_norms:
                # Load meg data and split into grad and mag
                meg_data = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="meg_data", type_of_norm=normalization)
                meg_data_complete = np.concatenate((meg_data["train"], meg_data["test"]))
                meg_dict = {"grad": {"meg": meg_data_complete[:,:204,:], "n_sensors": 204}, "mag": {"meg": meg_data_complete[:,204:,:], "n_sensors": 102}}
                
                for sensor_type in meg_dict:
                    data = meg_dict[sensor_type]["meg"]

                    # Calculate the mean over sensors for each timepoint and epoch
                    averaged_data = np.mean(data, axis=1)  # (epochs, timepoints)
                    # Store data in session dict
                    session_dict["norm"][normalization]["sensor_type"][sensor_type] = averaged_data
                    #logger.custom_debug(f"session_dict['norm'][norm]['sensor_type'][sensor_type].shape: {session_dict['norm'][normalization]['sensor_type'][sensor_type].shape}")

            for sensor_type in ["grad", "mag"]:
                timepoints = np.array(list(range(session_dict["norm"][plot_norms[0]]["sensor_type"][sensor_type].shape[0])))
                #raise ValueError(f"timepoints is {timepoints}, shape is {session_dict['norm'][plot_norms[0]]['sensor_type'][sensor_type].shape} debug this if necessary, else delete this Error raise.")
                #timepoints = np.array(list(range(601)))
                

                # Select timepoints to plot (f.e. 10 total, every 60th)
                plot_timepoints = []
                timepoint_plot_interval = 150

                for timepoint in range(1, max(timepoints)+1, timepoint_plot_interval):
                    plot_timepoints.append(timepoint)
                n_plot_timepoints = len(plot_timepoints)
                legend_elements = []  # List to hold the custom legend elements (colors for norms)

                num_epochs_for_x_axis = 0

                # Plotting
                if not seperate_plots:
                    plt.figure(figsize=(10, 6))
                for norm_idx, norm in enumerate(plot_norms):
                    if seperate_plots:
                        plt.figure(figsize=(10, 6))
                    # Get data for timepoints
                    meg_norm_sensor = session_dict["norm"][norm]["sensor_type"][sensor_type]

                    num_epochs_for_x_axis = num_epochs = meg_norm_sensor.shape[0]
                    
                    # Select epochs to plot (f.e. 200 total, every 10th or smth)
                    plot_epochs = []
                    epoch_plot_interval = 10
                    for epoch in range(1, num_epochs_for_x_axis, epoch_plot_interval):
                        plot_epochs.append(epoch)

                    plot_epochs = np.array(plot_epochs)
                    epochs = np.array(list(range(num_epochs)))

                    filtered_epoch_meg = meg_norm_sensor[plot_epochs, :]

                    for timepoint in plot_timepoints:
                        filtered_timepoint_meg = filtered_epoch_meg[:, timepoint]
                        plt.plot(plot_epochs, filtered_timepoint_meg, linewidth=0.5, color=f'C{norm_idx}')  # Use color index linked to normalization and line index linked to timepoint
                    # Create a custom legend element for this normalization
                    legend_elements.append(Line2D([0], [0], color=f'C{norm_idx}', lw=4, label=norm))

                    if seperate_plots:
                        # Set x-axis to show full range of epochs
                        plt.xlim(1, num_epochs_for_x_axis)

                        plt.xlabel('Epochs in Session)')
                        plt.ylabel('MEG Value averaged over Channels')
                        plt.title(f'Average MEG Signal over Channels per timepoint (showing {n_plot_timepoints} timepoints). Session {session_id_num} Sensor {sensor_type} {norm}')
                        plt.legend(handles=legend_elements, title="Normalization Methods")

                        # Save plot
                        plot_folder = f"data_files/{self.lock_event}/visualizations/meg_data/regression_model_perspective/{norm}/{sensor_type}"
                        plot_file = f"Session-{session_id_num}_Sensor-{sensor_type}_timepoint-overview.png"
                        self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)


                if not seperate_plots:
                    # Set x-axis to show full range of epochs
                    plt.xlim(1, num_epochs_for_x_axis)

                    plt.xlabel('Epochs in Session)')
                    plt.ylabel('MEG Value averaged over Channels')
                    plt.title(f'Average MEG Signal over Channels per timepoint (showing {n_plot_timepoints} timepoints). Session {session_id_num} Sensor {sensor_type}')
                    plt.legend(handles=legend_elements, title="Normalization Methods")

                    # Save plot
                    plot_folder = f"data_files/{self.lock_event}/visualizations/meg_data/regression_model_perspective/{sensor_type}"
                    plot_file = f"Session-{session_id_num}_Sensor-{sensor_type}_timepoint-overview.png"
                    self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)

    def new_visualize_model_perspective(self, plot_norms:list=None, before_preprocessing:bool=False, regions_of_interest:list=None, source_pca_type:str=None, whiten:bool=None):
        """
        Visualizes meg data from the regression models perspective. This means, we plot the values over the epochs for each timepoint, one line for each selected sensor.
        """
        whiten_folder = "/whiten" if whiten else ""
        def plot_model_perspective(meg_data_complete:np.ndarray, session_id:str, normalization:str, plot_timepoints:list, n_epochs_total:int, n_epochs_train:int, channel_names_by_indices:dict=None, voxel_indices:list=None, glaser_region:str=None, pca_folder:str=""):
            """
            Generates value plot over epochs based on selected meg_data, timepoints and sensors.
            """
            colormap = plt.cm.get_cmap('viridis', len(voxel_indices)) if voxel_indices is not None else plt.cm.get_cmap('viridis', len(list(channel_names_by_indices.keys())))

            for timepoint_idx in plot_timepoints:
                timepoint_name = self.timepoint_min+timepoint_idx
                legend_elements = []  # List to hold the custom legend elements (colors for norms)
                plt.figure(figsize=(10, 6))
                # Filter meg for timepoint
                meg_timepoint = meg_data_complete[:,:,timepoint_idx]  # [epochs, channels, timepoints]

                if channel_names_by_indices is not None:
                    second_dim = "sensor"
                    source_folder_addition = ""
                    n_channels = len(list(channel_names_by_indices.keys()))
                    for channel_idx in range(n_channels):
                        # Filter meg for channel
                        meg_timepoint_channel = meg_timepoint[:,channel_idx] # [epochs, channels]
                        plt.plot(list(range(n_epochs_total)), meg_timepoint_channel, linewidth=0.2, alpha=0.6, color=colormap(channel_idx))

                        legend_elements.append(Line2D([0], [0], color=colormap(channel_idx), lw=4, label=channel_names_by_indices[channel_idx]))
                elif voxel_indices is not None:
                    second_dim = "voxel" if pca_folder == "" else "pc"
                    source_folder_addition = f"/source_space{pca_folder}/{glaser_region}"
                    for voxel_nr, voxel_idx in enumerate(voxel_indices):  # iterate over voxels (or pcs in same dim)
                        # Filter meg for voxel
                        meg_timepoint_voxel = meg_timepoint[:,voxel_idx] # [epochs, voxels]
                        plt.plot(list(range(n_epochs_total)), meg_timepoint_voxel, linewidth=0.2, alpha=0.5, color=colormap(voxel_nr))
                        plt.axhline(np.mean(meg_timepoint_voxel), linestyle='--', linewidth=2, color=colormap(voxel_nr))

                        legend_elements.append(Line2D([0], [0], color=colormap(voxel_nr), lw=4, label=f"{second_dim} id {voxel_idx}"))
                else:
                    raise ValueError("Either channel_names_by_indices or regions_by_indices need to be provided.")

                # TODO: Color train and test seperately

                # Set x-axis to show full range of epochs
                #plt.xlim(1, n_total)

                plt.xlabel('Epochs in Session)')
                plt.ylabel('MEG Value')
                plt.axvline(x=n_epochs_train, color='r', linestyle='--', linewidth=1, label='Train/Test Split')
                plt.title(f'MEG Signal over Epochs for {second_dim}s. \n Session: {session_id}. Timepoint: {timepoint_name} Norm: {normalization}')
                plt.legend(handles=legend_elements, title=f"{second_dim}")
                
                # Save plot
                plot_folder = f"data_files/{self.lock_event}/visualizations/meg_data/new_regression_model_perspective{source_folder_addition}/{normalization}/timepoint_{timepoint_name}"
                plot_file = f"Session-{session_id}_timepoint-overview.png"
                self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)

        if regions_of_interest is None:
            if before_preprocessing:
                normalization = "no_norm"
                for session_id in self.session_ids_num:
                    session_dict = self.recursive_defaultdict()
                    # Load meg data without any preprocessing (normalization or clipping)
                    meg_data = {"train": None, "test": None}
                    for split in meg_data:
                        storage_folder = f"data_files/{self.lock_event}/meg_data/norm_{normalization}/subject_{self.subject_id}/session_{session_id}/{split}"  
                        storage_file = "meg_data.npy"
                        storage_path = os.path.join(storage_folder, storage_file)
                        meg_data[split] = np.load(storage_path)
                    meg_data_complete = np.concatenate((meg_data["train"], meg_data["test"]))

                    n_epochs_train = len(meg_data["train"])
                    n_epochs_total = n_epochs_train + len(meg_data["test"])

                    # Extract channel names for indices
                    selected_channel_indices = self.get_relevant_meg_channels(chosen_channels=self.chosen_channels)
                    channel_names_by_indices = {}
                    ch_ix = 0
                    for sensor_type in selected_channel_indices:
                        for sensor_index_within_type in selected_channel_indices[sensor_type]["sensor_index_within_type"]:
                            channel_names_by_indices[ch_ix] = selected_channel_indices[sensor_type]["sensor_index_within_type"][sensor_index_within_type]
                            ch_ix += 1
                    
                    timepoints = np.array(list(range(1 + self.timepoint_max - self.timepoint_min)))

                    # Select timepoints to plot (f.e. 10 total, every 60th)
                    plot_timepoints = []
                    timepoint_plot_interval = 20

                    for timepoint_index in range(min(timepoints), max(timepoints)+1, timepoint_plot_interval):
                        plot_timepoints.append(timepoint_index)
                    n_plot_timepoints = len(plot_timepoints)

                    plot_model_perspective(meg_data_complete=meg_data_complete, session_id=session_id, normalization=normalization, plot_timepoints=plot_timepoints, n_epochs_total=n_epochs_total, n_epochs_train=n_epochs_train, channel_names_by_indices=channel_names_by_indices)
            else:
                for session_id in self.session_ids_num:
                    session_dict = self.recursive_defaultdict()
                    for normalization in plot_norms:
                        # Load meg data and split into grad and mag
                        meg_data = self.load_split_data_from_file(session_id_num=session_id, type_of_content="meg_data", type_of_norm=normalization)
                        meg_data_complete = np.concatenate((meg_data["train"], meg_data["test"]))

                        n_epochs_train = len(meg_data["train"])
                        n_epochs_total = n_epochs_train + len(meg_data["test"])

                        # Extract channel names for indices
                        selected_channel_indices = self.get_relevant_meg_channels(chosen_channels=self.chosen_channels)
                        channel_names_by_indices = {}
                        ch_ix = 0
                        for sensor_type in selected_channel_indices:
                            for sensor_index_within_type in selected_channel_indices[sensor_type]["sensor_index_within_type"]:
                                channel_names_by_indices[ch_ix] = selected_channel_indices[sensor_type]["sensor_index_within_type"][sensor_index_within_type]
                                ch_ix += 1
                        
                        timepoints = np.array(list(range(1 + self.timepoint_max - self.timepoint_min)))

                        # Select timepoints to plot (f.e. 10 total, every 60th)
                        plot_timepoints = []
                        timepoint_plot_interval = 20

                        for timepoint_index in range(min(timepoints), max(timepoints)+1, timepoint_plot_interval):
                            plot_timepoints.append(timepoint_index)
                        n_plot_timepoints = len(plot_timepoints)

                        plot_model_perspective(meg_data_complete=meg_data_complete, session_id=session_id, normalization=normalization, plot_timepoints=plot_timepoints, n_epochs_total=n_epochs_total, n_epochs_train=n_epochs_train, channel_names_by_indices=channel_names_by_indices)
        else:
            if source_pca_type == 'voxels':
                pca_folder = f"/voxels_pca_reduced{whiten_folder}"
            elif source_pca_type == 'voxels_and_timepoints':
                raise NotImplementedError("Model perspective visualization not yet implemented for pca over timepoints.")
                #pca_folder = f"/voxels_and_timepoints_pca_reduced{whiten_folder}"
            else:
                pca_folder = ""
            for glaser_region in regions_of_interest:
                for session_id in self.session_ids_num:
                    # Use defaultdict to automatically create missing keys
                    session_dict = self.recursive_defaultdict()
                    for normalization in plot_norms:
                        # Load meg data and split into grad and mag
                        meg_data = {"train": None, "test": None}
                        for split in meg_data:
                            storage_folder = f"data_files/{self.lock_event}/meg_data/source_space{pca_folder}/{glaser_region}/{normalization}/subject_{self.subject_id}/session_{session_id}/{split}"  
                            storage_file = "meg_data.npy"
                            storage_path = os.path.join(storage_folder, storage_file)

                            meg_data[split] = np.load(storage_path)
                        meg_data_complete = np.concatenate((meg_data["train"], meg_data["test"]))

                        n_epochs_train = len(meg_data["train"])
                        n_epochs_total = len(meg_data_complete)

                        n_voxels = meg_data_complete.shape[1]
                        voxel_step_size = n_voxels // 5 if n_voxels // 5 >= 1 else 1

                        # Select some example voxels
                        voxel_indices = []
                        for voxel_idx in range(0, n_voxels, voxel_step_size):
                            voxel_indices.append(voxel_idx)

                        timepoints = np.array(list(range(1 + self.timepoint_max - self.timepoint_min)))

                        # Select example timepoints (f.e. 10 total, every 60th)
                        plot_timepoints = []
                        timepoint_plot_interval = 20

                        for timepoint_index in range(min(timepoints), max(timepoints)+1, timepoint_plot_interval):
                            plot_timepoints.append(timepoint_index)
                        n_plot_timepoints = len(plot_timepoints)

                        plot_model_perspective(meg_data_complete=meg_data_complete, session_id=session_id, normalization=normalization, plot_timepoints=plot_timepoints, n_epochs_total=n_epochs_total, n_epochs_train=n_epochs_train, voxel_indices=voxel_indices, glaser_region=glaser_region, pca_folder=pca_folder)
                

    def visualize_meg_means_stds(self, x_value_days:bool=False):
        """
        Visualizes mean (and if selected std) of meg data for 3 exemplary timepoints comparing all sessions, as requested by Carmen.
        """
        if self.subject_id == "02" and not x_value_days:
            # Adjust for repeated session 04
            new_session_ids = ["1", "2", "3", "10", "4", "5", "6", "7", "8", "9"]
            session_mapping = dict(zip(self.session_ids_num, new_session_ids))
            subject_02_title_addition = "\n Repeated Session 4 inserted as final session."
        else:
            subject_02_title_addition = ""
        session_date_differences = self.get_session_date_differences()
        for normalization in self.normalizations:
            # Collect means and stds of sessions
            mean_and_std_dict_by_timepoints = self.recursive_defaultdict()
            mean_and_std_dict_timepoint_avg = self.recursive_defaultdict()
            for session_id in self.session_ids_num:
                if x_value_days:
                    # aggregate values by distance in days from first session
                    x_value = session_date_differences["1"][session_id]
                else:
                    if self.subject_id != "02":
                        x_value = int(session_id)
                    else:
                        x_value = int(session_mapping[session_id])
                # Get session meg data (train and test split combined)
                session_meg_data = []
                for split in ["train", "test"]:
                    split_path = f"data_files/{self.lock_event}/meg_data/norm_{normalization}/subject_{self.subject_id}/session_{session_id}/{split}/meg_data.npy"  
                    split_data = np.load(split_path)
                    session_meg_data.extend(split_data)
                session_meg_data = np.array(session_meg_data)

                # Debugging
                mean_timepoints_normalized = np.mean(session_meg_data, axis=(0,1))
                mean_normalized = np.mean(session_meg_data, axis=None)
                print(f"Subject {self.subject_id}: Session {session_id}: session_meg_data.shape: {session_meg_data.shape}")
                #print(f"Session {session_id} mean_timepoints_normalized.shape: {mean_timepoints_normalized.shape}")
                #print(f"Session {session_id} mean_session: {mean_normalized}. \n")

                plt.figure()
                plt.plot([self.map_timepoint_idx_to_ms(timepoint_idx) for timepoint_idx in range(len(mean_timepoints_normalized))], mean_timepoints_normalized)
                plt.title(f'Mean across selected timepoints')
                plt.xlabel(f'Timepoint in ms relative to {self.lock_event} onset.')
                plt.ylabel('Mean')
                
                save_folder = f"data_files/{self.lock_event}/visualizations/meg_data/test/subject_{self.subject_id}"  
                save_file = f"session_{session_id}_test_mean_across_timepoints.png"
                self.save_plot_as_file(plt=plt, plot_folder=save_folder, plot_file=save_file)

                # Select exemplary timepoints and store their mean and std
                n_timepoints = session_meg_data.shape[2]  # epochs, sensors, timepoints
                selected_timepoint_idx = [timepoint_idx for timepoint_idx in range(0, n_timepoints, int(n_timepoints/4))]  # choose  timepoints: in beginning, middle and end
                
                for timepoint_idx in selected_timepoint_idx:
                    timepoint_mean = np.mean(session_meg_data[:,:,timepoint_idx])
                    timepoint_std = np.std(session_meg_data[:,:,timepoint_idx])

                    mean_and_std_dict_by_timepoints["timepoint"][timepoint_idx]["means"][x_value] = timepoint_mean
                    mean_and_std_dict_by_timepoints["timepoint"][timepoint_idx]["stds"][x_value] = timepoint_std

                # Calculate mean and std across all timepoints in session
                mean_and_std_dict_timepoint_avg["means"][x_value] = np.mean(session_meg_data[:,:,:])
                mean_and_std_dict_timepoint_avg["stds"][x_value] = np.std(session_meg_data[:,:,:])
            
            # Sort by key (session id or distance)
            for timepoint_idx, timepoint_dict in mean_and_std_dict_by_timepoints["timepoint"].items():
                for data_type, data_dict in timepoint_dict.items():
                    mean_and_std_dict_by_timepoints["timepoint"][timepoint_idx][data_type] = {x_value: data_dict[x_value] for x_value in sorted(data_dict, key=int)}
            for data_type, data_dict in mean_and_std_dict_timepoint_avg.items():
                mean_and_std_dict_timepoint_avg[data_type] = {x_value: data_dict[x_value] for x_value in sorted(data_dict, key=int)}

            for plot_type in ["means", "stds"]:
                measure_name = "Mean Value" if plot_type == "means" else "Standard Deviation Value"
                timepoints_ms = [self.map_timepoint_idx_to_ms(timepoint_idx) for timepoint_idx in mean_and_std_dict_by_timepoints["timepoint"].keys()]
                plt.figure(figsize=(10, 6))
                for timepoint_number, timepoint_idx in enumerate(mean_and_std_dict_by_timepoints["timepoint"]):
                    plt.plot(list(mean_and_std_dict_by_timepoints["timepoint"][timepoint_idx][plot_type].keys()), list(mean_and_std_dict_by_timepoints["timepoint"][timepoint_idx][plot_type].values()), label=f'{timepoints_ms[timepoint_number]} ms')
                plt.plot(list(mean_and_std_dict_timepoint_avg[plot_type].keys()), list(mean_and_std_dict_timepoint_avg[plot_type].values()), label=f'Average across {self.map_timepoint_idx_to_ms(0)} to {self.map_timepoint_idx_to_ms(self.timepoint_max-self.timepoint_min)}', color="black", linewidth=5)
                
                x_label_str = "Days since Session 1" if x_value_days else "Session ID"
                plt.xlabel(x_label_str)
                plt.ylabel(measure_name)
                plt.title(f'{measure_name} for each Session. \n Subject: {self.subject_id}, Norm: {normalization} {subject_02_title_addition}')
                plt.legend(title="Timepoints (ms)")  

                # Save plot
                plot_folder = f"data_files/{self.lock_event}/visualizations/meg_data/session_comparison/{plot_type}/{normalization}/"
                plot_file = f"Session_Comparison_{measure_name}_subject_{self.subject_id}.png"
                self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)

    
    def visualize_source_self_pred_pca(self, regions_of_interest:list, source_pca_type:str, plot_pcs_seperate:bool, whiten:str):
        assert source_pca_type == "voxels_and_timepoints"
        pc_storage_folder = "/pcs_separate" if plot_pcs_seperate else ""
        whiten_folder = "/whiten" if whiten else ""
        pca_folder = f"/voxels_and_timepoints_pca_reduced{whiten_folder}"

        def plot_region_self_preds_plots(session_fit_measures:dict, glaser_region:str, pca_folder:str, normalization:str):
            # Collect fit_measures for models on predictions on the own session
            self_pred_fit_measures_by_session= {"session": {}}
            for session_id in session_fit_measures['session_mapping']:
                fit_measure_session = session_fit_measures['session_mapping'][session_id]["session_pred"][session_id]
                self_pred_fit_measures_by_session["session"][session_id] = fit_measure_session

            # Plot all sessions in one plot
            session_labels = [session_id for session_id in self_pred_fit_measures_by_session.keys()]
            session_values = [session_id for session_id in self_pred_fit_measures_by_session.values()]

            plt.figure(figsize=(10, 6))
            plt.bar(session_labels, session_values, color='blue')
            plt.title(f'Variance Explained per Session, averaged across PCs (PCA over timepoints and voxels). \n {glaser_region}.')
            plt.xlabel(f'Session ID')
            plt.ylabel(f'Variance Explained')
            plt.grid(True)

            plot_folder = f"data_files/{self.lock_event}/visualizations/model_comparison/source_space/{pca_folder}/{glaser_region}/subject_{self.subject_id}/norm_{normalization}"
            plot_file = f"Model_comparison_voxels_and_timepoints_pca_reduced_{normalization}.png"
            self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
            plt.close()

        def plot_pcas_region_self_preds_plots(session_fit_measures:dict, glaser_region:str, pca_folder:str, normalization:str):
            # Collect fit_measures for models on predictions on the own session
            self_pred_fit_measures_by_session = self.recursive_defaultdict()
            for pc, pc_fit_measures in session_fit_measures["pcs"].items():
                for session_id in pc_fit_measures['session_mapping']:
                    fit_measure_pc_session = pc_fit_measures['session_mapping'][session_id]["session_pred"][session_id]
                    self_pred_fit_measures_by_session["session"][session_id][pc] = fit_measure_pc_session

            
            n_pcs = len(session_fit_measures["pcs"].keys())
            n_sessions = len(self_pred_fit_measures_by_session["session"].keys())
            session_labels = [f"Session {session_id}" for session_id in self.session_ids_num]
            pc_labels = [f"PC {pc_idx+1}" for pc_idx in range(n_pcs)]
            bar_width = 0.15

            fig, ax = plt.subplots(figsize=(10, 6))
            for session_idx, session_id in enumerate(self.session_ids_num):
                x_positions = [session_idx + pc * bar_width for pc in range(n_pcs)]
                y_positions = [session_pc_val for session_pc_val in self_pred_fit_measures_by_session["session"][session_id].values()]
                ax.bar(x_positions, y_positions, width=bar_width)  # , label=pc_labels

            ax.set_xlabel('Session ID')
            ax.set_ylabel('Variance Explained')
            ax.set_title('Model comparison PCA over Voxels and Timepoints')
            ticks_loc = [session_idx + bar_width * (n_pcs - 1)/2 for session_idx in range(n_sessions)]
            print(f"n_sessions: {n_sessions}")
            print(f"ticks_loc: {ticks_loc}")
            ax.set_xticks(ticks_loc)
            ax.set_xticklabels(session_labels)
            ax.legend()
            plt.tight_layout()

            plot_folder = f"data_files/{self.lock_event}/visualizations/model_comparison/source_space/{pca_folder}/pcs_separate/{glaser_region}/subject_{self.subject_id}/norm_{normalization}"
            plot_file = f"Model_comparison_voxels_and_timepoints_pca_reduced_{normalization}.png"
            self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
            plt.close()
            

        for normalization in self.normalizations:
            for glaser_region in regions_of_interest:
                storage_folder = f"data_files/{self.lock_event}/var_explained/source_space{pca_folder}{pc_storage_folder}/{glaser_region}/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/"
                json_storage_file = f"var_explained_dict.json"
                json_storage_path = os.path.join(storage_folder, json_storage_file)
                with open(json_storage_path, 'r') as file:
                    fit_measures_by_session = json.load(file)
                        
                if not plot_pcs_seperate:
                    plot_region_self_preds_plots(session_fit_measures=fit_measures_by_session, glaser_region=glaser_region, pca_folder=pca_folder, normalization=normalization)
                else:
                    plot_pcas_region_self_preds_plots(session_fit_measures=fit_measures_by_session, glaser_region=glaser_region, pca_folder=pca_folder, normalization=normalization)
                    # TODO: need new function, PCAs next to each other


class DebuggingHelper(VisualizationHelper):
    def __init__(self, norms:list, subject_id: str = "02"):
        super().__init__(norms, subject_id)


    def inspect_meg_data():
        pass


    def inspect_ridge_models():
        pass
