import os
import json
import h5py
import pickle
import pandas as pd
import numpy as np
import imageio
import mne
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Import Line2D for custom legend
from collections import defaultdict
from typing import Tuple, Dict

# ANN specific imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from thingsvision import get_extractor

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

class BasicOperationsHelper:
    def __init__(self, subject_id: str = "02"):
        self.subject_id = subject_id
        self.session_ids_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        self.session_ids_num = [str(session_id) for session_id in range(1,11)]


    def get_relevant_meg_channels(self, chosen_channels, subject_id, session_id, lock_event="saccade"):
        #{'grad': {}, 'mag': {194: 'MEG1731', 215: 'MEG1921', 236: 'MEG2111', 269: 'MEG2341', 284: 'MEG2511'}}
        session_id_padded = "0" + session_id if session_id != 10 else session_id
        fif_file_path = f'/share/klab/datasets/avs/population_codes/as{subject_id}/sensor/filter_0.2_200/{lock_event}_evoked_{subject_id}_{session_id_padded}_.fif'
        
        processing_channels_indices = {"grad": {}, "mag": {}}
        evoked = mne.read_evokeds(fif_file_path)[0]

        # Get indices of Grad and Mag sensors
        grad_indices = mne.pick_types(evoked.info, meg='grad')
        mag_indices = mne.pick_types(evoked.info, meg='mag')

        #print(f"grad_indices: {grad_indices}")
        #print(f"mag_indices: {mag_indices}")

        for sensor_type in processing_channels_indices: # grad, mag
            channel_indices = mne.pick_types(evoked.info, meg=sensor_type)
            sensor_index = 0
            for channel_idx in channel_indices:
                ch_name = evoked.ch_names[channel_idx]
                if int(ch_name[3:]) in chosen_channels:
                    processing_channels_indices[sensor_type][sensor_index] = ch_name
                sensor_index += 1

        print(processing_channels_indices)

        return processing_channels_indices


    def recursive_defaultdict(self) -> dict:
        return defaultdict(self.recursive_defaultdict)


    def map_session_letter_id_to_num(self, session_id_letter: str) -> str:
        """
        Helper function to map the character id from a session to its number id.
        For example: Input "a" will return "1".
        """
        # Create mapping
        session_mapping = {}
        for num in range(1,11):
            letter = self.session_ids_char[num-1]
            session_mapping[letter] = str(num)
        # Map letter to num
        session_id_num = session_mapping[session_id_letter]

        return session_id_num


    def read_dict_from_json(self, type_of_content: str, type_of_norm: str = None) -> dict:
        """
        Helper function to read json files into dicts.
        """
        valid_types = ["combined_metadata", "meg_metadata", "crop_metadata", "mse_losses", "mse_losses_timepoint", "var_explained"]
        if type_of_content not in valid_types:
            raise ValueError(f"Function read_dict_from_json called with unrecognized type {type_of_content}.")

        if type_of_content == "mse_losses":
            file_path = f"data_files/mse_losses/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{type_of_norm}/mse_losses_{type_of_norm}_dict.json"
        elif type_of_content == "mse_losses_timepoint":
            file_path = f"data_files/mse_losses/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/timepoints/norm_{type_of_norm}/mse_losses_timepoint_{type_of_norm}_dict.json"
        elif type_of_content == "var_explained":
            file_path = f"data_files/var_explained/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{type_of_norm}/var_explained_{type_of_norm}_dict.json"
        else:
            file_path = f"data_files/metadata/{type_of_content}/subject_{self.subject_id}/{type_of_content}_dict.json"
        
        try:
            with open(file_path, 'r') as data_file:
                data_dict = json.load(data_file)
            return data_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"In Function read_dict_from_json: The file {file_path} does not exist.")

        return data_dict


    def save_dict_as_json(self, type_of_content: str, dict_to_store: dict, type_of_norm: str = None) -> None:
        """
        Helper function to store dicts as json files.
        """
        valid_types = ["combined_metadata", "meg_metadata", "crop_metadata", "mse_losses", "mse_losses_timepoint", "var_explained"]
        if type_of_content not in valid_types:
            raise ValueError(f"Function save_dict_as_json called with unrecognized type {type_of_content}.")

        if type_of_content == "mse_losses" or type_of_content == "mse_losses_timepoint":
            if type_of_content == "mse_losses_timepoint":
                timepoint_folder = "timepoints/"
                timepoint_name = "_timepoints"
            else:
                timepoint_folder = ""
                timepoint_name = ""
            storage_folder = f"data_files/mse_losses/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/{timepoint_folder}norm_{type_of_norm}"
            name_addition = f"_{type_of_norm}"
        elif type_of_content == "var_explained":
            storage_folder = f"data_files/var_explained/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{type_of_norm}"
            name_addition = f"_{type_of_norm}"
        else:
            storage_folder = f'data_files/metadata/{type_of_content}/subject_{self.subject_id}'
            name_addition = ""
        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)
        json_storage_file = f"{type_of_content}{name_addition}_dict.json"
        json_storage_path = os.path.join(storage_folder, json_storage_file)

        with open(json_storage_path, 'w') as file:
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
        valid_types = ["trial_splits", "crop_data", "meg_data", "torch_dataset", "ann_features"]
        if type_of_content not in valid_types:
            raise ValueError(f"Function export_split_data_as_file called with unrecognized type {type_of_content}.")

        # Set file type
        if type_of_content == "torch_dataset":
            file_type = ".pt"
        else:
            file_type = ".npy"
        
        # Add additional folder for model type and extraction layer for ann_features
        if type_of_content == "ann_features":
            additional_model_folders = f"/{ann_model}/{module}/"
        else:
            additional_model_folders = "/"

        if type_of_content == "meg_data":
            additional_norm_folder = f"norm_{type_of_norm}/"
        else:
            additional_norm_folder = ""

        # Export train/test split arrays to .npz
        for split in array_dict:
            save_folder = f"data_files/{type_of_content}{additional_model_folders}{additional_norm_folder}subject_{self.subject_id}/session_{session_id}/{split}"  
            save_file = f"{type_of_content}{file_type}"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, save_file)

            if file_type == ".npy":
                np.save(save_path, array_dict[split])
            else:
                torch.save(array_dict[split], save_path)


    
    def load_split_data_from_file(self, session_id_num: str, type_of_content: str, type_of_norm:str = None, ann_model: str = None, module: str = None) -> dict:
        """
        Helper function to load the split for a given session.

        Parameters: 
            session_id_num (str): id of session that the arrays belong to
            type_of_content (str): Type of data in arrays. Allowed values: ["trial_splits", "crop_data", "meg_data", "torch_dataset", "ann_features"]
        """
        valid_types = ["trial_splits", "crop_data", "meg_data", "torch_dataset", "ann_features"]
        if type_of_content not in valid_types:
            raise ValueError(f"Function load_split_data_from_file called with unrecognized type {type_of_content}.")

        if type_of_content == "torch_dataset":
            file_type = ".pt"
        else:
            file_type = ".npy"

        # Add additional folder for model type and extraction layer for ann_features
        if type_of_content == "ann_features":
            additional_model_folders = f"/{ann_model}/{module}/"
        else:
            additional_model_folders = "/"

        if type_of_content == "meg_data":
            additional_norm_folder = f"norm_{type_of_norm}/"
        else:
            additional_norm_folder = ""

        split_dict = {}
        for split in ["train", "test"]:
            # Load split trial array
            split_path = f"data_files/{type_of_content}{additional_model_folders}{additional_norm_folder}subject_{self.subject_id}/session_{session_id_num}/{split}/{type_of_content}{file_type}"  
            if file_type == ".npy":
                split_data = np.load(split_path)
            else:
                split_data = torch.load(split_path)
            split_dict[split] = split_data

        return split_dict

    
    def save_plot_as_file(self, plt, plot_folder: str, plot_file: str, plot_type: str = None):
        """
        Helper function to save a plot as file.
        """
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plot_path = os.path.join(plot_folder, plot_file)
        plt.savefig(plot_path)
        # Mne plots cannot be closed
        if plot_type != "mne":
            plt.close()


    def normalize_array(self, data: np.ndarray, normalization: str, session_id: str = None):
        """
        Helper function to normalize meg
        normalization options: mean centered per channel and per timepoint, min-max over complete session, robust scaling, no normalization
                                ["min_max", "mean_centered_ch_t", "robust_scaling", "no_norm", "median_centered_ch_t"]
        """
        normalized_data = None

        match normalization: 

            case "min_max":
                data_min = data.min()
                data_max = data.max()
                normalized_data = (data - data_min) / (data_max - data_min)

            case "robust_scaling":
                medians = np.median(data, axis=None)  # Median across epochs
                q75, q25 = np.percentile(data, [75, 25], axis=None)
                iqr = q75 - q25
                normalized_data = (data - medians) / iqr  # Subtract medians and divide by IQR
                if (normalized_data == data).all():
                    raise ValueError(f"normalize_array: data the same before and after norm {normalization}")

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
                #if session_id == "1":
                #    print(f"mean_centered_ch_t normalized_data: {normalized_data}")

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
            print(f"[WARNING][normalize_array]: data the same before and after norm {normalization}")

        return normalized_data



class MetadataHelper(BasicOperationsHelper):
    def __init__(self, subject_id: str = "02", lock_event: str = "saccade"):
        super().__init__(subject_id)

        self.lock_event = lock_event
        self.crop_metadata_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_224/metadata/as{subject_id}_crops_metadata.csv"
        self.meg_metadata_folder = f"/share/klab/datasets/avs/population_codes/as{subject_id}/sensor/filter_0.2_200"


    def create_combined_metadata_dict(self, investigate_missing_data=False) -> None:
        """
        Creates the combined metadata dict with timepoints that can be found in both meg and crop metadata for the respective session and trial.
        """

        # Read crop metadata from json
        crop_metadata = self.read_dict_from_json(type_of_content="crop_metadata")
        # Read meg metadata from json
        meg_metadata = self.read_dict_from_json(type_of_content="meg_metadata")

        # Store information about timepoints that are present in both meg and crop data

        # Use defaultdict to automatically create missing keys
        combined_metadata_dict = self.recursive_defaultdict()

        meg_index = 0
        meg_missing_trials = []
        for session_id in meg_metadata["sessions"]:
            for trial_id in meg_metadata["sessions"][session_id]["trials"]:
                for timepoint_id in meg_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]:
                    # For each timepoint in the meg metadata: Check if this timepoint is in the crop metadata and if so store it
                    try:
                        crop_identifier = crop_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id].get("crop_identifier", None)
                        sceneID = crop_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id].get("sceneID", None)
                        combined_metadata_dict["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["crop_identifier"] = crop_identifier
                        combined_metadata_dict["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["sceneID"] = sceneID
                        combined_metadata_dict["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["meg_index"] = meg_index
                    except:
                        if trial_id not in meg_missing_trials and investigate_missing_data:
                            print(f"[Session {session_id}][Trial {trial_id}]: Within this Trial, data for at least one timepoint exists only in the meg-, and not the crop metadata.")
                            meg_missing_trials.append(trial_id)
                        pass
                    meg_index += 1

        if investigate_missing_data:
            crop_missing_trials = []
            # Do the same from the perspective of the crop_metadata to find datapoints that only exist in the crop-, but not the meg-metadata
            for session_id in crop_metadata["sessions"]:
                for trial_id in crop_metadata["sessions"][session_id]["trials"]:
                    for timepoint_id in crop_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]:
                        # For each timepoint in the crop metadata: Check if this timepoint is in the meg metadata and if so store it
                        try:
                            crop_identifier = meg_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["crop_identifier"]
                        except:
                            if trial_id not in crop_missing_trials:
                                print(f"[Session {session_id}][Trial {trial_id}]: Within this Trial, data for at least one timepoint exists only in the crop-, and not the meg metadata.")
                                crop_missing_trials.append(trial_id)
                            pass
            
        # Export dict to json 
        self.save_dict_as_json(type_of_content="combined_metadata", dict_to_store=combined_metadata_dict)


    def create_meg_metadata_dict(self) -> None:
        """
        Creates the meg metadata dict for the participant and stores it.
        """
        if self.lock_event == "saccade":
            time_column = "end_time" 
        else:
            time_column = "time_in_trial"


        data_dict = {"sessions": {}}
        # Read metadata for each session from csv
        for session_id_letter in self.session_ids_char:
            # Build path to session metadata file
            meg_metadata_file = f"as{self.subject_id}{session_id_letter}_et_epochs_metadata_{self.lock_event}.csv"
            meg_metadata_path = os.path.join(self.meg_metadata_folder, meg_metadata_file)

            session_id_num = self.map_session_letter_id_to_num(session_id_letter)

            data_dict["sessions"][session_id_num] = {}

            # Read metadata from csv 
            df = pd.read_csv(meg_metadata_path, delimiter=";")

            # Create ordered dict
            data_dict["sessions"][session_id_num]["trials"] = {}

            for index, row in df.iterrows():
                trial_id = int(row["trial"])
                timepoint = row[f"{time_column}"]
                
                # Create dict to store values for current trial if this is the first timepoint in the trial
                if trial_id not in data_dict["sessions"][session_id_num]["trials"].keys():
                    data_dict["sessions"][session_id_num]["trials"][trial_id] = {"timepoints": {}}

                # Store availability of current timepoint in trial
                data_dict["sessions"][session_id_num]["trials"][trial_id]["timepoints"][timepoint] = {"meg":True}

        # Export dict to json 
        self.save_dict_as_json(type_of_content="meg_metadata", dict_to_store=data_dict)


    def create_crop_metadata_dict(self) -> None:
        """
        Creates the crop metadata dict for the participant and stores it.
        """
        if self.lock_event == "saccade":
            time_column = "Start_time"
        else:
            time_column = "time_in_trial"

        # Read data from csv and set index to crop filename (crop_identifier before .png)
        df = pd.read_csv(self.crop_metadata_path, index_col = 'crop_filename')

        # Remove rows/fixations without crop identifiers
        df = df[df.index.notnull()]

        num_sessions = df["session"].max() #10 sessions for subj 2

        # Create ordered dict
        data_dict = {"sessions": {}}

        for nr_session in range(1,num_sessions+1):
            # Create dict for this session
            data_dict["sessions"][nr_session] = {}

            # Filter dataframe by session 
            session_df = df[df["session"] == nr_session]
            # Filter dataframe by type of recording (we are only interested in scene recordings, in the meg file I am using there is no data for caption of microphone recordings (or "nothing" recordings))
            session_df = session_df[session_df["recording"] == "scene"]

            # Get list of all trials in this session
            trial_numbers = list(map(int, set(session_df["trial"].tolist())))

            # Create dict for trials in this session
            data_dict["sessions"][nr_session]["trials"] = {}

            # For each trial in the session
            for nr_trial in trial_numbers:
                # Filter session dataframe by trial
                trial_df = session_df[session_df["trial"] == nr_trial]

                # Create dict for this trial
                data_dict["sessions"][nr_session]["trials"][nr_trial] = {}

                # Get list of all timepoints in this trial
                timepoints = trial_df[f"{time_column}"].tolist()

                # Create dict for timepoints in this trial
                data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"] = {}

                # For each timepoint in this trial
                for timepoint in timepoints:
                    # Filter trial dataframe by timepoint
                    timepoint_df = trial_df[trial_df[f"{time_column}"] == timepoint]

                    # get sceneID for this timepoint
                    sceneID = timepoint_df['sceneID'].iloc[0]

                    # Create dicts for this timepoint
                    data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint] = {}

                    # Fill in crop identifier value (here the index without ".png") in main data dict for this timepoint, as well as scene id and space for meg data
                    # metadata
                    data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint]["crop_identifier"] = timepoint_df.index.tolist()[0][:-4]  # slice with [0] to get single element instead of list, then slice off ".png" at the end
                    data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint]["sceneID"] = sceneID
        
        # Export dict to json 
        self.save_dict_as_json(type_of_content="crop_metadata", dict_to_store=data_dict)


        
class DatasetHelper(MetadataHelper):
    def __init__(self, normalizations:list, subject_id: str = "02", chosen_channels:list = [1731, 1921, 2111, 2341, 2511], lock_event: str = "saccade", timepoint_min: int = 50, timepoint_max:int = 250):
        super().__init__(subject_id=subject_id, lock_event=lock_event)

        self.normalizations = normalizations
        self.chosen_channels = chosen_channels
        self.timepoint_min = timepoint_min
        self.timepoint_max = timepoint_max

    def create_crop_dataset(self) -> None:
        """
        Creates the crop dataset with all crops in the combined_metadata (crops for which meg data exists)
        """

        # Read combined metadata from json
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")

        # Define path to read crops from
        crop_folder_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_224/crops/as{self.subject_id}"

        # For each session: create crop datasets based on respective splits
        for session_id in self.session_ids_num:
            # Get train/test split based on trials (based on scenes)
            trials_split_dict = self.load_split_data_from_file(session_id_num=session_id, type_of_content="trial_splits")

            crop_split = {"train": [], "test": []}
            for trial_id in combined_metadata["sessions"][session_id]["trials"]:
                # Check if trial belongs to train or test split
                if trial_id in trials_split_dict["train"]:
                    trial_type = "train"
                elif trial_id in trials_split_dict["test"]:
                    trial_type = "test"
                else:
                    raise ValueError(f"Trial_id {trial_id} neither in train nor test split.")
                for timepoint_id in combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]:
                    # Extract image for every epoch/fixation (if there is both crop and metadata for the timepoint)
                    if timepoint_id in combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]:
                        # Get crop path
                        crop_filename = ''.join([combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["crop_identifier"], ".png"])
                        crop_path = os.path.join(crop_folder_path, crop_filename)

                        # Read crop as array and concat
                        crop = imageio.imread(crop_path)
                        crop_split[trial_type].append(crop)
            # Convert to numpy array
            for split in crop_split:
                crop_split[split] = np.stack(crop_split[split], axis=0)

            # Export numpy array to .npz
            self.export_split_data_as_file(session_id=session_id, 
                                        type_of_content="crop_data",
                                        array_dict=crop_split)


    def create_meg_dataset(self) -> None:
        """
        Creates the crop dataset with all crops in the combined_metadata (crops for which meg data exists)
        """

        # Read combined and meg metadata from json
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")
        meg_metadata = self.read_dict_from_json(type_of_content="meg_metadata")

        meg_data_folder = f"/share/klab/datasets/avs/population_codes/as{self.subject_id}/sensor/filter_0.2_200"

        for session_id_char in self.session_ids_char:
            session_id_num = self.map_session_letter_id_to_num(session_id_char)
            print(f"Creating meg dataset for session {session_id_num}")
            # Load session MEG data from .h5
            meg_data_file = f"as{self.subject_id}{session_id_char}_population_codes_{self.lock_event}_500hz_masked_False.h5"
            with h5py.File(os.path.join(meg_data_folder, meg_data_file), "r") as f:
                meg_data = {}
                meg_data["grad"] = f['grad']['onset']  # shape participant 2, session a saccade: (2945, 204, 401), fixation: (2874, 204, 601) 
                meg_data["mag"] = f['mag']['onset']  # shape participant 2, session a saccade: (2945, 102, 401), fixation: (2874, 102, 601)

                num_meg_timepoints = meg_data['grad'].shape[0]
                print(f"[Session {session_id_num}]: Pre filtering: meg_data['grad'].shape: {meg_data['grad'].shape}")
                print(f"[Session {session_id_num}]: Pre filtering: meg_data['mag'].shape: {meg_data['mag'].shape}")


                # Select relevant channels
                channel_indices = self.get_relevant_meg_channels(chosen_channels=self.chosen_channels, subject_id=self.subject_id, session_id=session_id_num, lock_event="saccade")

                grad_selected = False
                mag_selected = False
                if channel_indices['grad']:
                    grad_selected = True
                    meg_data["grad"] = meg_data["grad"][:, list(channel_indices['grad'].keys()),:]
                if channel_indices['mag']:
                    mag_selected = True
                    meg_data["mag"] = meg_data["mag"][:, list(channel_indices['mag'].keys()),:]

                # Cut relevant timepoints. Range is -0.5 – 0.3 s. We want timepoints 50-250
                if self.timepoint_min is not None and self.timepoint_max is not None:
                    if grad_selected:
                        meg_data["grad"] = meg_data["grad"][:,:,self.timepoint_min:self.timepoint_max+1]
                    if mag_selected:
                        meg_data["mag"] = meg_data["mag"][:,:,self.timepoint_min:self.timepoint_max+1]
                    if not grad_selected and not mag_selected:
                        raise ValueError("Neither mag or grad channels selected.")
                

                # Create datasets based on specified normalizations
                for normalization in self.normalizations:
                    # Debugging
                    #if session_id_num == "1" and normalization == "no_norm":
                    if grad_selected:
                        print(f"[Session {session_id_num}]: Post filtering: meg_data['grad'].shape: {meg_data['grad'].shape}")
                    if mag_selected:
                        print(f"[Session {session_id_num}]: Post filtering: meg_data['mag'].shape: {meg_data['mag'].shape}")

                    meg_data_norm = {}
                    # Normalize grad and mag independently
                    if grad_selected:
                        meg_data_norm["grad"] = self.normalize_array(np.array(meg_data['grad']), normalization=normalization, session_id=session_id_num)
                    if mag_selected:
                        meg_data_norm["mag"] = self.normalize_array(np.array(meg_data['mag']), normalization=normalization, session_id=session_id_num)

                    # Combine grad and mag data
                    if grad_selected and mag_selected:
                        combined_meg = np.concatenate([meg_data_norm["grad"], meg_data_norm["mag"]], axis=1) #(2874, 306, 601)
                    elif grad_selected:
                        combined_meg = meg_data_norm["grad"]
                    elif mag_selected:
                        combined_meg = meg_data_norm["mag"]

                    print(f"[Session {session_id_num}]: After normalization and combination: combined_meg.shape: {combined_meg.shape}")

                    # Split meg data 
                    # Get train/test split based on trials (based on scenes)
                    trials_split_dict = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="trial_splits")

                    meg_split = {"train": [], "test": []}

                    # Debugging: Count timepoints in meg metadata
                    num_meg_metadata_timepoints = 0
                    for trial_id in meg_metadata["sessions"][session_id_num]["trials"]:
                        for timepoint in meg_metadata["sessions"][session_id_num]["trials"][trial_id]["timepoints"]:
                            num_meg_metadata_timepoints += 1
                    print(f"[Session {session_id_num}]: Timepoints in meg metadata: {num_meg_metadata_timepoints}")
                    #if num_meg_timepoints != num_meg_metadata_timepoints:
                    #    raise ValueError(f"The number of datapoints in the meg data and in the meg metadata is not identical. Dataset: {num_meg_timepoints}. Metadata: {num_meg_metadata_timepoints}") 

                    # Debugging: Count timepoints in combined metadata
                    num_combined_metadata_timepoints = 0
                    for trial_id in combined_metadata["sessions"][session_id_num]["trials"]:
                        for timepoint in combined_metadata["sessions"][session_id_num]["trials"][trial_id]["timepoints"]:
                            num_combined_metadata_timepoints += 1
                    print(f"[Session {session_id_num}]: Timepoints in combined metadata: {num_combined_metadata_timepoints}")


                    # Iterate through meg metadata for simplicity with indexing
                    index = 0
                    for trial_id in meg_metadata["sessions"][session_id_num]["trials"]:
                        if trial_id in trials_split_dict["train"]:
                            trial_type = "train"
                        elif trial_id in trials_split_dict["test"]:
                            trial_type = "test"
                        else:
                            # Raise error if the trial is in neither split, but is in the combined metadata (in this case it should be there)
                            if trial_id in combined_metadata["sessions"][session_id_num]["trials"]:
                                raise ValueError(f"Session {session_id_num}: Trial_id {trial_id} neither in train nor test split.")
                            # Otherwise skip the data in this trial (there is no corresponding crop data for the meg trial data)
                            else:
                                for timepoint in meg_metadata["sessions"][session_id_num]["trials"][trial_id]["timepoints"]:
                                    index += 1
                                print(f"[Session {session_id_num}]: Trial {trial_id} from meg metadata is not in combined metadata.")
                                continue
                        for timepoint in meg_metadata["sessions"][session_id_num]["trials"][trial_id]["timepoints"]:
                            # Check if there is both crop and metadata for the timepoint
                            if timepoint in combined_metadata["sessions"][session_id_num]["trials"][trial_id]["timepoints"]:
                                # Assign to split
                                meg_split[trial_type].append(combined_meg[index])
                            # Advance index
                            index += 1
                    
                    print(f"[Session {session_id_num}]: Index after dataset creation: {index}")

                    # Export meg dataset arrays to .npz
                    self.export_split_data_as_file(session_id=session_id_num, 
                                                type_of_content="meg_data",
                                                array_dict=meg_split,
                                                type_of_norm=normalization)


    def create_train_test_split(self):
        """
        Creates train/test split of trials based on scene_ids.
        """
        # Read combined metadata from json
        combined_metadata = self.read_dict_from_json(type_of_content="combined_metadata")

        # Prepare splits for all sessions: count scenes
        scene_ids = {session_id: {} for session_id in combined_metadata["sessions"]}
        trial_ids = {session_id: [] for session_id in combined_metadata["sessions"]}
        num_datapoints = {session_id: 0 for session_id in combined_metadata["sessions"]}
        for session_id in combined_metadata["sessions"]:
            for trial_id in combined_metadata["sessions"][session_id]["trials"]:
                # Store trials in session for comparison with scenes
                trial_ids[session_id].append(trial_id)
                for timepoint in combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"]:
                    # get sceneID of current timepoint
                    scene_id_current = combined_metadata["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint]["sceneID"]
                    # First occurance of the scene? Store its occurange with the corresonding trial
                    if scene_id_current not in scene_ids[session_id]:
                        scene_ids[session_id][scene_id_current] = {"trials": [trial_id]}
                    # SceneID previously occured in another trial, but this trial is not stored yet: store it under the scene_id aswell
                    elif trial_id not in scene_ids[session_id][scene_id_current]["trials"]:
                        scene_ids[session_id][scene_id_current]["trials"].append(trial_id)

                    # Count total datapoints/timepoints/fixations in combined metadata in session
                    num_datapoints[session_id] += 1

        # Create splits for each session
        for session_id in combined_metadata["sessions"]:
            # Debugging: Identical number of trials and scenes in this session?
            num_scenes = len(scene_ids[session_id])  # subject 2, session a: 300 
            num_trials = len(trial_ids[session_id])
            assert num_scenes == num_trials, f"Session {session_id}: Number of trials and number of scenes is not identical. Doubled scenes need to be considered"

            # Choose scene_ids for 80/20 split
            num_trials_train = int(num_trials*0.8)
            num_trials_test = num_trials - num_trials_train

            # Split based on scene ids, but trial information is sufficient to identify datapoint
            train_split_trials = []
            test_split_trials = []
            # Create split sceneID (by ordering the trials by scenes, trials that belong to the same scene will most likely be in the same split)
            index = 0
            for scene_id in scene_ids[session_id]:
                for trial_id in scene_ids[session_id][scene_id]["trials"]:
                    # Each unique combination is one point in the dataset
                    if index < num_trials_train:
                        train_split_trials.append(trial_id)
                    else:
                        test_split_trials.append(trial_id)
                    index += 1

            # To-do: Make sure that scenes only double in the same split

            # Debugging
            """
            if session_id == "1":
                print(f"Session {session_id}: len train_split_trials: {len(train_split_trials)}")
                print(f"Session {session_id}: len test_split_trials: {len(test_split_trials)}")
                print(f"Session {session_id}: Total number of trials in train+test dataset: {len(train_split_trials) + len(test_split_trials)}")
                print(f"Session {session_id}: Total number of trials in session: {num_trials}")
                print(f"Session {session_id}: Total number of datapoints in session: {num_datapoints[session_id]}")
            """

            # Export trial_split arrays to .npz
            split_dict = {"train": train_split_trials, "test": test_split_trials}
            self.export_split_data_as_file(session_id=session_id, 
                                        type_of_content="trial_splits",
                                        array_dict=split_dict)


    def create_pytorch_dataset(self):
        """
        Creates pytorch datasets from numpy image datasets.
        """
        
        for session_id_num in self.session_ids_num:
            # Load numpy datasets for session
            crop_ds = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="crop_data")

            # Create torch datasets with helper 
            torch_dataset = {}
            torch_dataset["train"] = DatasetHelper.TorchDatasetHelper(crop_ds['train'])
            torch_dataset["test"] = DatasetHelper.TorchDatasetHelper(crop_ds['test'])

            # Store datasets
            self.export_split_data_as_file(session_id=session_id_num, type_of_content="torch_dataset", array_dict=torch_dataset)


    class TorchDatasetHelper(Dataset):
        """
        Inner class to create Pytorch dataset from numpy arrays (images).
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
            image = self.numpy_array[idx]

            if self.transform:
                image = self.transform(image)

            # Since no labels are associated, we only return the image
            return image



class ExtractionHelper(BasicOperationsHelper):
    def __init__(self, subject_id: str = "02", ann_model: str = "Resnet50", module_name : str = "fc", batch_size: int = 32):
        super().__init__(subject_id)

        self.ann_model = ann_model
        self.module_name = module_name  # Name of Layer to extract features from
        self.batch_size = batch_size

    
    def extract_features(self):
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

        for session_id_num in self.session_ids_num:
            # Load torch datasets for session
            torch_crop_ds = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="torch_dataset")

            # Create a DataLoader to handle batching
            model_input = {}
            model_input["train"] =  DataLoader(torch_crop_ds["train"], batch_size=self.batch_size, shuffle=False)
            model_input["test"] = DataLoader(torch_crop_ds["test"], batch_size=self.batch_size, shuffle=False)
    
            features_split = {}
            for split in ["train", "test"]:
                # Extract features
                features_split[split] = extractor.extract_features(
                    batches=model_input[split],
                    module_name=self.module_name,
                    flatten_acts=True  # flatten 2D feature maps from convolutional layer
                )

                # Debugging
                if session_id_num == "1":
                    print(f"Session {session_id_num}: {split}_features.shape: {features_split[split].shape}")

            # Export numpy array to .npz
            self.export_split_data_as_file(session_id=session_id_num, type_of_content="ann_features", array_dict=features_split, ann_model=self.ann_model, module=self.module_name)



class GLMHelper(DatasetHelper, ExtractionHelper):
    def __init__(self, norms: list, subject_id: str = "02", chosen_channels: list = [1731, 1921, 2111, 2341, 2511]):
        DatasetHelper.__init__(self, normalizations=norms, subject_id=subject_id, chosen_channels=chosen_channels)
        ExtractionHelper.__init__(self, subject_id=subject_id)


    def train_mapping(self):
        """
        Trains a mapping from ANN features to MEG data over all sessions.
        """
        for session_id_num in self.session_ids_num:
            print(f"Training mapping for session {session_id_num}")
            for normalization in self.normalizations:
                # Get ANN features for session
                ann_features = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="ann_features", ann_model=self.ann_model, module=self.module_name)

                # Get MEG data for sesssion
                meg_data = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="meg_data", type_of_norm=normalization)

                X_train, Y_train = ann_features['train'], meg_data['train']

                # Initialize Helper class
                ridge_model = GLMHelper.MultiDimensionalRidge(alpha=0.5)

                # Fit model on train data
                ridge_model.fit(X_train, Y_train)

                # Store trained models as pickle
                save_folder = f"data_files/GLM_models/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/session_{session_id_num}"  
                save_file = "GLM_models.pkl"
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = os.path.join(save_folder, save_file)

                with open(save_path, 'wb') as file:
                    pickle.dump(ridge_model.models, file)

        
    def predict_from_mapping(self, store_timepoint_based_losses=False):
        """
        Based on the trained mapping for each session, predicts MEG data over all sessions from their respective test features.
        """
        # Debugging
        ridge_models_session_1 = []

        for normalization in self.normalizations:
            print(f"Predicting from mapping for normalization {normalization}")
            variance_explained_dict =self.recursive_defaultdict()
            mse_session_losses = {"session_mapping": {}}
            for session_id_model in self.session_ids_num:
                mse_session_losses["session_mapping"][session_id_model] = {"session_pred": {}}
                # Get trained ridge regression model for this session
                # Load ridge model
                storage_folder = f"data_files/GLM_models/{self.ann_model}/{self.module_name}/subject_{self.subject_id}/norm_{normalization}/session_{session_id_model}"  
                storage_file = "GLM_models.pkl"
                storage_path = os.path.join(storage_folder, storage_file)
                with open(storage_path, 'rb') as file:
                    ridge_models = pickle.load(file)

                # Debugging
                if session_id_model == 1:
                    assert ridge_models not in ridge_models_session_1, "Ridge models doubled."
                    ridge_models_session_1.append(ridge_models)
                
                # Initialize MultiDim GLM class with stored models
                ridge_model = GLMHelper.MultiDimensionalRidge(alpha=0.5, models=ridge_models)

                # Generate predictions for test features over all sessions and evaluate them 
                for session_id_pred in self.session_ids_num:
                    # Get ANN features and MEG data for session where predictions are to be evaluated
                    ann_features = self.load_split_data_from_file(session_id_num=session_id_pred, type_of_content="ann_features", ann_model=self.ann_model, module=self.module_name)
                    meg_data = self.load_split_data_from_file(session_id_num=session_id_pred, type_of_content="meg_data", type_of_norm=normalization)
                    X_test, Y_test = ann_features['test'], meg_data['test']

                    # Generate predictions
                    predictions = ridge_model.predict(X_test)

                    if store_timepoint_based_losses:
                        mse_session_losses["session_mapping"][session_id_model]["session_pred"][session_id_pred] = {"timepoint":{}}
                        # Calculate loss seperately for each timepoint/model
                        n_timepoints = predictions.shape[2]
                        for t in range(n_timepoints):
                            mse_timepoint = mean_squared_error(Y_test[:,:,t].reshape(-1), predictions[:,:,t].reshape(-1))
                            # Save loss
                            mse_session_losses["session_mapping"][session_id_model]["session_pred"][session_id_pred]["timepoint"][str(t)] = mse_timepoint
                    else:
                        # Calculate the mean squared error across all flattened features and timepoints
                        mse = mean_squared_error(Y_test.reshape(-1), predictions.reshape(-1))

                        # Calculate variance explained 
                        var_explained = explained_variance_score(Y_test.reshape(-1), predictions.reshape(-1))

                        # Save loss and variance explained
                        mse_session_losses["session_mapping"][session_id_model]["session_pred"][session_id_pred] = mse
                        variance_explained_dict["session_mapping"][session_id_model]["session_pred"][session_id_pred] = var_explained

            # Store loss dict
            if store_timepoint_based_losses:
                mse_type_of_content = "mse_losses_timepoint"
            else:
                mse_type_of_content = "mse_losses"

            self.save_dict_as_json(type_of_content=mse_type_of_content, dict_to_store=mse_session_losses, type_of_norm=normalization)
            self.save_dict_as_json(type_of_content="var_explained", dict_to_store=variance_explained_dict, type_of_norm=normalization)


    class MultiDimensionalRidge:
        """
        Inner class to apply Ridge Regression over all timepoints. Enables training and prediction, as well as initialization of random weights for baseline comparison.
        """
        def __init__(self, alpha=0.5, models=[], random_weights=False):
            self.alpha = alpha
            self.random_weights = random_weights
            self.models = models  # Standardly initialized as empty list, otherwise with passed, previously trained models

        def fit(self, X=None, Y=None):
            n_features = X.shape[1]
            n_sensors = Y.shape[1]
            n_timepoints = Y.shape[2]
            self.models = [Ridge(alpha=self.alpha) for _ in range(n_timepoints)]
            if self.random_weights:
                # Randomly initialize weights and intercepts
                for model in self.models:
                    model.coef_ = np.random.rand(n_sensors, n_features) - 0.5  # Random weights centered around 0
                    model.intercept_ = np.random.rand(n_sensors) - 0.5  # Random intercepts centered around 0
            else:
                for t in range(n_timepoints):
                    Y_t = Y[:, :, t]
                    self.models[t].fit(X, Y_t)

        def predict(self, X):
            n_samples = X.shape[0]
            n_sensors = self.models[0].coef_.shape[0]
            n_timepoints = len(self.models)
            predictions = np.zeros((n_samples, n_sensors, n_timepoints))
            for t, model in enumerate(self.models):
                if self.random_weights:
                    # Use the random weights and intercept to predict; we are missing configurations implicitly achieved when calling .fit()
                    predictions[:, :, t] = X @ model.coef_.T + model.intercept_
                else:
                    predictions[:, :, t] = model.predict(X)
            return predictions    
    


class VisualizationHelper(GLMHelper):
    def __init__(self, norms:list, subject_id: str = "02"):
        super().__init__(norms=norms, subject_id=subject_id)


    def visualize_GLM_results(self, by_timepoints:bool = False, only_distance:bool = False, separate_plots:bool = False):
        """
        Visualizes results from GLMHelper.predict_from_mapping
        """
        print(f"self.normalizations: {self.normalizations}")
        if not by_timepoints:
            type_of_content = "mse_losses"
        else:
            type_of_content = "mse_losses_timepoint"
        mse_losses_norms = {}
        for normalization in self.normalizations:
            # Load loss dict
            mse_session_losses = self.read_dict_from_json(type_of_content=type_of_content, type_of_norm=normalization)
            mse_losses_norms[normalization] = mse_session_losses


            if only_distance:
                # Plot loss as a function of distance of predicted session from "training" session
                fig, ax1 = plt.subplots(figsize=(12, 8))

                # Iterate over each training session
                losses_by_distances = {}
                for train_session, data in mse_session_losses['session_mapping'].items():
                    # Calculate distance and collect corresponding losses
                    for pred_session, mse in data['session_pred'].items():
                        if train_session != pred_session:
                            distance = abs(int(train_session) - int(pred_session))
                            if distance not in losses_by_distances:
                                losses_by_distances[distance] = {"loss": mse, "num_losses": 1}
                            else:
                                losses_by_distances[distance]["loss"] += mse
                                losses_by_distances[distance]["num_losses"] += 1

                # Calculate average losses over distances
                avg_losses = {}
                num_datapoints = {}
                for distance in range(1,10):
                    avg_losses[distance] = losses_by_distances[distance]["loss"] / losses_by_distances[distance]["num_losses"]
                    num_datapoints[distance] = losses_by_distances[distance]["num_losses"]


                # Plot
                ax1.plot(avg_losses.keys(), avg_losses.values(), marker='o', linestyle='-', label=f'Average loss')
                ax1.set_xlabel('Distance between "train" and "test" Session')
                ax1.set_ylabel('Mean Squared Error')
                ax1.tick_params(axis='y', labelcolor='b')
                ax1.grid(True)
        
                # Add secondary y-axis for datapoints
                ax2 = ax1.twinx()
                ax2.plot(num_datapoints.keys(), num_datapoints.values(), 'r--', label='Number of datapoints/losses averaged')
                ax2.set_ylabel('Number of Datapoints', color='r')
                ax2.tick_params(axis='y', labelcolor='r')

                # Add a legend with all labels
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels + labels2, loc='upper right')
                
                ax1.set_title(f'MSE vs Distance for Predictions Averaged Across all Sessions with Norm {normalization}')
                plt.grid(True)
                plt.show()

                # Save the plot to a file
                plot_folder = f"data_files/visualizations/only_distance/subject_{self.subject_id}/norm_{normalization}"
                plot_file = f"MSE_plot_over_distance_{normalization}.png"
                self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)

                print(f"avg_losses after plot {normalization}: {avg_losses}")

            elif not by_timepoints:
                # Collect self-prediction MSEs for baseline and prepare average non-self-MSE calculation
                self_prediction_mses = {}
                average_prediction_mses = {}
                for session in mse_session_losses['session_mapping']:
                    self_prediction_mses[session] = mse_session_losses['session_mapping'][session]['session_pred'][session]
                    average_prediction_mses[session] = 0

                # Calculate average MSE for each session from all other training sessions
                for train_session, data in mse_session_losses['session_mapping'].items():
                    for pred_session, mse in data['session_pred'].items():
                        if train_session != pred_session:
                            average_prediction_mses[pred_session] += mse
                num_other_sessions = len(self.session_ids_num) - 1
                for session in average_prediction_mses:
                    average_prediction_mses[session] /= num_other_sessions

                if separate_plots:
                    # Generate separate plots for each training session
                    for train_session, data in mse_session_losses['session_mapping'].items():
                        plt.figure(figsize=(10, 6))
                        sessions = list(data['session_pred'].keys())
                        losses = list(data['session_pred'].values())
                        plt.plot(sessions, losses, marker='o', linestyle='-', label=f'Training Session {train_session}')
                        plt.plot(self_prediction_mses.keys(), self_prediction_mses.values(), 'r--', label='Self-prediction MSE')
                        plt.plot(average_prediction_mses.keys(), average_prediction_mses.values(), 'g-.', label='Average Non-self-prediction MSE')

                        plt.title(f'MSE for Predictions from Training Session {train_session}')
                        plt.xlabel('Predicted Session')
                        plt.ylabel('Mean Squared Error')
                        plt.legend()
                        plt.grid(True)

                        # Save the plot to a file
                        plot_folder = f"data_files/visualizations/seperate_plots_{separate_plots}/subject_{self.subject_id}/norm_{normalization}"
                        plot_file = f"MSE_plot_session{train_session}.png"
                        self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
                else:
                    # Generate a single plot with all training sessions
                    plt.figure(figsize=(12, 8))
                    for train_session, data in mse_session_losses['session_mapping'].items():
                        sessions = list(data['session_pred'].keys())
                        losses = list(data['session_pred'].values())
                        plt.plot(sessions, losses, marker='o', linestyle='-', label=f'Trained on Session {train_session}')
                    plt.plot(self_prediction_mses.keys(), self_prediction_mses.values(), 'r--', label='Self-prediction MSE')
                    plt.plot(average_prediction_mses.keys(), average_prediction_mses.values(), 'g-.', label='Average Non-self-prediction MSE')
                    
                    plt.title('MSE for Predictions Across All Sessions')
                    plt.xlabel('Prediction Session')
                    plt.ylabel('Mean Squared Error')
                    plt.legend()
                    plt.grid(True)

                    # Save the plot to a file
                    plot_folder = f"data_files/visualizations/seperate_plots_{separate_plots}/subject_{self.subject_id}/norm_{normalization}"
                    plot_file = f"MSE_plot_all_sessions.png"
                    self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
            elif by_timepoints:
                # Collect losses for timepoint models on predictions on the own session
                losses_by_session_by_timepoint = {"session": {}}
                for session_id in mse_session_losses['session_mapping']:
                    losses_by_session_by_timepoint["session"][session_id] = {"timepoint":{}}
                    for timepoint in mse_session_losses['session_mapping'][session_id]["session_pred"][session_id]["timepoint"]:
                        mse_timepoint = mse_session_losses['session_mapping'][session_id]["session_pred"][session_id]["timepoint"][timepoint]
                        losses_by_session_by_timepoint["session"][session_id]["timepoint"][timepoint] = mse_timepoint

                # Plot results averaged over all sessions
                num_timepoints = len([timepoint for timepoint in losses_by_session_by_timepoint["session"]["1"]["timepoint"]])
                timepoint_average_losses = {}
                for timepoint in range(num_timepoints):
                    # Calculate average over all within session predictions for this timepoint
                    losses = []
                    for session in losses_by_session_by_timepoint["session"]:
                        timepoint_session_loss = losses_by_session_by_timepoint["session"][session]["timepoint"][str(timepoint)]
                        losses.append(timepoint_session_loss)
                    avg_loss = np.sum(losses) / len(losses)
                    timepoint_average_losses[timepoint] = avg_loss
                timepoint_avg_loss_list = [timepoint_average_losses[timepoint] for timepoint in timepoint_average_losses]

                plt.figure(figsize=(10, 6))
                plt.bar(list(range(num_timepoints)), timepoint_avg_loss_list, color='blue')
                plt.title('MSE Loss per Timepoint Model. Averaged across all Sessions, predicting themselves.')
                plt.xlabel('Timepoints')
                plt.ylabel('MSE Loss')
                plt.grid(True)

                # Save the plot to a file
                plot_folder = f"data_files/visualizations/timepoint_model_comparison/subject_{self.subject_id}/norm_{normalization}"
                plot_file = f"MSE_timepoint_comparison_{normalization}.png"
                self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)
            else:
                raise ValueError("[ERROR][visualize_GLM_results] Function called with invalid parameter configuration.")

        if not by_timepoints:
            mse_losses_new = {}
            for norm, loss in mse_losses_norms.items():
                if loss in mse_losses_new.values():
                    # get key
                    for key, value in mse_losses_new.items():
                        if value == loss:
                            norm_new = key
                    print(f"Same loss for norm {norm} and norm {norm_new}")  # raise ValueError()

                mse_losses_new[norm] = loss

    
    def visualize_meg_epochs_mne(self):
        """
        Visualizes meg data at various processing steps
        """
        for session_id_num in self.session_ids_num:
            # Load meg data and split into grad and mag
            meg_data = self.load_split_data_from_file(session_id_num=session_id_num, type_of_content="meg_data")
            meg_data = meg_data["test"]
            meg_dict = {"grad": {"meg": meg_data[:,:204,:], "n_sensors": 204}, "mag": {"meg": meg_data[:,204:,:], "n_sensors": 102}}
            

            print(f"meg_data.shape: {meg_data.shape}")
            print(f"meg_dict['grad']['meg'].shape: {meg_dict['grad']['meg'].shape}")
            print(f"meg_dict['mag']['meg'].shape: {meg_dict['mag']['meg'].shape}")

            print(f"len(range(204)): {len(range(204))}")

            for sensor_type in meg_dict:
                # Read in with mne
                meg_info = mne.create_info(ch_names=[str(sensor_nr) for sensor_nr in range(meg_dict[sensor_type]["n_sensors"])], sfreq=500, ch_types=sensor_type)  # Create minimal Info objects with default values
                epochs = mne.EpochsArray(meg_dict[sensor_type]["meg"], meg_info)

                # Plot
                epochs_plot = epochs.plot()
                plot_folder = f"data_files/visualizations/meg_data/subject_{self.subject_id}/session_{session_id_num}/{sensor_type}"
                plot_file = f"{sensor_type}_plot.png"
                self.save_plot_as_file(plt=epochs_plot, plot_folder=plot_folder, plot_file=plot_file, plot_type="mne")


    def visualize_meg_ERP_style(self, plot_norms: list = ["mean_centered_ch_t"]):
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
                    #print(f"averaged_data.shape: {averaged_data.shape}")
                    # Store data in session dict
                    session_dict["norm"][normalization]["sensor_type"][sensor_type] = averaged_data
                    #print(f"session_dict['norm'][norm]['sensor_type'][sensor_type].shape: {session_dict['norm'][normalization]['sensor_type'][sensor_type].shape}")

            for sensor_type in ["grad", "mag"]:
                timepoints = np.array(list(range(session_dict["norm"][norm]["sensor_type"][sensor_type].shape[0])))
                raise ValueError(f"timepoints is {timepoints}, shape is {session_dict['norm'][norm]['sensor_type'][sensor_type].shape} debug this if necessary, else delete this Error raise.")

                # Plotting
                plt.figure(figsize=(10, 6))
                for norm in plot_norms:
                    if sensor_type in session_dict["norm"][norm]["sensor_type"]:
                        plt.plot(timepoints, session_dict["norm"][norm]["sensor_type"][sensor_type], label=f'{norm}')
                        #print(f"session_dict['norm'][norm]['sensor_type'][sensor_type].shape: {session_dict['norm'][norm]['sensor_type'][sensor_type].shape}")
                    else:
                        print(f"Wrong key combination: {norm} and {sensor_type}")

                plt.xlabel('Timepoints)')
                plt.ylabel('Average MEG Value')
                plt.title(f'ERP-like Average MEG Signal over Epochs and Sensors. Session {session_id_num} Sensor {sensor_type}')
                plt.legend()

                # Save plot
                plot_folder = f"data_files/visualizations/meg_data/ERP_like/{sensor_type}_combined-norms"
                plot_file = f"Session-{session_id_num}_Sensor-{sensor_type}_plot.png"
                self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)

    def visualize_model_perspective(self, plot_norms: list = ["mean_centered_ch_t"], seperate_plots=False):
        """
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
                    #print(f"session_dict['norm'][norm]['sensor_type'][sensor_type].shape: {session_dict['norm'][normalization]['sensor_type'][sensor_type].shape}")

            for sensor_type in ["grad", "mag"]:
                timepoints = np.array(list(range(session_dict["norm"][norm]["sensor_type"][sensor_type].shape[0])))
                raise ValueError(f"timepoints is {timepoints}, shape is {session_dict['norm'][norm]['sensor_type'][sensor_type].shape} debug this if necessary, else delete this Error raise.")
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
                        plot_folder = f"data_files/visualizations/meg_data/regression_model_perspective/{norm}/{sensor_type}"
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
                    plot_folder = f"data_files/visualizations/meg_data/regression_model_perspective/{sensor_type}"
                    plot_file = f"Session-{session_id_num}_Sensor-{sensor_type}_timepoint-overview.png"
                    self.save_plot_as_file(plt=plt, plot_folder=plot_folder, plot_file=plot_file)



class DebuggingHelper(VisualizationHelper):
    def __init__(self, norms:list, subject_id: str = "02"):
        super().__init__(norms, subject_id)


    def inspect_meg_data():
        pass


    def inspect_ridge_models():
        pass
