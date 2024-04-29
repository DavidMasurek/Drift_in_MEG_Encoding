import pandas as pd
import numpy as np
import json
import os
from collections import defaultdict

class BasicOperationsHelper:
    def __init__(self, subject_id: str = "02"):
        self.subject_id = subject_id

    def read_dict_from_json(self, type_of_content: str) -> dict:
        """
        Helper function to read json files into dicts.

        Allowed Parameters: "combined_metadata", "meg_metadata", "crop_metadata"
        """
        valid_types = ["combined_metadata", "meg_metadata", "crop_metadata"]
        if type_of_content not in valid_types:
            raise ValueError(f"Function read_dict_from_json called with unrecognized type {type_of_content}.")

        file_path = f"data_files/metadata/{type_of_content}/subject_{self.subject_id}/{type_of_content}_dict.json"
        try:
            with open(file_path, 'r') as metadata_file:
                metadata_dict = json.load(metadata_file)
            return metadata_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"In Function read_dict_from_json: The file {file_path} does not exist.")

        return metadata_dict


    def save_dict_as_json(self, type_of_content: str, metadata_dict: dict) -> None:
        """
        Helper function to store dicts as json files.

        Allowed Parameters: "combined_metadata", "meg_metadata", "crop_metadata"
        """
        storage_folder = f'data_files/metadata/{type_of_content}/subject_{self.subject_id}'
        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)
        json_storage_file = f"{type_of_content}_dict.json"
        json_storage_path = os.path.join(storage_folder, json_storage_file)

        with open(json_storage_path, 'w') as file:
            # Serialize and save the dictionary to the file
            json.dump(metadata_dict, file, indent=4)

class MetadataHelper(BasicOperationsHelper):
    def __init__(self, subject_id: str = "02"):
        super().__init__(subject_id)
        self.session_ids_num = list(range(1, 11))
        self.session_ids_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        
        self.session_ids_num = [session_id for session_id in range(1,11)]
        self.session_ids_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

        self.crop_metadata_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_224/metadata/as{subject_id}_crops_metadata.csv"
        self.meg_metadata_folder = f"/share/klab/datasets/avs/population_codes/as{subject_id}/sensor/filter_0.2_200"


    def recursive_defaultdict(self) -> dict:
        return defaultdict(self.recursive_defaultdict)


    def map_session_letter_id_to_num(self, session_id_letter: str) -> str:
        # Create mapping
        session_mapping = {}
        for num in range(1,11):
            letter = self.session_ids_char[num-1]
            session_mapping[letter] = str(num)
        # Map letter to num
        session_id_num = session_mapping[session_id_letter]

        return session_id_num


    def create_combined_metadata_dict(self) -> None:
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
                        pass
                    meg_index += 1
        
        # Export dict to json 
        self.save_dict_as_json(type_of_content="combined_metadata", metadata_dict=combined_metadata_dict)


    def create_meg_metadata_dict(self) -> None:
        """
        Creates the meg metadata dict for the participant and stores it.
        """

        data_dict = {"sessions": {}}
        # Read metadata for each session from csv
        for session_id_letter in self.session_ids_char:
            # Build path to session metadata file
            meg_metadata_file = f"as{self.subject_id}{session_id_letter}_et_epochs_metadata_fixation.csv"
            meg_metadata_path = os.path.join(self.meg_metadata_folder, meg_metadata_file)

            session_id_num = self.map_session_letter_id_to_num(session_id_letter)

            data_dict["sessions"][session_id_num] = {}

            # Read metadata from csv 
            df = pd.read_csv(meg_metadata_path, delimiter=";")

            # Create ordered dict
            data_dict["sessions"][session_id_num]["trials"] = {}

            for index, row in df.iterrows():
                trial_id = int(row["trial"])
                timepoint = row["time_in_trial"]
                
                # Create dict to store values for current trial if this is the first timepoint in the trial
                if trial_id not in data_dict["sessions"][session_id_num]["trials"].keys():
                    data_dict["sessions"][session_id_num]["trials"][trial_id] = {"timepoints": {}}

                # Store availability of current timepoint in trial
                data_dict["sessions"][session_id_num]["trials"][trial_id]["timepoints"][timepoint] = {"meg":True}

        # Export dict to json 
        self.save_dict_as_json(type_of_content="meg_metadata", metadata_dict=data_dict)


    def create_crop_metadata_dict(self) -> None:
        """
        Creates the crop metadata dict for the participant and stores it.
        """

        # Read data from csv and set index to crop identifier (filename before .png)
        df = pd.read_csv(self.crop_metadata_path, index_col = 'crop_identifier')

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
                timepoints = trial_df["time_in_trial"].tolist()

                # Create dict for timepoints in this trial
                data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"] = {}

                # For each timepoint in this trial
                for timepoint in timepoints:
                    # Filter trial dataframe by timepoint
                    timepoint_df = trial_df[trial_df["time_in_trial"] == timepoint]

                    # get sceneID for this timepoint
                    sceneID = timepoint_df['sceneID'].iloc[0]

                    # Create dicts for this timepoint
                    data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint] = {}

                    # Fill in crop identifier value (here the index) in main data dict for this timepoint, as well as scene id and space for meg data
                    # metadata
                    data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint]["crop_identifier"] = timepoint_df.index.tolist()[0]  # slice with [0] to get single element instead of string
                    data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint]["sceneID"] = sceneID
        
        # Export dict to json 
        self.save_dict_as_json(type_of_content="crop_metadata", metadata_dict=data_dict)


class DatasetHelper:
    def __init__(self, subject_id: str = "02"):
        self.subject_id = subject_id
        
        self.session_ids_num = [session_id for session_id in range(1,11)]
        self.session_ids_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

        self.crop_metadata_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_224/metadata/as{subject_id}_crops_metadata.csv"
        self.meg_metadata_folder = f"/share/klab/datasets/avs/population_codes/as{subject_id}/sensor/filter_0.2_200"


    def create_crop_dataset(self) -> None:
        """
        Creates the crop dataset with all crops in the combined_metadata (crops for which meg data exists)
        """

        # Read combined metadata from json
        combined_metadata_file = open(f"data_files/metadata/combined_metadata/_subject_{self.subject_id}/combined_metadata_dict.json")
        combined_metadata_string = combined_metadata_file.read()
        combined_metadata = json.loads(combined_metadata_string)
        
        for session_id in self.session_ids_num:
            pass


    def create_meg_dataset(self) -> None:
        """
        Creates the crop dataset with all crops in the combined_metadata (crops for which meg data exists)
        """

        # Read combined metadata from json
        combined_metadata_file = open(f"data_files/metadata/combined_metadata/_subject_{self.subject_id}/combined_metadata_dict.json")
        combined_metadata_string = combined_metadata_file.read()
        combined_metadata = json.loads(combined_metadata_string)
        
        for session_id in self.session_ids_num:
            pass
        
    
    def create_train_test_split(self):
        """
        Creates train/test split of trials based on sceneIDs.
        """
        # Read combined metadata from json
        combined_metadata_file = open(f"data_files/metadata/combined_metadata/_subject_{self.subject_id}/combined_metadata_dict.json")
        combined_metadata_string = combined_metadata_file.read()
        combined_metadata = json.loads(combined_metadata_string)

        num_total_datapoints = 0
        sceneIDs = {}
        trialIDs = []
        for trial_id in combined_metadata["trials"]:
            trialIDs.append(trial_id)
            for timepoint in combined_metadata["trials"][trial_id]["timepoints"]:
                # get sceneID of current timepoint
                sceneID_current = combined_metadata["trials"][trial_id]["timepoints"][timepoint]["sceneID"]
                # First occurance of the scene? Store its occurange with the corresonding trial
                if sceneID_current not in sceneIDs:
                    sceneIDs[sceneID_current] = {"trials": [trial_id]}
                # SceneID previously occured in another trial? add current trial to stored data of scene
                elif trial_id not in sceneIDs[sceneID_current]["trials"]:
                    sceneIDs[sceneID_current]["trials"].append(trial_id)

                # Count total datapoints/timepoints/fixations in combined metadata
                num_total_datapoints += 1
        