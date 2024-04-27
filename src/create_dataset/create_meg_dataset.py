import sys
import os
from pathlib import Path
import importlib.util

import json
import h5py

import numpy as np

# Choose dataset params
subject_id = "02"
session_id_letter = "a"

# Map letter session_id to number
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
session_mapping = {}
for num in range(1,11):
    letter = letters[num-1]
    session_mapping[letter] = str(num)
session_id_num = session_mapping[session_id_letter]

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Add AVS directory to sys.path to allow import
avs_project_root_path = "/share/klab/camme/camme/dmasurek/AVS-machine-room-copy/"
sys.path.insert(0, avs_project_root_path)

# Load the required functions
from avs_machine_room.dataloader.load_population_codes.load_h5 import norm_per_voxel

# Read combined metadata from json
combined_metadata_file = open(f"data_files/metadata/combined_metadata_subject_{subject_id}_session_{session_id_num}.json")
combined_metadata_string = combined_metadata_file.read()
combined_metadata = json.loads(combined_metadata_string)

# Read meg metadata from json
meg_metadata_file = open(f"data_files/metadata/meg_metadata_subject_{subject_id}_session_{session_id_num}.json")
meg_metadata_string = meg_metadata_file.read()
meg_metadata = json.loads(meg_metadata_string)

# Load and inspect MEG data
path_to_meg_data = f"/share/klab/datasets/avs/population_codes/as{subject_id}/sensor/filter_0.2_200"
meg_data_file = f"as{subject_id}{session_id_letter}_population_codes_fixation_500hz_masked_False.h5"

# Define globals
meg_data = {}
num_epochs = None

def normalize_array(data):
    # Normalize across complete session
    data_min = data.min()
    data_max = data.max()
    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data

# Read .h5 file
with h5py.File(os.path.join(path_to_meg_data, meg_data_file), "r") as f:
    # session 1 trial 1 starts at 5350
    # we have 2875 (2874 starting from 0) epochs in session a, those belong to 2874 different crops/time_in_trials

    # Get meg data
    meg_data["grad"] = f['grad']['onset']  # participant 2, session a (2874, 204, 601)
    meg_data["mag"] = f['mag']['onset']  # participant 2, session a (2874, 102, 601)
    num_epochs = f['grad']['onset'].shape[0]

    # Debugging: print meg (grad) shape
    print(f"meg_data['grad'].shape: {meg_data['grad'].shape}")

    # Normalize grad and mag independently
    meg_data["grad"] = normalize_array(np.array(meg_data['grad']))
    meg_data["mag"] = normalize_array(np.array(meg_data['grad']))

    # Combine grad and mag data
    combined_meg = np.concatenate([meg_data["grad"], meg_data["mag"]], axis=1) #(2874, 306, 601)

    # Train-test split based on scene ids

    # Save sceneIDs in session for split
    num_total_datapoints = 0
    sceneIDs = {}
    trialIDs = []
    scene_freq = {}
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


    num_scenes = len(sceneIDs)  # subject 2, session a: 300 
    num_trials = len(trialIDs)

    assert num_scenes == num_trials == len(set(sceneIDs)), "Number of trials and number of scenes is not identical. Doubled scenes need to be considered"

    # Choose sceneIds for 80/20 split
    num_trials_train = int(num_trials*0.8)
    num_trials_test = num_trials - num_trials_train

    # Split based on scene ids, but trial information is sufficient to identify datapoint
    train_split_trials = []
    test_split_trials = []
    # Create split sceneID (by ordering the trials by scenes, trials that belong to the same scene will most likely be in the same split)
    index = 0
    for sceneID in sceneIDs:
        for trial_id in sceneIDs[sceneID]["trials"]:
            # Each unique combination is one point in the dataset
            if index < num_trials_train:
                train_split_trials.append(trial_id)
            else:
                test_split_trials.append(trial_id)
            index += 1
    
    # To-do: Make sure that scenes only double in the same split

    print(f"len train_split_trials: {len(train_split_trials)}")
    print(f"len test_split_trials: {len(test_split_trials)}")

    print(f"Total number of datapoints available: {num_total_datapoints}")
    # to do: print total number of datapoints in train+test dataset
    print(f"Total number of trials in train+test dataset: {len(train_split_trials) + len(test_split_trials)}")


    # Iterate over combined metadata dict and store meg data from respective index in train or test set
    def create_meg_dataset(meg_metadata, train_split_trials, test_split_trials):
        train_ds = []
        test_ds = []

        # Iterate through meg metadata for simplicity with indexing
        index = 0
        for trial_id in meg_metadata["trials"]:
            if trial_id in train_split_trials:
                trial_type = "train"
            elif trial_id in test_split_trials:
                trial_type = "test"
            else:
                raise ValueError(f"Trial_id {trial_id} neither in train nor test split.")
            for timepoint in meg_metadata["trials"][trial_id]["timepoints"]:
                # Check if there is both crop and metadata for the timepoint
                if timepoint in combined_metadata["trials"][trial_id]["timepoints"]:
                    # Assign to split
                    if trial_type == "train":
                        train_ds.append(combined_meg[index])
                    elif trial_type == "test":
                        test_ds.append(combined_meg[index])
                # Advance index
                index += 1

        return train_ds, test_ds

    train_ds, test_ds = create_meg_dataset(meg_metadata, train_split_trials, test_split_trials)
        
    train_ds = np.array(train_ds)
    test_ds = np.array(test_ds)

    # Debug
    print(f"train_ds.shape: {train_ds.shape}")  #(2269, 408, 601)
    print(f"test_ds.shape: {test_ds.shape}")  #(556, 408, 601)

    num_meg_datapoints = train_ds.shape[0] + test_ds.shape[0]
    print(f"num_meg_datapoints: {num_meg_datapoints}")

    # Export meg numpy array and split based on sceneIDs to .npz
    for split in ["train", "test"]:
        if split == "train":
            ds = train_ds
            trials = train_split_trials
        else:
            ds = test_ds
            trials = test_split_trials
        # Save meg
        meg_save_path = f"data_files/meg_data/meg_{split}_ds_subj_{subject_id}_sess_{session_id_num}.npy"
        np.save(meg_save_path, ds)
        # save trial split based on scenes
        split_save_path = f"data_files/split/trials_{split}_subj_{subject_id}_sess_{session_id_num}.npy"
        np.save(split_save_path, trials)

print("Done creating meg dataset.")




