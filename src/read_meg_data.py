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
__location__ = Path(__file__).parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Add AVS directory to sys.path to allow import
avs_project_root_path = "/share/klab/camme/camme/dmasurek/AVS-machine-room-copy/"
sys.path.insert(0, avs_project_root_path)

# Load the required functions
from avs_machine_room.dataloader.load_population_codes.load_h5 import norm_per_voxel

# Read timepoint_dict_crop from json
timepoint_dict_crops_file = open(f"data_files/timepoint_dict_crops_subject_{subject_id}.json")
timepoint_dict_crops_string = timepoint_dict_crops_file.read()
timepoint_dict_crops = json.loads(timepoint_dict_crops_string)

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

# Read meg metadata to match meg epochs to scenes and fixations
meg_metadata_path = 

# Read .h5 file
with h5py.File(os.path.join(path_to_meg_data, meg_data_file), "r") as f:
    # session 1 trial 1 starts at 5350
    # we have 2875 (2874 starting from 0) epochs in session a, those belong to 2874 different crops/time_in_trials

    meg_data["grad"] = f['grad']['onset']  # participant 2, session a (2874, 204, 601)
    meg_data["mag"] = f['mag']['onset']  # participant 2, session a (2874, 102, 601)
    num_epochs = f['grad']['onset'].shape[0]

    # Get meg (grad) shape
    print(f"meg_data['grad'].shape: {meg_data['grad'].shape}")

    # Normalize grad and mag independently
    meg_data["grad"] = normalize_array(np.array(meg_data['grad']))
    meg_data["mag"] = normalize_array(np.array(meg_data['grad']))

    # Combine grad and mag data
    combined_meg = np.concatenate([meg_data["grad"], meg_data["mag"]], axis=1) #(2874, 306, 601)

    # Train-test split based on scene ids

    # Save sceneIDs in session for split
    num_crop_datapoints = 0
    sceneIDs = []
    trialIDs = []
    scene_freq = {}
    for trial_id in timepoint_dict_crops["sessions"][session_id_num]["trials"]:
        trialIDs.append(trial_id)
        for timepoint in timepoint_dict_crops["sessions"][session_id_num]["trials"][trial_id]["timepoints"]:
            # get sceneID of current timepoint
            sceneID_current = timepoint_dict_crops["sessions"][session_id_num]["trials"][trial_id]["timepoints"][timepoint]["sceneID"]
            if sceneID_current not in sceneIDs:
                sceneIDs.append(sceneID_current)

            # Count total datapoints/timepoints/fixations
            num_crop_datapoints += 1

            # Count occurances of sceneIDs
            if sceneID_current not in scene_freq.keys():
                scene_freq[sceneID_current] = 1
            else:
                scene_freq[sceneID_current] += 1

    assert len(sceneIDs) == len(set(sceneIDs))

    num_scenes = len(sceneIDs)  # subject 2, session a: 300 

    if num_scenes != len(trialIDs):
        raise ValueError("Number of trials and number of scenes is not identical. Doubled scenes need to be considered")


    # Choose sceneIds for 80/20 split
    num_scenes_train = int(num_scenes*0.8)
    num_scenes_test = num_scenes - num_scenes_train

    test_scenes = sceneIDs[:num_scenes_test]
    train_scenes = sceneIDs[num_scenes_test:]

    print(f"len train_scenes: {len(train_scenes)}")
    print(f"len test_scenes: {len(test_scenes)}")

    # Debugging: Calculate sum of datapoints that should be in test_scenes
    sum_datapoints_train = 0
    sum_datapoints_test = 0
    for sceneID in scene_freq.keys():
        if sceneID in train_scenes:
            sum_datapoints_train += scene_freq[sceneID]
        elif sceneID in test_scenes:
            sum_datapoints_test += scene_freq[sceneID]
        else:
            raise ValueError("Found ID that is neither in train nor test set.")

    print(f"num of datapoints that should be in train_ds: {sum_datapoints_train}")
    print(f"num of datapoints that should be in test_ds: {sum_datapoints_test}")

    print(f"num_crop_datapoints: {num_crop_datapoints}")


    # Iterate over metadata dict and store meg data from respective index in train or test set
    def create_meg_dataset(timepoint_dict_crops, train_scenes, test_scenes):
        train_ds = []
        test_ds = []

        index = 0
        for trial_id in timepoint_dict_crops["sessions"][session_id_num]["trials"]:
            for timepoint in timepoint_dict_crops["sessions"][session_id_num]["trials"][trial_id]["timepoints"]:
                if index >= 2874:
                    return train_ds, test_ds
                
                # get sceneID of current timepoint
                sceneID_current = timepoint_dict_crops["sessions"][session_id_num]["trials"][trial_id]["timepoints"][timepoint]["sceneID"]
                if sceneID_current in train_scenes:
                    train_ds.append(combined_meg[index])
                if sceneID_current in test_scenes:
                    test_ds.append(combined_meg[index])
                if sceneID_current == "113810.0":
                    print("found")
                #else:
                #    raise ValueError(f"SceneID {sceneID_current} does not belong to train or test set.")

                # Advance index
                index += 1

    train_ds, test_ds = create_meg_dataset(timepoint_dict_crops, train_scenes, test_scenes)
        
    train_ds = np.array(train_ds)
    test_ds = np.array(test_ds)

    # Debug
    print(f"train_ds.shape: {train_ds.shape}")
    print(f"test_ds.shape: {test_ds.shape}")

    num_meg_datapoints = train_ds.shape[0] + test_ds.shape[0]
    print(f"num_meg_datapoints: {num_meg_datapoints}")

    # Export meg numpy array and split based on sceneIDs to .npz
    for split in ["train", "test"]:
        if split == "train":
            ds = train_ds
            scenes = train_scenes
        else:
            ds = test_ds
            scenes = test_scenes
        # Save meg
        meg_save_path = f"data_files/meg_data/meg_{split}_ds_subj_{subject_id}_sess_{session_id_num}.npy"
        np.save(meg_save_path, ds)
        # save scene split
        split_save_path = f"data_files/split/sceneIDs_{split}_subj_{subject_id}_sess_{session_id_num}.npy"
        np.save(split_save_path, scenes)

print("Done creating meg dataset.")




