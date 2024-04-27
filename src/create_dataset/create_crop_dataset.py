import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import imageio

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Params
subject_id = "02"
session_id_num = "1"

# Read combined metadata from json
combined_metadata_file = open(f"data_files/metadata/combined_metadata_subject_{subject_id}_session_{session_id_num}.json")
combined_metadata_string = combined_metadata_file.read()
combined_metadata = json.loads(combined_metadata_string)

# Get train/test split based on trials (based on scenes)
trials_split_dict = {}
for split in ["train", "test"]:
    # Load trial array
    split_trials_path = f"data_files/split/trials_{split}_subj_{subject_id}_sess_{session_id_num}.npy"
    trials_split = np.load(split_trials_path)
    trials_split_dict[split] = trials_split
train_trials = trials_split_dict["train"]
test_trials = trials_split_dict["test"]

# Define path to read crops from
crop_folder_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_224/crops/as{subject_id}"

# Iterate over dict, counting timepoints, extracting the image for each timepoint/epoch/fixation
def create_crop_ds_from_metadata(combined_metadata: dict, 
                                train_trials: np.ndarray,
                                test_trials: np.ndarray
                            ):
    
    crop_folder_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_224/crops/as{subject_id}"

    crop_split = {"crops_train": [], "crops_test": []}
    for trial_id in combined_metadata["trials"].keys():
        # Check if trial belongs to train or test split
        if trial_id in train_trials:
            trial_type = "crops_train"
        elif trial_id in test_trials:
            trial_type = "crops_test"
        else:
            raise ValueError(f"Trial_id {trial_id} neither in train nor test split.")
        for timepoint_id in combined_metadata["trials"][trial_id]["timepoints"].keys():
            # Extract image for every epoch/fixation (if there is both crop and metadata for the timepoint)
            if timepoint_id in combined_metadata["trials"][trial_id]["timepoints"]:
                # Get crop path
                crop_filename = ''.join([combined_metadata["trials"][trial_id]["timepoints"][timepoint_id]["crop_identifier"], ".png"])
                crop_path = os.path.join(crop_folder_path, crop_filename)

                # Read crop as array and concat
                crop = imageio.imread(crop_path)
                crop_split[trial_type].append(crop)
    # Convert to numpy array
    for split_key in crop_split.keys():
        crop_split[split_key] = np.stack(crop_split[split_key], axis=0)

    # return ds
    return crop_split

crop_ds_dict = create_crop_ds_from_metadata(
    combined_metadata=combined_metadata,
    train_trials=train_trials,
    test_trials=test_trials,
    )

print(f"Done creating crop dataset for participant {subject_id}, session {session_id_num}")
print(f"crop_train_ds.shape: {crop_ds_dict['crops_train'].shape}")  # (2269, 224, 224, 3)
print(f"crop_test_ds.shape: {crop_ds_dict['crops_test'].shape}")  # (556, 224, 224, 3)

# Export numpy array to .npz
for split_key in crop_ds_dict:
    np_save_path = f"data_files/crop_data/{split_key}_ds_subj_{subject_id}_sess_{session_id_num}.npy"
    np.save(np_save_path, crop_ds_dict[split_key])

print("Done exporting to .npz files")