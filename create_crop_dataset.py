import os
import json
import pandas as pd
import numpy as np
import imageio

# Params
subject_id = "02"
session_id = "1"
epochs_in_session = 2874

# Read timepoint_dict_crop from json
timepoint_dict_crops_file = open(f"data_files/timepoint_dict_crops_subject_{subject_id}.json")
timepoint_dict_crops_string = timepoint_dict_crops_file.read()
timepoint_dict_crops = json.loads(timepoint_dict_crops_string)

# Define path to read crops from
crop_folder_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_224/crops/as{subject_id}"


# Iterate over dict, counting timepoints, extracting the image for each timepoint/epoch/fixation
def create_crop_ds_from_metadata(timepoint_dict_crops: dict, 
                                num_epochs: int, 
                                session_id: str="1", 
                                subject_id: str="02",
                            ):
    
    crop_folder_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_224/crops/as{subject_id}"

    crop_list = []
    total_index = 0
    for trial_id in timepoint_dict_crops["sessions"][session_id]["trials"].keys():
        for timepoint_id in timepoint_dict_crops["sessions"][session_id]["trials"][trial_id]["timepoints"].keys():
            # Stop if all images for a session have been extracted
            if total_index >= num_epochs:
                # Create np array from list and return ds
                return np.stack(crop_list, axis=0)

            # Extract image for every epoch/fixation
            # Get crop path
            crop_filename = ''.join([timepoint_dict_crops["sessions"][session_id]["trials"][trial_id]["timepoints"][timepoint_id]["crop_identifier"], ".png"])
            crop_path = os.path.join(crop_folder_path, crop_filename)

            # Read crop as array and concat
            crop = imageio.imread(crop_path)
            crop_list.append(crop)

            # Advance iteration
            total_index += 1

crop_ds = create_crop_ds_from_metadata(
    timepoint_dict_crops=timepoint_dict_crops,
    num_epochs=epochs_in_session,
    session_id=session_id,
    subject_id=subject_id
    )

print(f"Done creating crop dataset for participant {subject_id}")
print(f"crop_ds.shape: {crop_ds.shape}")  #(2874, 224, 224, 3)

# Export numpy array to .npz
np_save_path = f"data_files/crop_ds_subj_{subject_id}_sess_{session_id}.npy"
np.save(np_save_path, crop_ds)

print("Done exporting to .npz file")