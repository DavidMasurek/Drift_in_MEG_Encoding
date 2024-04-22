import sys
import os
from pathlib import Path
import importlib.util

import json
import h5py

import numpy as np

# Choose dataset params
participant_id = "02"
session_id_letter = "a"

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
timepoint_dict_crops_file = open("data_files/timepoint_dict_crops.json")
timepoint_dict_crops_string = timepoint_dict_crops_file.read()
timepoint_dict_crops = json.loads(timepoint_dict_crops_string)

# Load and inspect MEG data
path_to_meg_data = f"/share/klab/datasets/avs/population_codes/as{participant_id}/sensor/filter_0.2_200"
meg_data_file = f"as{participant_id}{session_id_letter}_population_codes_fixation_500hz_masked_False.h5"

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
    # Debugging: meg file info
    print(f"num timepoints: {len(f.attrs['times'])}") # participant 2, session a: 2874

    # session 1 trial 1 starts at 5350
    # we have 2875 (2874 starting from 0) epochs in session a, those belong to 2874 different crops/time_in_trials

    meg_data["grad"] = f['grad']['onset']  # participant 2, session a (2874, 204, 601)
    meg_data["mag"] = f['mag']['onset']  # participant 2, session a (2874, 102, 601)
    num_epochs = f['grad']['onset'].shape[0]

    # Normalize grad and mag independently
    meg_data["grad"] = normalize_array(np.array(meg_data['grad']))
    meg_data["mag"] = normalize_array(np.array(meg_data['grad']))

    # Combine grad and mag data
    combined_meg = np.concatenate([meg_data["grad"], meg_data["mag"]], axis=1) #(2874, 306, 601)

    # Train-test split based on scene ids




print("Done filling meg data in dict.")

# No export yet, datatype to be debated

print("Done exporting meg data.")




