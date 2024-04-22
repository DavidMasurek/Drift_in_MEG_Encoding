import sys
import os
from pathlib import Path
import importlib.util
import json
import h5py

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Read timepoint_dict_crop from json
timepoint_dict_crops_file = open("data_files/timepoint_dict_crops.json")
timepoint_dict_crops_string = timepoint_dict_crops_file.read()
timepoint_dict_crops = json.loads(timepoint_dict_crops_string)

# Load and inspect MEG data
path_to_meg_data = "/share/klab/datasets/avs/population_codes/as02/sensor/filter_0.2_200"
meg_data_file = "as02a_population_codes_fixation_500hz_masked_False.h5"

# Define globals
meg_data = {}
num_epochs = None

# Read .h5 file
with h5py.File(os.path.join(path_to_meg_data, meg_data_file), "r") as f:
    # Debugging: meg file info
    print(f"num timepoints: {len(f.attrs['times'])}") # participant 2, session a: 2874

    # session 1 trial 1 starts at 5350
    # we have 2875 (2874 starting from 0) epochs in session a, those belong to 2874 different crops/time_in_trials
    # so I have to go through the dict in the order of creation to map crops to meg epochs?

    meg_data["grad"] = f['grad']['onset']  # participant 2, session a (2874, 204, 601)
    meg_data["mag"] = f['mag']['onset']  # participant 2, session a (2874, 102, 601)
    num_epochs = f['grad']['onset'].shape[0]

    # Normalize grad and mag independently

    # Train-test split based on scene ids

    combined_meg = np.concatenate([meg_data["grad"], meg_data["mag"]], axis=1) #(2874, 306, 601)



print("Done filling meg data in dict.")

# No export yet, datatype to be debated

print("Done exporting meg data.")




