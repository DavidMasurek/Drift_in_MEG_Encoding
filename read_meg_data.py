import json
import sys
import importlib.util

# Add AVS directory to sys.path to allow import
avs_project_root_path = "/share/klab/camme/camme/dmasurek/AVS-machine-room-copy/"
sys.path.insert(0, avs_project_root_path)

# Load the required functions
from avs_machine_room.dataloader.load_population_codes.load_h5 import load_pop_code_dynamics

# Read crop_dict from json
crop_dict_file = open("data_files/crop_dict.json")
crop_dict_string = crop_dict_file.read()
crop_dict = json.loads(crop_dict_string)

# Load and inspect MEG data
path_to_meg_data = "/share/klab/datasets/avs/population_codes/as02/sensor/filter_0.2_200"
meg_data_file = "as02a_population_codes_fixation_500hz_masked_False.h5"

pop_code_dynamics = load_pop_code_dynamics(subject=["2"], 
                                        session=1,  
                                        input_dir=path_to_meg_data, 
                                        event_type="fixation", 
                                        s_freq=500, 
                                        masked=False, 
                                        take_norm=True
                    )

print(f"pop_code_dynamics.keys(): {pop_code_dynamics.keys()}")