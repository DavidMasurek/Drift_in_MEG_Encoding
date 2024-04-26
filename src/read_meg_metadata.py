from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import json

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

subject_id = "02"
session_id_letter = "a"

# Map letter session_id to number
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
session_mapping = {}
for num in range(1,11):
    letter = letters[num-1]
    session_mapping[letter] = str(num)
session_id_num = session_mapping[session_id_letter]

meg_csv_folder = "/share/klab/datasets/avs/population_codes/as02/sensor/filter_0.2_200"
meg_csv_file = f"as{subject_id}{session_id_letter}_et_epochs_metadata_fixation.csv"
meg_csv_path = os.path.join(meg_csv_folder, meg_csv_file)

# Read metadata from csv 
df = pd.read_csv(meg_csv_path, delimiter=";")

def create_metadata_dict(df):
    # Create ordered dict
    data_dict = {"trials":{}}

    for index, row in df.iterrows():
        trial_id = int(row["trial"])
        timepoint = row["time_in_trial"]
        
        # Create dict to store values for current trial if this is the first timepoint in the trial
        if trial_id not in data_dict["trials"].keys():
            data_dict["trials"][trial_id] = {"timepoints": {}}

        # Store availability of current timepoint in trial
        data_dict["trials"][trial_id]["timepoints"][timepoint] = {"meg":True}

    return data_dict

data_dict = create_metadata_dict(df)

# Export dict to json 
json_file_path = f'data_files/metadata/meg_metadata_subject_{subject_id}_session_{session_id_num}.json'

with open(json_file_path, 'w') as file:
    # Serialize and save the dictionary to the file
    json.dump(data_dict, file, indent=4)
