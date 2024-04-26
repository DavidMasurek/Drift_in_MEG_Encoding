from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict

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

# Read crop metadata from json
crop_metadata_file = open(f"data_files/metadata/crop_metadata_subject_{subject_id}.json")
crop_metadata_string = crop_metadata_file.read()
crop_metadata = json.loads(crop_metadata_string)

# Read meg metadata from json
meg_metadata_file = open(f"data_files/metadata/meg_metadata_subject_{subject_id}_session_{session_id_num}.json")
meg_metadata_string = meg_metadata_file.read()
meg_metadata = json.loads(meg_metadata_string)

# Store information about timepoints that are present in both meg and crop data

# Use defaultdict to automatically create missing keys
def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

combined_metadata = recursive_defaultdict()
for trial_id in meg_metadata["trials"]:
    for timepoint_id in meg_metadata["trials"][trial_id]["timepoints"]:
        # For each timepoint in the meg metadata: Check if this timepoint is in the crop metadata
        try:
            crop_identifier = crop_metadata["sessions"][session_id_num]["trials"][trial_id]["timepoints"][timepoint_id]["crop_identifier"]
            combined_metadata["trials"][trial_id]["timepoints"][timepoint_id]["crop_identifier"] = crop_identifier
        except:
            continue

# Export dict to json 
json_file_path = f'data_files/metadata/combined_metadata_subject_{subject_id}_session_{session_id_num}.json'

with open(json_file_path, 'w') as file:
    # Serialize and save the dictionary to the file
    json.dump(combined_metadata, file, indent=4)