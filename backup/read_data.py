import numpy as np
import pandas as pd
import json

crop_csv_path = "as02_crops_metadata.csv"

# Read data from csv and set index to crop identifier (filename before .png)
df = pd.read_csv(crop_csv_path, index_col = 'crop_identifier')

# Remove rows/fixations without crop identifiers
df = df[df.index.notnull()]


num_sessions = df["session"].max() #10 sessions

# Create ordered dict
data_dict = {}

for nr_session in range(1,num_sessions+1):
    # Create dict for this session
    data_dict[f"session_{nr_session}"] = {}

    # Filter dataframe by session
    session_df = df[df["session"] == nr_session]

    # Get list of all trials in this session
    trial_numbers = list(map(int, set(session_df["trial"].tolist())))

    # For each trial in the session
    for nr_trial in trial_numbers:
        # Filter session dataframe by trial
        trial_df = session_df[session_df["trial"] == nr_trial]

        # Create dict for this trial
        data_dict[f"session_{nr_session}"][f"trial_{nr_trial}"] = {}

        # Get list of all timepoints in this trial
        timepoints = trial_df["time_in_trial"].tolist()

        # For each timepoint in this trial
        for timepoint in timepoints:
            # Filter trial dataframe by timepoint
            timepoint_df = trial_df[trial_df["time_in_trial"] == timepoint]

            # Create dicts for this timepoint
            data_dict[f"session_{nr_session}"][f"trial_{nr_trial}"][f"time_in_trial_{timepoint}"] = {}

            # Fill in crop identifier value (here the index) in main data dict for this timepoint
            data_dict[f"session_{nr_session}"][f"trial_{nr_trial}"][f"time_in_trial_{timepoint}"]["crop_identifier"] = timepoint_df.index.tolist()[0]  # slice with [0] to get single element instead of string

            data_dict[f"session_{nr_session}"][f"trial_{nr_trial}"][f"time_in_trial_{timepoint}"]["meg"] = None

print("Done creating data_dict.")

# Export dict to json 
json_file_name = 'data.json'

with open(json_file_name, 'w') as file:
    # Serialize and save the dictionary to the file
    json.dump(data_dict, file, indent=4)




# Debugging/Drawing

#crop = data_dict["sessions"]["1"]["trials"]["1"]["timepoint"]["0.55"]["crop"]

#vs.

#crop = data_dict["session_1"]["trial_1"]["timepoint_0.55"]["crop"]