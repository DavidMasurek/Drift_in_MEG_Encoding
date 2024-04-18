import numpy as np
import pandas as pd
import json

subject_id = "02"

crop_csv_path = f"/share/klab/psulewski/psulewski/active-visual-semantics/input/fixation_crops/avs_meg_fixation_crops_scene_224/metadata/as{subject_id}_crops_metadata.csv"

# Read data from csv and set index to crop identifier (filename before .png)
df = pd.read_csv(crop_csv_path, index_col = 'crop_identifier')

# Remove rows/fixations without crop identifiers
df = df[df.index.notnull()]


num_sessions = df["session"].max() #10 sessions

# Create ordered dict
data_dict = {"sessions": {}}

for nr_session in range(1,num_sessions+1):
    # Create dict for this session
    data_dict["sessions"][nr_session] = {}

    # Filter dataframe by session
    session_df = df[df["session"] == nr_session]

    # Get list of all trials in this session
    trial_numbers = list(map(int, set(session_df["trial"].tolist())))

    # Create dict for trials in this session
    data_dict["sessions"][nr_session]["trials"] = {}

    # For each trial in the session
    for nr_trial in trial_numbers:
        # Filter session dataframe by trial
        trial_df = session_df[session_df["trial"] == nr_trial]

        # Create dict for this trial
        data_dict["sessions"][nr_session]["trials"][nr_trial] = {}

        # Get list of all timepoints in this trial
        timepoints = trial_df["time_in_trial"].tolist()

        # Create dict for timepoints in this trial
        data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"] = {}

        # For each timepoint in this trial
        for timepoint in timepoints:
            # Filter trial dataframe by timepoint
            timepoint_df = trial_df[trial_df["time_in_trial"] == timepoint]

            # Create dicts for this timepoint
            data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint] = {}

            # Fill in crop identifier value (here the index) in main data dict for this timepoint
            data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint]["crop_identifier"] = timepoint_df.index.tolist()[0]  # slice with [0] to get single element instead of string
            data_dict["sessions"][nr_session]["trials"][nr_trial]["timepoints"][timepoint]["meg"] = {}

print("Done creating data_dict.")

# Export dict to json 
json_file_path = f'data_files/timepoint_dict_crops_subject_{subject_id}.json'

with open(json_file_path, 'w') as file:
    # Serialize and save the dictionary to the file
    json.dump(data_dict, file, indent=4)