from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from utils import MetadataHelper

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Choose subject
subject_id = "02"

### Handle metadata for subject ###

# Initialize metadata helper
metadata_helper = MetadataHelper(subject_id=subject_id)

# Read crop metadata over all sessions 
metadata_helper.create_crop_metadata_dict()

# Read meg metadata over all sessions
metadata_helper.create_meg_metadata_dict()

# Combine meg and crop metadata over all sessions
metadata_helper.create_combined_metadata_dict()

### Create crop and meg dataset based on metadata ###

# Create train/test split based on sceneIDs (based on trial_ids)

