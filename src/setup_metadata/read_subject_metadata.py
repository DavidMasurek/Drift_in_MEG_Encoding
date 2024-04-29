from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from metadata_utils import metadata_helper

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Choose subject
subject_id = "02"

# Initialize metadata helper
metadata_helper = metadata_helper(subject_id=subject_id)

# Read crop metadata over all sessions 
metadata_helper.create_crop_metadata_dict()

# Read meg metadata over all sessions
metadata_helper.create_meg_metadata_dict()

# Combine meg and crop metadata over all sessions
metadata_helper.create_combined_metadata_dict()
