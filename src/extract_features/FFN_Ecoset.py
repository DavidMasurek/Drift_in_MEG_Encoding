# Overall imports
import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import imageio

# ANN specific imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset#, DataLoader


# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

print(__location__)

# Params
# Dataset
subject_id = "02"
session_id_num = "1"

# Model
ann_model = "Resnet50"
batch_size = 32

# Get train/test split based on trials (based on scenes)
crop_ds = {}
# Get crop dataset from .npz
for split in ["train", "test"]:
    # Load trial array
    np_crop_path =f"data_files/crop_data/crops_{split}_ds_subj_{subject_id}_sess_{session_id_num}.npy"
    crops = np.load(np_crop_path)
    crop_ds[split] = crops

print(f"crop_ds['train'].shape: {crop_ds['train'].shape}")
print(f"crop_ds['test'].shape: {crop_ds['test'].shape}")

# Convert dataset to torch
class NumpyImageDataset(Dataset):
    def __init__(self, numpy_array, transform=None):
        """
        Args:
            numpy_array (numpy.ndarray): A Numpy array of images (should be in CHW format if channels are present)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.numpy_array = numpy_array
        self.transform = transform

    def __len__(self):
        return len(self.numpy_array)

    def __getitem__(self, idx):
        image = self.numpy_array[idx]

        if self.transform:
            image = self.transform(image)

        # Since no labels are associated, we only return the image
        return image

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert numpy arrays to torch tensors
])

# Create torch dataset
train_ds = NumpyImageDataset(crop_ds['train'], transform=transform)
test_ds = NumpyImageDataset(crop_ds['test'], transform=transform)
# Create a DataLoader to handle batching
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Load model
model_name = f'{ann_model}_ecoset'
source = 'custom'
device = 'cuda'

extractor = get_extractor(
  model_name=model_name,
  source=source,
  device=device,
  pretrained=True
)

#print(f"extractor.show_model(): {extractor.show_model()}")

# Choose last (fully connected) layer
module_name = "fc"


# Extract and save features
model_input = {"train": train_dataloader, "test": test_dataloader}
for split in ["train", "test"]:
    # Extract features
    features = extractor.extract_features(
    batches=model_input[split],
    module_name=module_name,
    flatten_acts=True  # flatten 2D feature maps from convolutional layer
    )

    print(f"{split}_features.shape: {features.shape}")

    # Export numpy array to .npz
    feature_save_path = f"data_files/features/{split}_features_{model_name}_{module_name}_subj-{subject_id}_sess-{session_id_num}"
    np.save(feature_save_path, features)



