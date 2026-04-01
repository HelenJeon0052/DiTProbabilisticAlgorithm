import matplotlib.pyplot as plt

from monai.apps import DecathlonDataset

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized
)
from monai.data import DataLoader

# pre-processing
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image", "label"], spatial_size = (96, 96, 96)),
])


# MSD Task

train_dataset = DacathlonDataset(
    root_dir="./dataset",
    task="Task09_Spleen",
    section="training",
    transform = train_transforms,
    download=True,
    cache_rate = 1.0
)

# unit as a batch

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)




check_data = first(train_loader)
image, label = check_data["image"][0][0], check_data["label"][0][0]
print(f"Image Shape: {image.shape}")

