import torch
import torchvision
from torchvision.transforms import transforms

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from typing import Any, Callable, Optional, Tuple
from DiT_ProbabilisticAlgotithm.datas.mdz import loading_pcam_md


SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)







torch.manual_seed(SEED)



def pcam_split_exists(root: str, split: str):
    if split not in PCAM_FILES:
        raise ValueError(f"Invalid split: {split!r} | expected one of {set(PCAM_FILES)}")

    base_dir = Path(root) / "pcam"
    return all((base_dir / filename).is_file() for filename in PCAM_FILES[split])


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    base = [transforms.Resize((img_size, img_size))]

    if train:
        base += [
                    transforms.RandomHorizontalFlip(),
                ]

    base += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),] # [0, 1]

    return transforms.Compose(base)

def load_pcam_datasets(root:str, train_transforms, val_transforms, download: bool = True):
    print(f"root: {root}")

    # when gdown denies ...
    # loading_pcam_md(root = root, splits = ("train", "val", "test"), download = download)
    
    train_dataset = PCamDataset(data_root=root, split='train', transform=train_transforms, download=download)
    val_dataset = PCamDataset(data_root=root, split='val', transform=val_transforms, download=download)
    test_dataset = PCamDataset(data_root=root, split='test', transform=val_transforms, download=download)

    return train_dataset, val_dataset, test_dataset



def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):

    return DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = True,
        drop_last = False,
        persistent_workers = (num_workers > 0),
    )

# ------------------------------
# visualizing
# ------------------------------

def denormalize(img_chw: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """
    img_chw = torch.Tensor [C, L, W] normalized
    :return: ndarray [L, W, C] in [0, 1]
    """
    mean = torch.tensor(mean, dtype=img_chw.dtype, device=img_chw.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=img_chw.dtype, device=img_chw.device).view(3, 1, 1)

    x = img_chw * std + mean
    x = x.clamp(0, 1)

    return x.permute(1, 2, 0).cpu().numpy()

def require_pcam_split(root: str, split: str):
    base_dir = Path(root) / "pcam"
    missing = [
        str(base_dir / filename)
        for filename in PCAM_FILLES[split]
        if not (base_dir / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"PCAM split {split!r} is not complete\n"
            +"\n".join(missing)
        )

def show_batch(
    loader,
    num_images: int = 10,
    cols: int = 5,
    title: str = "batch",
    denormalize: bool = True,
):
    images, labels = next(iter(loader))

    images = images.detach().cpu()
    labels = labels.detach().cpu()

    num_images = min(num_images, images.size(0))
    rows = int(np.ceil(num_images / cols))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    plt.figure(figsize=(cols * 3, rows * 3))
    plt.suptitle(title, fontsize=14)

    for i in range(num_images):
        ax = plt.subplot(rows, cols, i + 1)

        img = images[i]

        if denormalize:
            img = img * std + mean

        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()

        label = int(labels[i].item())
        label_text = "Tumor (1)" if label == 1 else "Normal (0)"
        color = "red" if label == 1 else "green"

        ax.imshow(img)
        ax.set_title(label_text, color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

class PCamDataset(Dataset):
    """
    return: (image_tensor, label)
    """
    def __init__(self, data_root:str, split:str, transform: Optional[Callable]=None, download: bool=False):
        assert split in ['train', 'val', 'test'], f'invalid split: "{split}'

        is_download = download and not pcam_split_exists(data_root, split)

        self._h5 = None
        self._pcam = None
        self._h5_path = os.path.join(data_root, f"pcam_{split}.h5")
        self.dataset = torchvision.datasets.PCAM(
            root=data_root,
            split=split,
            transform=transform,
            download=download,
        )
    
    def __len__(self):
        return len(self.dataset)

    """def _ensure_open(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self._h5_path, 'r')"""
            
    def __getitem__(self, idx):
        x, y = self.dataset[idx] # x: [3, L, W], y: int (0/1)
        return x, int(y)
    
    def __getstate__(self):
        d = dict(self.__dict__)
        
        return d

def build_pcam_loader(
    data_root = PCAM_ROOT,
    img_size=224,
    batch_size=64,
    num_workers=2,
    download=True,
):
    # train_transforms = build_transforms(img_size, train=Ture)
    # eval_transforms = build_transforms(img_size, train=True)

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset, val_dataset, test_dataset = load_pcam_datasets(
        root = data_root,
        train_transforms = train_transforms,
        val_transforms = val_transforms,
        download = download,
    )

    train_loader = make_loader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    val_loader = make_loader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True)
    return train_loader, val_loader


