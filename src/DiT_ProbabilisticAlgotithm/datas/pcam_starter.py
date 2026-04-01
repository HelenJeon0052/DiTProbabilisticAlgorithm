import torch
import torchvision
from torchvision.transforms import transforms

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 64 # * (3.5)
IMG_SIZE = 224
DATA_ROOT = './data'
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

torch.manual_seed(SEED)

def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    base = [transforms.Resize((img_size, img_size))]

    if train:
        base += [
                    transforms.RandomHorizontalFlip(p=.5),
                ]

    base += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),] # [0, 1]

    return transforms.Compose(base)

train_transforms = build_transforms(IMG_SIZE, train=True)
eval_transforms = build_transforms(IMG_SIZE, train=False)


def load_pcam_datasets(root:str, train_transforms, eval_transforms, download: bool = True):
    train_dataset = torchvision.datasets.PCAM(root=root, split='train', transform=train_transforms, download=download)
    val_dataset = torchvision.datasets.PCAM(root=root, split='val', transform=eval_transforms, download=download)
    test_dataset = torchvision.datasets.PCAM(root=root, split='test', transform=eval_transforms, download=download)

    return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = load_pcam_datasets(
    root = DATA_ROOT,
    train_transforms = train_transforms,
    eval_transforms = eval_transforms,
    download = True
)

# root: download data, download = True
print('dataset loaded')

def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 2):
    use_cuda = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = use_cuda,
        drop_last = False,
        persistent_workers = (num_workers > 0),
    )
# Generated via DataLoader

train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = make_loader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = make_loader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2)

print(f'train dataset: {len(train_dataset)}')
print(f'val dataset: {len(val_dataset)}')
print(f'test dataset: {len(test_dataset)}')



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


def show_batch(loader, num_images: int = 10, cols: int = 5, title: str='batch'):
    images, labels = next(iter(loader))

    num_images = min(num_images, images.size(0))
    rows = int(np.ceil(num_images / cols))
    plt.figure(figsize=(cols * 3, rows * 3))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(num_images):
        # ax = plt.subplot(2, 5, i+1)
        img = images[i].numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.imshow(img)
        label_text = 'Tumor (1)' if labels[i].item() == 1 else 'Normal (0)'
        color = 'red' if labels[i].item() == 1 else 'green'
        plt.title(label_text, color=color)
        plt.xticks([])
        plt.yticks([])
    plt.show()






def build_pcam_loader(
    data_root='./data',
    img_size=224,
    batch_size=64,
    num_workers=2,
    download=True,
):
    train_transforms = build_transforms(img_size, train=Ture)
    eval_transforms = build_transforms(img_size, train=True)

    train_dataset, val_dataset, test_dataset = load_pcam_datasets(
        root = data_root,
        train_transforms = train_transforms,
        eval_transforms = eval_transforms,
        download = download,
    )

    train_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_loader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


class PCamDataset(Dataset):
    """
    return: (image_tensor, label)
    """

    def __init__(self, data_root:str, split:str, transform=None, download=False):
        assert split in ['train', 'val', 'test'], f'invalid split: "{split}'
        self._h5 = None,
        self._pcam = None
        self.dataset = torchvision.datasets.PCAM(
            root=data_root,
            split=split,
            transform=transform,
            download=download,
        )
    
    def __len__(self):
        return len(self.dataset)

    def _ensure_open(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self._h5_path(), 'r')
            
    def __getitem__(self, idx):
        self._ensure_open()
        x, y = self.dataset[idx] # x: [3, L, W], y: int (0/1)
        return x, int(y)
    
    def __get_state__(self):
        d = dict(self.__dict__)
        d['_h5'] = None
        d['_pcam'] = None
        
        return d
