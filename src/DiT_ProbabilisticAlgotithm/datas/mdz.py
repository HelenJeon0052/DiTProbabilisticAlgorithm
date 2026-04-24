import torch
import torchvision
import gzip


from torch.utils.data import DataLoader, Dataset
import hashlib
import shutil
from pathlib import Path
import urllib.request
from typing import Any, Callable, Optional, Tuple





def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), "b"):
            h.update(chunk)

    return h.hexdigest()        

def download_md_file(url: str, dst: Path):
    dst.parent.mkdir(parents = True, exist_ok = True)

    temp = dst.with_name(dst.name + ".part")
    if temp.exists():
        temp.unlink()


    urllib.request.urlretrieve(url, temp)
    temp.replace(dst)

def gunzip_keep_archive(gz_path: Path, h5_path: Path):
    temp = h5_path.with_name(h5_path.name + ".part")

    if temp.exists():
        temp.unlink()


    with gzip.open(gz_path, "rb") as src, temp.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    temp.replace(h5_path)

def pcam_split_md(root:str, split: str, download: bool = True):
    if split not in PCAM_ZENODO_FILES:
        raise ValueError(f"not valid split: {split}")
    

    base_dir = Path(root) / "pcam"
    base_dir.mkdir(parents=True, exist_ok = True)

    for _, (gz_name, expected_md) in PCAM_ZENODO_FILES[split].items():
        gz_path = base_dir / gz_name
        h5_path = base_dir / gz_name.removesuffix(".gz")

        if h5_path.exists():
            continue

        if not gz_path.exists():
            if not download:
                raise FileNotFoundError(f"no {h5_path}. set download as True or manual download required")
        
            url = f"https://zenodo.org"
            print(f" downloading {gz_name} from Zenodo")
            download_md_file(url, gz_path)

        actual_md = md5sum(gz_path)
        if actual_md != expected_md:
            raise RuntimeError(f"actual files and expected are mismatch. delete and try again")
        
        print(f"decompressing {gz_name} to {h5_path.name}")
        gunzip_keep_archive(gz_path, h5_path)

def loading_pcam_md(root:str, splits = ("train", "val", "test"), download = True):
    for split in splits:
        pcam_split_md(root = root, split = split, download = download)