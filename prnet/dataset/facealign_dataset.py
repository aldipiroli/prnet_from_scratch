import os
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms

class FaceAlignDataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        self.data_path = self.root_dir / cfg["DATA"]["sub_dataset"]
        self.bfm_path = self.root_dir / cfg["DATA"]["BFM"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img = self.transform_to_tensor(img)
        img = self.transform_resize(img)
        return img
