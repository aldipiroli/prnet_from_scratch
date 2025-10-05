import os
from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pickle


class PreprocessedFaceAlignDataset(Dataset):
    def __init__(self, cfg, mode, logger):
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        self.data_path = self.root_dir / cfg["DATA"]["sub_dataset"] / "processed"
        self.files = os.listdir(self.data_path)
        self.logger.info(f"Files found {len(self.files)}")
        assert len(self.files) > 0
        self.files = self.get_split()
        self.logger.info(f"Mode: {self.mode}, n_files: {len(self.files)}")
        self.img_size = cfg["DATA"]["img_size"]
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def get_split(self, ratio=0.8):
        n = int(len(self.files) * ratio)
        if self.mode == "train":
            return self.files[:n]
        else:
            return self.files[n:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_path = self.files[idx]
        with open(f"{self.data_path}/{data_path}", "rb") as file:
            data = pickle.load(file)
        img = data["img"]
        gt = data["gt"]
        return img, gt
