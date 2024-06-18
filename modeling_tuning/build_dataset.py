import os
import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx])
        mask = Image.open(self.mask_files[idx])
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # mask = (mask > 0).float()  # Convert to binary (0 or 1)
        
        return image, mask