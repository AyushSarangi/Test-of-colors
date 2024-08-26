
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 
import os

class CustomDataset(Dataset):
    
    # custom data loader
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
