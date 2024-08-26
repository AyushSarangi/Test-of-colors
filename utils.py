
import torch
from torchvision import transforms
import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transformation():    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=20, shear=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

def create_embedding(grayscale_images, transform, embedder, device='cuda'): 
    # Set the embedder to evaluation mode and move it to the correct device
    embedder = embedder.to(device)
    embedder.eval()

    # Prepare input batch
    input_batch = torch.stack([transform(img) for img in grayscale_images])
    input_batch = input_batch.to(device).float()

    # Extract embeddings
    with torch.no_grad():
        embeddings = embedder(input_batch)

    return embeddings
