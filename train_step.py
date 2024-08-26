
from repo.data_generator import image_ab_gen
from repo.model import enc_dec as model
import numpy as np
import cv2 
import os
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_step(model: torch.nn.Module, data_loader, loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, embedder, device):
    
    model = model.to(device)
    
    # setting training mode on
    model.train()
    train_loss = 0
    num_batches = 0
    
    for (X_batch, embeds), Y_batch in data_loader():
        X_batch, embeds, Y_batch = X_batch.to(device), embeds.to(device), Y_batch.to(device)
        num_batches += 1
        
        # prediction
        Y_pred = model(X_batch, embeds)
        
        # computing loss
        loss = loss_fn(Y_pred, Y_batch)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        
        # bacward prop
        loss.backward()
        optimizer.step()
    
    train_mse = train_loss / num_batches
    return train_mse
