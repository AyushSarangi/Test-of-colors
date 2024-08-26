
import torch
from repo.data_generator import image_ab_gen
from repo.model import enc_dec as model
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def val_step(model: torch.nn.Module, data_loader, loss_fn: torch.nn.Module, 
             embedder, device='cuda'):
    
    model = model.to(device)
    
    # setting validation mode on
    model.eval()
    val_loss = 0     # validation loss
    num_batches = 0  # number of batches
    
    # setting inference mode on
    with torch.inference_mode():
        for (X_batch, embeds), Y_batch in data_loader():
            X_batch, embeds, Y_batch = X_batch.to(device), embeds.to(device), Y_batch.to(device)
            num_batches += 1

            Y_pred = model(X_batch, embeds)
            loss = loss_fn(Y_pred, Y_batch)
            val_loss += loss.item()
 
    val_mse = val_loss / num_batches
    return val_mse
