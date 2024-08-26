
from tqdm import tqdm
from repo.train_step import train_step
from repo.val_step import val_step
from repo.model import enc_dec as model
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def train(model: torch.nn.Module, train_loader, val_loader, scheduler, 
          loss_fn: torch.nn.Module, epochs: int, optimizer: torch.optim.Optimizer, 
          embedder, device='cuda'):
    
    results = {'train_loss': [], 'val_loss': []}
    model = model.to(device)
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, train_loader, loss_fn, optimizer, embedder, device)
        val_loss = val_step(model, val_loader, loss_fn, embedder, device)
        
        # applying scheduler on train loss to prevent overfitting
        scheduler.step(val_loss)
        
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        
        print(f'Epoch: {epoch+1} | train_loss: {train_loss:.7f} | val_loss: {val_loss:.7f}') 
    
    return results
