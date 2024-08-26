
import torch
from torch import nn, optim
import numpy as np
import cv2 

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# enc-dec with residual network
class enc_dec(nn.Module):
    def __init__(self, input_shape: int):
        super(enc_dec,self).__init__()
        
        # Encoders
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1,padding='same'),
            nn.LeakyReLU(0.1))
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,padding='same'),
            nn.ReLU())
        
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        # Decoders
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels=1256, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding='same'),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
        self.dec4 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding='same'),
            nn.Tanh())
      
    def forward(self, images, embeddings):
        # Encoder
        enc1_out = self.enc1(images)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        encoded_img = self.enc4(enc3_out)
        
        # Preparing embeddings
        b, c, h, w = encoded_img.shape
        embeddings = embeddings.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
        
        # Fusion layer
        fused = torch.cat((encoded_img, embeddings), dim=1)
        
        # Decoder with skip connections
        dec1_out = self.dec1(fused)
        dec2_out = self.dec2(torch.cat((dec1_out, enc3_out), dim=1))  # Skip connection from enc3 to dec2
        dec3_out = self.dec3(torch.cat((dec2_out, enc2_out), dim=1))  # Skip connection from enc2 to dec3
        decoded_channels = self.dec4(dec3_out)

        # return predicted ab channels
        return decoded_channels
