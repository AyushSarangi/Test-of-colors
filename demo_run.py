
import torch
from torch import nn
import numpy as np
import cv2 
import random
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb

from repo.model import enc_dec
from repo.utils import create_embedding,transformation
from repo.visualisation import visualize
transform = transformation()

# test_paths = []

# # replace it with actual directory
# for root, dirs, files in os.walk('/kaggle/input/landscape-image-colorization/landscape Images/color'):
#     for file in files:
#         test_paths.append(os.path.join(root, file))

# Load the model
model = enc_dec(input_shape = 256)
state_dict = torch.load('/kaggle/working/repo/colorization_model.pth')
model.load_state_dict(state_dict)

# load the embedder
embedder = torch_hub_load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.DEFAULT')

# set these params as per your need.
fig, axes = plt.subplots(20,3, figsize = (9,40))
samples = random.sample(test_paths, 20)

for i,sample in enumerate(samples):
    visualize(i,sample,axes,model,transform,embedder)
plt.savefig('test_images_3.png')
plt.tight_layout()
