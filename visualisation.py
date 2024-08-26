
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
transform = transformation()

def visualize(i, sample, axes, model, transform, embedder):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device = device)
    color_me = sample
    try:
        color_me = cv2.cvtColor(cv2.imread(sample), cv2.COLOR_BGR2RGB)
        color_me=cv2.resize(color_me,(224,224))

        axes[i][1].imshow(color_me)
        axes[i][1].set_title('Actual')
        axes[i][1].axis('off')
        axes[i][0].imshow(cv2.cvtColor(color_me, cv2.COLOR_RGB2GRAY), cmap = 'gray')
        axes[i][0].axis('off')
        axes[i][0].set_title('Grayscale')

        model.eval()  # Set the model to evaluation mode

        # Run inference
        with torch.inference_mode():

            # Convert RGB to grayscale
            color_me= rgb2gray(color_me)

            temp = np.empty(shape = color_me.shape)
            temp = np.stack((color_me,)*3,axis = -1)
            gray_batch = temp[np.newaxis,:,:,:]

            # Get embeddings for grayscale RGB images
            embeddings = create_embedding(gray_batch, transform, embedder, device).to(device = device)

            color_me = color_me[np.newaxis,np.newaxis,:,:]
            color_me = torch.from_numpy(color_me).float().to(device = device)

            output = model(color_me, embeddings).to(device=device)
            output = output * 128
            color_me = color_me*100

        final_img = torch.cat((color_me,output), dim = 1).permute(0,2,3,1).cpu().numpy()
        for img in final_img:
            axes[i][2].imshow(lab2rgb(img))
            axes[i][2].axis('off')
            axes[i][2].set_title('Predicted')
    except Exception as e:
        print(f"Error in visualization: {e}")
        pass
