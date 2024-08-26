
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.color import rgb2gray, rgb2lab
from repo.custom_loader import CustomDataset
from repo.utils import create_embedding

# ab channel and embeddings generator from rgb images.
def image_ab_gen(data, transform, embedder, batch_size=8, device='cuda'):
   
    dataloader = DataLoader(CustomDataset(data, transform), batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        batch_np = batch.numpy().transpose((0, 2, 3, 1))  # Convert to HxWxC format

        # Convert RGB to grayscale
        gray_batch = rgb2gray(batch_np)  # already normalized 0-1

        # Convert RGB to LAB
        lab_batch = rgb2lab(batch_np)
        
        # Extract L channel
        x_batch = lab_batch[:, :, :, 0] / 100.0   # range = [0,1]
        x_batch = x_batch[..., np.newaxis]
        x_batch = torch.from_numpy(x_batch).float()
        x_batch = torch.permute(x_batch, (0, 3, 1, 2))  # b,c,h,w
        
        # Stacking of gray channel to 3-d channel
        temp = np.empty(shape=batch_np.shape)
        for i in range(len(gray_batch)):
            temp[i] = np.stack((gray_batch[i],) * 3, axis=-1)
        
        gray_batch = temp
        
        # Extract AB channels
        y_batch = lab_batch[:, :, :, 1:] / 128       # noramlize AB channels and range = [-1,1]
        y_batch = torch.from_numpy(y_batch).float()
        y_batch = torch.permute(y_batch, (0, 3, 1, 2))  # b,c,h,w

        # Embeddings from grayscale images
        embeddings = create_embedding(gray_batch, transform, embedder, device)
        
        # Yield the batches
        yield [x_batch, embeddings], y_batch
