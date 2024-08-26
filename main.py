
import torch
from torch import nn, optim
from repo.data_setup import load_images
from repo.utils import transformation
from repo.data_generator import image_ab_gen
from repo.train import train
from repo.model import enc_dec
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torch.hub import load as torch_hub_load


def main(img_dir, batch_size, epochs, 
         learning_rate, patience,
         factor, test_size, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loading images
    images = load_images(img_dir)
    
    # Spliting the data into training and validation sets
    x_train, x_val = train_test_split(images, test_size=test_size, random_state=42, shuffle=True)

    # Setting up the data transformations
    transform = transformation()

    # Loading the pre-trained Inception v3 model for embeddings
    embedder = torch_hub_load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.DEFAULT')
    
    # Initializing the model
    model = enc_dec(input_shape = 256).to(device)
#     state_dict = torch.load('/kaggle/working/repo/colorization_model.pth')
#     model.load_state_dict(state_dict)

    # Set up the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Set up the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.1, verbose=True)

    # Set up data loaders for training and validation
    train_loader = lambda: image_ab_gen(x_train, transform, embedder, batch_size=batch_size, device=device)
    val_loader = lambda: image_ab_gen(x_val, transform, embedder, batch_size=batch_size, device=device)

    # Train the model
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=epochs,
        optimizer=optimizer,
        embedder=embedder,
        device=device
    )

    #save the model
#     torch.save(model.state_dict(), "/kaggle/working/repo/colorization_model_2.pth")

    return results


if __name__ == "__main__":
    # Specifying the directory containing images
    img_dir = "/kaggle/input/pokemon-image-dataset/images"  # Specify the directory containing the images
    main(img_dir, batch_size = 8, epochs = 50, 
         patience = 10,learning_rate=0.001,
         factor = 0.1, test_size = 0.20)  # factor, patience are params for learning rate scheduler 
