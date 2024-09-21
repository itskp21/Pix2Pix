import torch
import os
from torchvision.utils import save_image

def save_model_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """ Save model checkpoint """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

def load_model_checkpoint(model, optimizer, checkpoint_path):
    """ Load model checkpoint """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def save_generated_images(images, epoch, output_dir):
    """ Save generated images """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, image in enumerate(images):
        save_image(image, os.path.join(output_dir, f"generated_{epoch}_{i}.png"))

def visualize_images(images, title='Images'):
    """ Simple image visualization """
    import matplotlib.pyplot as plt
    import numpy as np

    images = [image.cpu().numpy().transpose(1, 2, 0) for image in images]
    fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
    for img, ax in zip(images, axes):
        ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
    plt.title(title)
    plt.show()
