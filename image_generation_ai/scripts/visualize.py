import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

def show_image_batch(images, nrow=8, title=None):
    """
    images: Tensor of shape [B, C, H, W], values in [-1,1]
    nrow:   how many images per row in the grid
    title: optional title for the plot
    """
    # make a grid of images
    grid = vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)
    # move channels last and convert to numpy
    npimg = grid.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.imshow(npimg)
    plt.show()
