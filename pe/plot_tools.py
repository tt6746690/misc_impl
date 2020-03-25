import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

def plot_im(
    ims,
    nrow=8,
    normalize=True,
    plot_title=None,
    im_path=None,
    figsize=None):
    
    if isinstance(ims, torch.Tensor):
        ims = ims.cpu().detach()
        ims = ims.float()
    ims = torchvision.utils.make_grid(
        ims, padding=1, normalize=normalize, nrow=nrow)
    ims = np.transpose(ims, (1,2,0))
    plt.figure(figsize=figsize)
    plt.imshow(ims)
    plt.axis('off')
    if plot_title:
        plt.title(plot_title)
    if im_path:
        plt.savefig(im_path)
        plt.clf()
        plt.close('all')
    else:
        plt.show()