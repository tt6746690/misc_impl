import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

def plot_im(
    ims,
    n_row=8,
    normalize=True,
    plot_title=None,
    im_path=None,
    fig_size=None):
    
    if isinstance(ims, torch.Tensor):
        ims = ims.cpu().detach()
        ims = ims.float()
    ims = torchvision.utils.make_grid(
        ims, padding=1, normalize=normalize, nrow=n_row)
    ims = np.transpose(ims, (1,2,0))
    plt.figure(figsize=fig_size)
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