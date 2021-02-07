import numpy as np
import matplotlib.pyplot as plt

def plt_subplots_1x1_if_not_exists(fig, ax, gridspec_kw=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, gridspec_kw=gridspec_kw)
        fig.set_size_inches(5, 5)
    return fig, ax

def plt_subplots_1x2_if_not_exists(fig, axs, s=5, gridspec_kw=None):
    if fig is None and axs is None:
        fig, axs = plt.subplots(1, 2, gridspec_kw=gridspec_kw)
        fig.set_size_inches(s*2, s)
    return fig, axs


def plt_scaled_colobar_ax(ax):
    """ `fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))` """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return cax

    
def plt_savefig(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)