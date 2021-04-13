import jax
import jax.numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.family'] = 'Times New Roman'
cmap = plt.cm.get_cmap('bwr')

import sys
sys.path.append('../kernel')
from gpax import *


def plt_scaled_colobar_ax(ax):
    """ `fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))` """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return cax


def plt_kernel_matrix_one(fig, ax, K, title=None, n_ticks=5, custom_ticks=True, vmin=None, vmax=None):
    im = ax.imshow(K, vmin=vmin, vmax=vmax)
    ax.set_title(title if title is not None else '')
    fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))
    # custom ticks
    if custom_ticks:
        n = len(K)
        ticks = list(range(n))
        ticks_idx = np.rint(np.linspace(
            1, len(ticks), num=min(n_ticks,    len(ticks)))-1).astype(int)
        ticks = list(np.array(ticks)[ticks_idx])
        ax.set_xticks(np.linspace(0, n-1, len(ticks)))
        ax.set_yticks(np.linspace(0, n-1, len(ticks)))
        ax.set_xticklabels(ticks)
        ax.set_yticklabels(ticks)
    return fig, ax


ℓs = np.array([[.1, .2, .3],
               [.4, .5, .6],
               [.7, .8, .9],
               [1., 1.1, 1.2],
               [1.3,1.4,1.5],
               [2, 3, 5]])
m, n = ℓs.shape
fig, axs = plt.subplots(m,n,figsize=(5*n,5*m))
X = np.linspace(0,1,100)
    
for i in range(ℓs.shape[0]):
    for j in range(ℓs.shape[1]):
        ℓ = ℓs[i,j]
        k = CovSE()
        k = k.bind({'params': {'ℓ':  BijSoftplus.reverse(np.array([ℓ], dtype=np.float32)),
                               'σ2': BijSoftplus.reverse(np.array([1.], dtype=np.float32)),}})
        K = k(X)
        plt_kernel_matrix_one(fig, axs[i,j], K, f'SE(ℓ={ℓ:.2f})', vmin=0)


fig.tight_layout()
fig.savefig('summary/assets/plt_cov_vary_lengthscales.png', bbox_inches='tight', dpi=100)