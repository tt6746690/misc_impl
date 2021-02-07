# Reference:
#    https://peterroelants.github.io/posts/gaussian-process-kernels/
#    https://peterroelants.github.io/posts/gaussian-process-kernels/
# 
import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib as mpl
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.family'] = 'Times New Roman'
cmap = plt.cm.get_cmap('bwr')

import sys
sys.path.append('../kernel')
from jaxkern import (rbf_kernel, linear_kernel, cov_se, cov_rq, cov_pe)

from plt_utils import plt_savefig, plt_scaled_colobar_ax

def plt_gp_samples(X, f, μ, Σ, fig=None, axs=None, xlim=(-2,2), ylim=None, description=''):

    if fig is None and axs is None:
        gridspec_kw = {'width_ratios': [2, 1], 'height_ratios': [1]}
        fig, axs = plt.subplots(1, 2, gridspec_kw=gridspec_kw)
        fig.set_size_inches(15, 5)

    ax = axs[0]
    for i in range(f.shape[1]):
        ax.plot(X, f[:,i], c=cmap(1-i*.2))
    std = np.sqrt(np.diag(Σ))
    ax.fill_between(X.squeeze(), μ-2*std, μ+2*std, alpha=.2, color=cmap(.3))
    ax.set_xlim(xlim)
    if ylim is None:
        ylim = (-np.max(np.abs(f))*1.1, np.max(np.abs(f))*1.1)
    ax.set_ylim(ylim)
    ax.set_xlabel('$x$', labelpad=0)
    ax.set_ylabel('$f(x)$', labelpad=0)
    ax.grid()
    ax.set_title('$f\sim \mathcal{G}\mathcal{P}(0, k)$' + description)

    ax = axs[1]
    im = ax.imshow(Σ, cmap=cmap)
    fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))
    ax.set_title('$K$')
    # 5 custom ticks 
    n_ticks = 5
    ticks = list(range(xlim[0], xlim[1]+1))
    ticks_idx = np.rint(np.linspace(
        1, len(ticks), num=min(n_ticks,len(ticks)))-1).astype(int)
    ticks = list(np.array(ticks)[ticks_idx])
    ax.set_xticks(np.linspace(0, len(X), len(ticks)))
    ax.set_yticks(np.linspace(0, len(X), len(ticks)))
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    
    return fig, axs


## Parameters

nt = 500
xlim = (-3,3)
ylim = (-3,3)
ϵ = .0001
kernels = {
    'SE(ℓ=1)': cov_se,
    'RQ(α=1,ℓ=1)': cov_rq,
    'PE(p=1,ℓ=1)': lambda X: cov_pe(X, p=1)}

## Plotting

gridspec_kw = {'width_ratios': [2, 1], 'height_ratios': [1,1,1]}
fig, axs = plt.subplots(3, 2, gridspec_kw=gridspec_kw)
fig.set_size_inches(15, 15)

for i, (kernel_name, kernel) in enumerate(kernels.items()):
    X = np.expand_dims(np.linspace(*xlim, nt), 1)
    μ = np.zeros(nt)
    Σ = kernel(X)
    f = np.random.multivariate_normal(μ, Σ+ϵ*np.eye(nt), size=3).T
    plt_gp_samples(X, f, μ, Σ, fig, axs[i,:], description=f" where  $k$=${kernel_name}$", xlim = xlim, ylim=ylim)
fig.tight_layout()
plt_savefig(fig, 'summary/assets/plt_sample_from_gp.png')
