import numpy as onp
import jax
import jax.numpy as np
import jax.random as random
from functools import partial

import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.family'] = 'Times New Roman'
cmap = plt.cm.get_cmap('bwr')


import sys; sys.path.insert(0,'../gp/')
from gpax import *
from plt_utils import *
from otax import *

from sklearn.neighbors import KernelDensity
## Parameters

N, M = (50, 50)
lr = 0.01
blur = .05
ϵ = blur**2
n_iters = 100

## Data

key = random.PRNGKey(0)
t_i = np.linspace(0, 1, N).reshape(-1, 1)
t_j = np.linspace(0, 1, M).reshape(-1, 1)
x_i, y_j = 0.2 * t_i, 0.4 * t_j + 0.6

## Jitting 

a = np.ones((N,), dtype=np.float32) / N
b = np.ones((M,), dtype=np.float32) / M
sink = partial(sinkhorn_log_stabilized, ϵ=ϵ, ρ=1e5, n_iters=100)

def loss(x, y):
    P, L_αβ = sinkhorn_divergence(a, b, x, y, sqdist, sink)
    return L_αβ

loss_grad = jax.jit(jax.value_and_grad(loss))

## Plotting 


def jax_display_samples(ax, x, color):
    """Displays samples on the unit interval using a density curve."""
    kde = KernelDensity(kernel="gaussian", bandwidth=0.005).fit(x)
    t_plot = np.linspace(-0.1, 1.1, 1000)[:, np.newaxis]
    dens = np.exp(kde.score_samples(t_plot))
    dens = jax.ops.index_update(dens, jax.ops.index[np.array([0,-1])], [0,0])
    ax.fill(t_plot, dens, color=color)


Nsteps = int(5/lr) + 1
display_its = [int(t/lr) for t in [0., 0.25, 0.5, 1., 5.]]
fig, axs = plt.subplots(1, 5, figsize=(25, 5))
k = 0

for i in range(Nsteps):
    # gradient flow
    L_αβ, g = loss_grad(x_i, y_j)
    x_i = jax.ops.index_update(x_i, jax.ops.index[:], x_i - lr*len(x_i)*g)

    if i in display_its:
        ax = axs[k]
        k = k + 1
        jax_display_samples(ax, y_j, cmap(.2))
        jax_display_samples(ax, x_i, cmap(.8))
        ax.set_title(f"t = {lr*i:.2f}", fontsize=30)
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim((-0.5, 5.5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(x_i, np.ones_like(x_i)*-.2+.1*random.normal(key,x_i.shape),
                "rx", mew=1, alpha=.9, label='x')
        ax.plot(y_j, np.ones_like(y_j)*-.2+.1*random.normal(key,y_j.shape),
                "bx", mew=1, alpha=.9, label='y')
        if k == 1:
            ax.legend(loc='upper right', fontsize=25)
    
plt.tight_layout()
plt_savefig(fig, 'summary/assets/plt_gradientflow1d.png')
