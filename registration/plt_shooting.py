## Geodesic shooting for several points using se kernel
# 
from functools import partial

import jax
from jax import random, grad
import jax.numpy as np
import jax.numpy.linalg as linalg

import matplotlib.pyplot as plt
import matplotlib as mpl
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.family'] = 'Times New Roman'
cmap = plt.cm.get_cmap('bwr')


import sys
sys.path.append('../gp')
from plt_utils import *

from jax_registration import *


## Parameters

n = 5
xlim = (0, 1)
ylim = (0, 1)
ℓ = .25
euler_steps = 20
δt = .1
grid_nlines = 11

## partials 

k = partial(cov_se, ℓ=ℓ)
shooting_step = jax.jit(partial(HamiltonianStep, k=k, δt=δt))

## Data

key = random.PRNGKey(5)
key, sk = random.split(key); q0 = random.uniform(sk, (n, 2))*.5+.25
key, sk = random.split(key); p0 = random.normal(sk, (n, 2))*.2

q0 = np.array([[.2,.3], [.4,.7], [.5,.65], [.8,.4]])
p0 = np.array([[.15,.05], [-.05,-.1], [.1,-.1], [0,.15]])

g0, gL = GridData(nlines=grid_nlines)
## Plotting

fig, axs = plt.subplots(1,4,figsize=(20,5))

q = q0; p = p0; g = g0
qs = [q]; ps = [p]

axi = 0
display_ts = [int(x*euler_steps) for x in [0.,.25,.5,1-1/euler_steps]]

for t in range(euler_steps):
    
    if t in display_ts:
        ax = axs[axi]; axi += 1
        plt_grid(ax, g, gL)
        plt_vectorfield(ax, g, k(g, q)@p, scale=10, color='k')
        plt_vectorfield(ax, q, p, color=cmap(.7))
        for qi, (q_, p_) in enumerate(zip(qs[::-1], ps[::-1])):
            ax.scatter(q_[:,0], q_[:,1], color=cmap(.9-.02*qi))
            plt_vectorfield(ax, q_, p_, color=cmap(.1+.02*qi))
        ax.set_title(f'$t$={t/euler_steps}')
        ax.set_xlabel('$\mathcal{H}$'+f'$(q_t,p_t)$={Hqp(q,p,k):.4f}')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    q, p, g = shooting_step(q, p, g)
    qs.append(q)
    ps.append(p)
    
    

fig.tight_layout()
plt_savefig(fig, 'summary/assets/plt_shooting.png')
