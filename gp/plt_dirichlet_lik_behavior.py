# Reference:
#    
import numpy as onp
onp.set_printoptions(precision=3,suppress=True)

import jax
import jax.numpy as np
from jax import grad, jit, vmap, device_put, random
from flax import linen as nn
from jax.scipy.stats import dirichlet

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.family'] = 'Times New Roman' 
cmap = plt.cm.get_cmap('bwr')

from tabulate import tabulate

from plt_utils import *
from gpax import *




## plot σ̃2, ỹᵢ as a function of α_ϵ
#      - smaller α_ϵ, ỹ will be further apart, and σ̃2 will be larger
#
#
fig, axs = plt.subplots(1, 2, figsize=(20,10))

ax = axs[0]
α_ϵ = np.linspace(0.001, 10, 100)
α = np.column_stack((α_ϵ, 1+α_ϵ))
σ2 = np.log( 1/α + 1 )
σ = np.sqrt(σ2)
y = np.log(α) - σ2/2

ax.plot(α_ϵ, y[:,0], label='y=0', c='r')
ax.plot(α_ϵ, y[:,1], label='y=1', c='b')
ax.fill_between(α_ϵ, (y-2*σ)[:,0], (y+2*σ)[:,0], ls='--', lw=3, facecolor='r', alpha=.1)
ax.fill_between(α_ϵ, (y-2*σ)[:,1], (y+2*σ)[:,1], ls='--', lw=3, facecolor='b', alpha=.1)
ax.set_ylabel('ỹ', fontsize=50)
ax.set_xlabel('α_ϵ', fontsize=50)
ax.set_xscale('log')
ax.legend(loc='lower right')
ax.grid()

ax = axs[1]
ax.plot(y[:,0], α_ϵ, c='r')
ax.set_ylabel('α', fontsize=50)
ax.set_xlabel('f', fontsize=50)
ax.grid()


fig.tight_layout()
plt_savefig(fig, 'summary/assets/plt_dirichlet_lik_behavior.png')
    