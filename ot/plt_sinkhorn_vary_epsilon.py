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

from ot.datasets import make_1D_gauss

import sys
sys.path.insert(0,'../gp/')
from gpax import *
from plt_utils import *

from otax import *


# https://pythonot.github.io/auto_examples/plot_OT_1D.html

n = 100

x = np.arange(n, dtype=np.float32)
a = np.asarray(make_1D_gauss(n, m=20, s=5)+make_1D_gauss(n,m=50,s=10))
b = np.asarray(make_1D_gauss(n, m=60, s=10))
a = a/np.sum(a)
b = b/np.sum(b)

C = sqdist(x)
C = C / C.max()


## Plotting


fig, axs = plt.subplots(1,4,figsize=(20,5))

ax = axs[0]
ax.plot(x, a, 'b', label='α')
ax.plot(x, b, 'r', label='β')
ax.set_ylim((0,.1))
ax.grid()
ax.set_title('histograms')
ax.legend(fontsize=20)


for i, ϵ in enumerate([.1,.01,.001]):
    ax = axs[i+1]
    P = sinkhorn_log_stabilized(a, b, C, ϵ, 100)
    ax.imshow(P)
    ax.set_title('$\epsilon$'+f'={ϵ}, C={np.sum(P*C):.3f}')
    

fig.tight_layout()
plt_savefig(fig, 'summary/assets/plt_sinkhorn_vary_epsilon.png')
    
