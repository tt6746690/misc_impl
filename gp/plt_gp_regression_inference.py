import sys
sys.path.append('../kernel')
from plt_utils import plt_savefig
from jaxkern import cov_se

import numpy as np
import numpy.random as npr
from numpy.linalg import inv, det, cholesky
from numpy.linalg import solve as backsolve

import jax
from jax import grad, jit, vmap, device_put
import jax.numpy as jnp
import jax.numpy.linalg as jnp_linalg
from jax.experimental import optimizers
from flax.core import freeze, unfreeze
import itertools

import matplotlib.pyplot as plt
import matplotlib as mpl
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.family'] = 'Times New Roman'
cmap = plt.cm.get_cmap('bwr')

from gp import gp_regression_chol, run_sgd
from gp import Gpr, flax_run_optim

## Parameters 
xlim = (-2, 2)
ylim = (-3, 3)
n_train = 3
n_test = 100
σn = .3
logsn = np.log(σn)
ℓ = 1
ℓs = [.1, .3, 1]
train_sizes = [5, 10, 50]
lr = .01
num_steps = 50

def f_gen(x):
    return np.sin(x)+np.sin(x*5)+np.cos(x*3)


def log_func(i, f, params):
    if i&10==0:
        print(f'[{i:3}]\tLoss={f(params):.3f}\t'
            f'σn={jnp.exp(params["logσn"][0]):.3f}')


## Plotting

fig, axs = plt.subplots(3, 3, sharey=True)
fig.set_size_inches(30, 15)

key = jax.random.PRNGKey(0)
npr.seed(0)
X_test = np.expand_dims(np.linspace(*xlim, n_test), 1)
X_train_all = np.expand_dims(
    npr.uniform(xlim[0], xlim[1], size=train_sizes[-1]), 1)
ϵ_all = σn*npr.rand(train_sizes[-1], 1)

for i, ℓ in enumerate(ℓs):
    for j, n_train in enumerate(train_sizes):
        
        X_train = X_train_all[:n_train]
        ϵ = ϵ_all[:n_train]
        y_train = f_gen(X_train) + ϵ
        
        if i == 1:
            model = Gpr((X_train, y_train))
            params = model.get_init_params(key)
            def nmll(params):
                return -model.apply({'params': params}, method=model.mll)
            params = flax_run_optim(nmll, params, lr=lr,
                                    num_steps=num_steps, log_func=log_func)
        else:
            params = {'k': {'logσ': np.log(np.array([1.])),
                            'logℓ': np.log(np.array([ℓ]))},
                      'logσn': np.log(np.array([σn]))}
            params = freeze(params)
        
        ℓ = np.exp(params['k']['logℓ'])[0]
        params = {'params': params}

        model = Gpr((X_train, y_train))
        mll = model.apply(params, method=model.mll)
        μ, Σ = model.apply(params, X_test, method=model.pred_f)
        std = np.expand_dims(np.sqrt(np.diag(Σ)), 1)

        ax = axs[i, j]
        ax.plot(X_test, μ, color='k')
        ax.fill_between(X_test.squeeze(), (μ-2*std).squeeze(), (μ+2*std).squeeze(), alpha=.2, color=cmap(.3))
        ax.scatter(X_train, y_train, marker='x', color='r', s=50)
        ax.grid()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title('$-\log p(\mathbf{y}\mid X)$'+f'={-mll:.2f}')
        
        if j == 0 or i == 1:
            ax.set_ylabel(f"ℓ={ℓ:.2f}", fontsize=45)
        if i == 2:
            ax.set_xlabel("$n$"+f"={n_train}", fontsize=45)

fig.tight_layout()
plt_savefig(fig, 'summary/assets/plt_gp_regression_inference.png')