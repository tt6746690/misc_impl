import numpy as np
from numpy.linalg import inv, det, cholesky
from numpy.linalg import solve as backsolve
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
from jaxkern import cov_se

from plt_utils import plt_savefig


def gp_regression(X, y, Xt, k, σn):
    n = len(X)
    K  = k(X, X)+(σn**2)*np.eye(n)
    Km = k(X, Xt)
    Kt = k(Xt, Xt)
    Kinv = inv(K)
    μ = Km.T@Kinv@y_train
    Σ = Kt - Km.T@Kinv@Km
    logml = -(1/2)*y.T@Kinv@y - (1/2)*np.log(det(K)) - (n/2)*np.log(2*np.pi)
    return μ, Σ, logml

def gp_regression_chol(X, y, Xt, k, σn):
    n = len(X)
    K  = k(X, X)+(σn**2)*np.eye(n)
    Km = k(X, Xt)
    Kt = k(Xt, Xt)
    L = cholesky(K)
    α = backsolve(L.T, backsolve(L, y))
    μ = Km.T@α
    v = inv(L)@Km
    Σ = Kt - v.T@v
    logml = -(1/2)*y.T@α - np.sum(np.log(np.diag(L))) - (n/2)*np.log(2*np.pi)
    return μ, Σ, logml

## Parameters 

xlim = (-2, 2)
ylim = (-3, 3)
n_train = 3
n_test = 100
σn = .1
ℓ = 1
ℓs = [.1, .5, 1]
train_sizes = [3, 5, 10]

def f_gen(x):
    return np.sin(x)+np.sin(x*5)+np.cos(x*3)

## Plotting

fig, axs = plt.subplots(3, 3, sharey=True)
fig.set_size_inches(30, 15)

np.random.seed(0)
X_test = np.expand_dims(np.linspace(*xlim, n_test), 1)
X_train_all = np.expand_dims(
    np.random.uniform(xlim[0], xlim[1], size=np.max(train_sizes)), 1)
ϵ_all = σn*np.random.rand(np.max(train_sizes), 1)

for i, ℓ in enumerate(ℓs):
    for j, n_train in enumerate(train_sizes):
        
        k = lambda X, Y: cov_se(X, Y, ℓ=ℓ)
        X_train = X_train_all[:n_train]
        ϵ = ϵ_all[:n_train]
        y_train = f_gen(X_train) + ϵ

        μ, Σ, logml = gp_regression_chol(X_train, y_train, X_test, k, σn)
        std = np.expand_dims(np.sqrt(np.diag(Σ)), 1)

        ax = axs[i, j]
        ax.plot(X_test, μ, color='k')
        ax.fill_between(X_test.squeeze(), (μ-2*std).squeeze(), (μ+2*std).squeeze(), alpha=.2, color=cmap(.3))
        ax.scatter(X_train, y_train, marker='x', color='r', s=50)
        ax.grid()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title('$n=$'+f'{n_train}'+', $-\log p(\mathbf{y}\mid X)$'+f'={-logml.item():.2f}')
        if j == 0:
            ax.set_ylabel('$K_{SE}$'+f'(ℓ={ℓ})')

fig.tight_layout()
plt_savefig(fig, 'summary/assets/plt_gp_regression_vary_trainsize.png')
