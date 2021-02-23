# Reference:
#    https://github.com/ebonilla/mtgp
#    https://proceedings.neurips.cc/paper/2007/file/66368270ffd51418ec58bd793f2d9b1b-Paper.pdf
#    https://gpflow.readthedocs.io/en/master/notebooks/advanced/coregionalisation.html
#    
#
import numpy as np
np.set_printoptions(precision=3,suppress=True)
from sklearn.metrics import mean_squared_error

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, device_put

import matplotlib.pyplot as plt
import matplotlib as mpl
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.family'] = 'Times New Roman'
cmap = plt.cm.get_cmap('bwr')

import sys
sys.path.append('../kernel')
from jaxkern import (cov_se2, LookupKernel, normalize_K)

from plt_utils import plt_savefig, plt_scaled_colobar_ax
from gp import gp_regression_chol, run_sgd


## Parameters

M = 2
n_train = 50
n_test = 50
ylim = (-3,3)
xlim = (-.2,1.2)
σn = [.03, .1]
ℓ = .2
max_mml = True
mtl = True
lr = .002
num_steps = 200
verbose = False

def log_func(f, params):
    if verbose:
        print(f"loss={f(params):.3f}\t"
              f"ℓ={jnp.exp(params['logℓ']):.3f}\t"
              f"σn={np.asarray(jnp.exp(params['logsn']))}\t")

## Data
np.random.seed(0)

B = jnp.eye(M)

X0 = np.random.rand(n_train*2//4, 1) # *.5+.5
X1 = np.random.rand(n_train-len(X0), 1)*.5
X_train = np.vstack((np.hstack((X0, np.zeros_like(X0))),
                     np.hstack((X1, np.ones_like(X1)))))

f0 = lambda X: np.sin(6*X)
f1 = lambda X: np.sin(6*X + 1)
fs = [f0,f1]
Y0 = f0(X0) + np.random.randn(*X0.shape)*σn[0]
Y1 = f1(X1) + np.random.randn(*X1.shape)*σn[1]
y_train = np.vstack((Y0,Y1))

X_test = np.vstack((np.tile(np.linspace(xlim[0], xlim[1], n_test), M),
                    np.hstack([t*np.ones(n_test) for t in range(M)]))).T

## Plotting

colors_b = [cmap(.1), cmap(.3)]
colors_r = [cmap(.9), cmap(.7)]

gridspec_kw = {'width_ratios': [2, 1], 'height_ratios': [1, 1]}
fig, axs = plt.subplots(2, 2, gridspec_kw=gridspec_kw)
fig.set_size_inches(15, 10)


for i, mtl in enumerate([False, True]):
    ax = axs[i, 0]

    ## Training

    def mtgp_k(XT, XTp, logℓ, B):
        X, Xp = XT[:,0], XTp[:,0]
        Kx = cov_se2(X, Xp, logℓ=logℓ)
        T, Tp = np.asarray(XT[:,1], np.int), np.asarray(XTp[:,1], np.int)
        Kt = LookupKernel(T, Tp, B)
        K = Kx*Kt
        return K

    if max_mml:
        if mtl:
            def nmll(params):
                L = jnp.exp(params['logL'])
                B = L@L.T
                k = lambda X, Y: mtgp_k(X, Y, params['logℓ'], B)
                μ, Σ, mll = gp_regression_chol(
                    X_train, y_train, X_test, k, logsn=params['logsn'])
                return -mll
            params = {'logℓ': jnp.log(1.),
                      'logsn': jnp.log(.1*jnp.ones(M)),
                      'logL': jnp.log(jnp.array(np.random.rand(M,M)))}
            res = run_sgd(nmll, params, lr=lr, num_steps=num_steps, log_func=log_func)
            logℓ, logsn = res['logℓ'].item(), res['logsn']
            ℓ, σn = jnp.exp(logℓ), jnp.exp(logsn)
            L = jnp.exp(params['logL'])
            B = L@L.T
        else:
            def nmll(params):
                k = lambda X, Y: mtgp_k(X, Y, params['logℓ'], B)
                μ, Σ, mll = gp_regression_chol(
                    X_train, y_train, X_test, k, logsn=params['logsn'])
                return -mll
            params = {'logℓ': jnp.log(1.),
                      'logsn': jnp.log(.1*jnp.ones(M))}
            res = run_sgd(nmll, params, lr=lr, num_steps=num_steps, log_func=None)
            logℓ, logsn = res['logℓ'].item(), res['logsn']
            ℓ, σn = jnp.exp(logℓ), jnp.exp(logsn)

    ## Plotting

    k = lambda X, Y: mtgp_k(X, Y, logℓ, B)
    μ, Σ, mll = gp_regression_chol(X_train, y_train, X_test, k, logsn)
    std = np.expand_dims(np.sqrt(np.diag(Σ)), 1)

    for t in range(M):
        # task-specific mll
        I = X_test[:,1] == t
        # posterior predictive distribution
        X_test_, μ_, std_ = X_test[I,0].squeeze(), μ[I].squeeze(), std[I].squeeze()
        ax.plot(X_test_, μ_, color=colors_b[t], lw=2)
        ax.fill_between(X_test_, μ_-2*std_, μ_+2*std_, alpha=.2, color=colors_b[t])
        # generating function for main task
        if t == 1:
            ax.plot(X_test_, fs[t](X_test_), color='k', linestyle='dashed', linewidth=1)
        
        mse = mean_squared_error(μ[I], f1(X_test[I,0]))
        # train data points
        I = X_train[:,1] == t
        ax.scatter(X_train[I,0], y_train[I],
                   marker='x', color=colors_r[t], s=50,
                   label=f'Task {t}'+' ($\sigma_n$'+f'={σn[t]:.2f}, '+'$mse$'+f'={mse:.3f})')
        
    ax.grid()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(fontsize=15)
    title = '$\ell$'+f'={ℓ:.2f}'+ \
        ' $B_{01}/B_{00}$'+f'={B[0,1]*2/(B[0,0]+B[1,1]):.2f}'+ \
        ' $-mll$'+f'={-mll:.2f}'
    ax.set_title(title, fontsize=30)
    

    ax = axs[i, 1]
    XX = np.vstack((X_train, X_test[X_test[:,1]==1]))
    K = k(XX, XX)
    im = ax.imshow(normalize_K(K), cmap=cmap)
    fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))
    ax.set_title('$K(X_{train}, X_{test@1})$')
    
fig.tight_layout()
plt_savefig(fig, 'summary/assets/plt_mtgp.png')