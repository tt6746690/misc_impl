from IPython.core.getipython import get_ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import itertools
import time

import numpy as np
import numpy.random as npr
import scipy.stats as stats

import matplotlib.pyplot as plt
from jaxkern import (
    rbf_kernel, linear_kernel, estimate_sigma_median, hsic, cka, mmd, squared_l2_norm)

import jax
from jax.experimental import optimizers
from jax import grad, random
import jax.numpy as jnp

def plt_kde(X, ax=None, lim=None):
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(6,6)
    if lim is None:
        lim = np.min(X)-1, np.max(X)+1
    XX,YY = np.meshgrid(
        np.linspace(lim[0], lim[1], 100),
        np.linspace(lim[0], lim[1], 100))
    XY = np.vstack([XX.ravel(), YY.ravel()])
    kernel = stats.gaussian_kde(X.T)
    Z = kernel(XY).reshape(XX.shape)
    ax.imshow(Z, cmap='Blues', extent=[lim[0], lim[1], lim[0], lim[1]])
    ax.set_xlim([lim[0], lim[1]])
    ax.set_ylim([lim[0], lim[1]])
    if ax is None:
        return fig, ax
    else:
        return ax

## Parameters
n = 500
d = 2
lr = .1
nprint = 10
num_steps = 50
batch_size = 50
plt_fig = False

## Plotting
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.family'] = 'Times New Roman'
fig, axs = plt.subplots(1, num_steps//nprint, sharey=True)
fig.set_size_inches(5*(num_steps//nprint), 5)

## Data
npr.seed(0)
mu  = np.ones(d)
cov = np.eye(d)
Xdist = stats.multivariate_normal(mu, cov)
X = Xdist.rvs(size=(n,))
X = jax.device_put(X)

## Kernel
sigma = estimate_sigma_median(X)
gamma = 1/(2*(sigma**2))
print(f'sigma={sigma}')
print(f'gamma={gamma}')
def kernel(X,Y):
    return rbf_kernel(X,Y,gamma=gamma)

a1 = jnp.ones(d, dtype=jnp.float32)/d
params = {
    'a1': a1,
    'a2': a1*0.98,
}

def f(params, X):
    return params['a1']@X
def g(params, X):
    return params['a2']@X
f = jax.vmap(f, (None, 0), 0)
g = jax.vmap(g, (None, 0), 0)

def loss_fn(params, X):
    fX = f(params, X)
    gX = g(params, X)
    loss_hsic = hsic(fX, gX, kernel, kernel) 
    loss_l2 = squared_l2_norm(jnp.vstack((params['a1'],params['a2']))@mu.T - jnp.array([1,1]))
    return loss_hsic + loss_l2
loss_fn = jax.jit(loss_fn)
grad_fn = jax.jit(grad(loss_fn))


num_complete_batches, leftover = divmod(n, batch_size)
num_batches = num_complete_batches + bool(leftover)
def data_stream():
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(n)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield X[batch_idx]
batches = data_stream()

opt_init, opt_update, get_params = optimizers.sgd(lr)
opt_state = opt_init(params)

itercount = itertools.count()
for i in range(num_steps):
    start_time = time.time()
    for j in range(num_batches):
        batch = next(batches)
        params = get_params(opt_state)
        loss_hsic = hsic(f(params, X), g(params, X), kernel, kernel) 
        params_grad = grad_fn(params, batch)
        opt_state = opt_update(next(itercount), params_grad, opt_state)
    epoch_time = time.time() - start_time

    if i%nprint == 0:
        print(f'[{i:3}] time={epoch_time:4.4f}\t hsic={loss_hsic:5.8f}\t'
              f'a1={params["a1"]}\t'
              f'a2={params["a2"]}')

        ## Plotting
        ax = axs[i//nprint]
        lim = [-4,6]
        XX, YY = np.meshgrid(
            np.linspace(lim[0], lim[1], 100),
            np.linspace(lim[0], lim[1], 100))
        XY = np.vstack([XX.ravel(), YY.ravel()])
        # contour
        A = jnp.vstack((params['a1'], params['a2']))
        ymu, ycov = A@mu.T, A@cov@A.T
        Ydist = stats.multivariate_normal(ymu, ycov)
        Z = Ydist.pdf(XY.T).reshape(XX.shape)
        ax.contour(XX, YY, Z, cmap='Reds', linestyles='-',
                   levels=np.linspace(np.min(Z), np.max(Z), 5))
        # kde density
        y = jnp.hstack([f(params, X).reshape(-1,1), g(params, X).reshape(-1,1)])
        kde_kernel = stats.gaussian_kde(y.T)
        Z = kde_kernel(XY).reshape(XX.shape)
        ax.imshow(jnp.rot90(Z), cmap='Reds', extent=[lim[0], lim[1], lim[0], lim[1]])
        # scatter
        ax.scatter(y[:,0], y[:,1], c='k', s=3)

        ax.set_title(f'HSIC={hsic(y[:,0], y[:,1], kernel, kernel):5.4f}')
        ax.set_xlim([lim[0], lim[1]])
        ax.set_ylim([lim[0], lim[1]])
        ax.set_xticks([])
        ax.set_yticks([])


plt.tight_layout()
save_path = './summary/assets/hsic_2d_gaussian.png'
fig.savefig(save_path, bbox_inches='tight', dpi=100)