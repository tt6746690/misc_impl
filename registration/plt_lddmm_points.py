## Geodesic shooting for registering simple shapes
# 
from functools import partial

import os
os.environ['JAX_ENABLE_X64'] = 'False'

import jax
from jax import random, grad, jit, value_and_grad
import jax.numpy as np
import jax.numpy.linalg as linalg
from jax.experimental import optimizers

import matplotlib.pyplot as plt
import matplotlib as mpl
# https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['font.size'] = 25
mpl.rcParams['font.family'] = 'Times New Roman'
cmap = plt.cm.get_cmap('bwr')


import sys
sys.path.append('../gp')
sys.path.append('../ot')
from plt_utils import plt_savefig
from otax import sinkhorn_log_stabilized, sinkhorn_divergence

from jax_registration import *

## Parameters

n = 4*20
nlayers = 4
circ_radius = .3
circ_center = (-.5, 0)
rect_center = (.5,  0)
rect_radius = .3
shapes = (square, circle)
fill = False
xlim = (-1, 1)
ylim = (-1, 1)
ℓ = .1
euler_steps = 5
δt = .15
grid_nlines = 11
ot_ρ = 1e5
ot_ϵ = (.05)**2
ot_n_iters=100
λ_regu = .1

## partials

k = jax.jit(partial(cov_se, ℓ=ℓ))

@jax.jit
def k(X, Y=None):
    return cov_se(X,Y,σ2=.5,ℓ=.025) + cov_se(X,Y,σ2=.3,ℓ=.15) + cov_se(X,Y,σ2=.2,ℓ=.3)

shooting_step = jit(partial(HamiltonianStep, k=k, δt=δt))
shooting = jit(partial(HamiltonianShooting, k=k, euler_steps=euler_steps, δt=δt))
regu_fn = jit(partial(Hqp, k=k))
data_fn = jit(partial(sinkhorn_log_stabilized, ϵ=ot_ϵ, ρ=ot_ρ, n_iters=ot_n_iters))
cost_fn = jit(sqdist)


## Data

key = random.PRNGKey(5)
if fill:
    X = np.vstack((shapes[0](circ_center, circ_radius*((i+1)/nlayers), n//nlayers)
                   for i in range(nlayers)))
    Y = np.vstack((shapes[1](rect_center, rect_radius*((i+1)/nlayers), n//nlayers)
                   for i in range(nlayers)))
else:
    X = shapes[0](circ_center, circ_radius, n)
    Y = shapes[1](rect_center, rect_radius, n)
g0, gL = GridData(nlines=grid_nlines, xlim=xlim, ylim=ylim, nsubticks=6)

p0 = np.zeros(X.shape) * 1.
μ = np.ones((X.shape[0],))
ν = np.ones((Y.shape[0],))


def plt_shape(ax, q, y):
    ax.scatter(y[:,0], y[:,1], color=cmap(.1), marker='x')
    ax.scatter(q[:,0], q[:,1], color=cmap(.9), marker='o')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

fig, ax = plt.subplots(1,1,figsize=(10,10))
plt_shape(ax, X, Y)
plt_grid(ax, g0, gL)


## Plotting

def loss_fn(params, x, μ, y, ν, g, λ_regu):
    """ Compute loss    D(q1, μ, y, ν) + λ*R(q̇)
            - geodeisc shooting obtain q1
            - compute data matching term D(q1, y)
    """
    p = params['p0']
    q1, p1, g1 = shooting(x, p, g)

    C = cost_fn(q1, y)
    π, loss_data = data_fn(μ, ν, C)
    loss_regu = regu_fn(x, p) * λ_regu

    loss = loss_regu + loss_data

    return loss, {'loss': loss,
                  'loss_regu': loss_regu,
                  'loss_data': loss_data,
                  'π': π,
                  'q1': q1, 'p1': p1, 'g1': g1}

loss_fn_capture = jit(partial(loss_fn, x=X, μ=μ, y=Y, ν=ν, g=g0, λ_regu=λ_regu))
value_and_grad_fn = jit(value_and_grad(loss_fn_capture, has_aux=True))
    

n_steps = 500
lr = .003
params = {'p0': p0}

opt_init, opt_update, get_params = optimizers.adam(step_size=lr)
opt_state = opt_init(params)

axi = 0
display_its = [int(x*n_steps) 
               for x in [0.,.03, .1,.3,1-1/n_steps]]
fig, axs = plt.subplots(2, len(display_its),
                        figsize=(5*len(display_its), 5*2), sharex=True, sharey=True)


def plt_momentum_shooting(axs, p0, q1, g1):
    ax = axs[0]
    plt_grid(ax, g0, gL)
    plt_vectorfield(ax, g0, k(g0, X)@p0, scale=None, color='k')
    plt_shape(ax, X, Y)
    plt_vectorfield(ax, X, p0, color=cmap(.8), scale=.4)

    ax = axs[1]
    plt_grid(ax, g1, gL)
    plt_shape(ax, q1, Y)

for it in range(n_steps):
    
    params = get_params(opt_state)
    (loss, info), grads = value_and_grad_fn(params)
    opt_state = opt_update(it, grads, opt_state)
    
    if it%(n_steps//10) == 0:
        print(f'[{it:4}] loss={info["loss"]:7.3f}'
              f'({info["loss_data"]:7.3f} +{info["loss_regu"]:7.3f})')
    
    if it in display_its:
        plt_momentum_shooting(axs[:,axi], params['p0'], info['q1'], info['g1'])
        axs[0,axi].set_title(f't={it}', fontsize=40)
        axi += 1


fig.tight_layout()
plt_savefig(fig, f'summary/assets/plt_lddmm_points_{fill}.png')



fig, axs = plt.subplots(1,3,figsize=(15,5))
ax = axs[0]
plt_shape(ax, X, Y)
plt_grid(ax, g0, gL)
plt_momentum_shooting(axs[1:], params['p0'], info['q1'], info['g1'])
fig.tight_layout()
plt_savefig(fig, f'summary/assets/plt_lddmm_points_{fill}_summary.png')

