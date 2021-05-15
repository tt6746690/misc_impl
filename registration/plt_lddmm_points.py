## Geodesic shooting for registering simple shapes
# 
from functools import partial

import jax
from jax import random, grad
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

n = 4*10
circ_radius = .3
circ_center = (-.5, 0)
rect_center = (.5,  0)
rect_radius = .3
xlim = (-1, 1)
ylim = (-1, 1)
ℓ = .10
euler_steps = 10
δt = .1
grid_nlines = 11
ot_ρ = 1e5
ot_ϵ = (.05)**2
ot_n_iters=100
λ_regu = 1  
lr = .1
n_steps = 10

## partials
# k = jax.jit(partial(cov_se, ℓ=ℓ))

@jax.jit
def k(X, Y=None):
    return cov_se(X,Y,σ2=1.,ℓ=.025) + cov_se(X,Y,σ2=.75,ℓ=.15)


shooting_step = jax.jit(partial(HamiltonianStep, k=k, δt=δt))
shooting = jax.jit(partial(HamiltonianShooting, k=k, euler_steps=euler_steps, δt=δt))
regu_fn = jax.jit(partial(Hqp, k=k))
data_fn = jax.jit(partial(sinkhorn_log_stabilized, ϵ=ot_ϵ, ρ=ot_ρ, n_iters=ot_n_iters))
cost_fn = jax.jit(sqdist)


## Data

key = random.PRNGKey(5)

θ = np.linspace(0, 2*np.pi, n)
X = np.array(circ_center) + circ_radius*np.column_stack((np.cos(θ), np.sin(θ)))

Yx = np.hstack((np.linspace(-rect_radius,rect_radius,n//4),
                np.linspace(-rect_radius,rect_radius,n//4),
                -rect_radius + np.zeros((n//4,)),
                +rect_radius + np.zeros((n//4,)),)) + rect_center[0]
Yy = np.hstack((+rect_radius + np.zeros((n//4,)),
                -rect_radius + np.zeros((n//4,)),
                np.linspace(-rect_radius,rect_radius,n//4),
                np.linspace(-rect_radius,rect_radius,n//4),)) + rect_center[1]
Y = np.column_stack((Yx, Yy))

g0, gL = GridData(nlines=grid_nlines, xlim=xlim, ylim=ylim)

p0 = random.normal(key, (n, 2))*.1
p0 = np.zeros(X.shape, dtype=np.float32) # zero solution gives smooth momentum !

# uniform measures on point sets
μ = np.ones((q.shape[0],))
ν = np.ones((y.shape[0],))


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


loss_fn_capture = jax.jit(partial(loss_fn, x=X, μ=μ, y=Y, ν=ν, g=g0, λ_regu=λ_regu))
value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn_capture, has_aux=True))


def plt_shape(ax, q, y, gX, gL):
    ax.scatter(q[:,0], q[:,1], color=cmap(.9))
    ax.scatter(y[:,0], y[:,1], color=cmap(.1))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt_grid(ax, gX, gL)
    

n_steps = 500
params = {'p0': p0}

opt_init, opt_update, get_params = optimizers.sgd(step_size=.0003)
opt_state = opt_init(params)

axi = 0
display_its = [int(x*n_steps) 
               for x in [0.,.25,.5,1-1/n_steps]]
fig, axs = plt.subplots(2, len(display_its),
                        figsize=(5*len(display_its), 5*2), sharex=True, sharey=True)

for it in range(n_steps):
    
    params = get_params(opt_state)
    (loss, info), grads = value_and_grad_fn(params)
    opt_state = opt_update(it, grads, opt_state)
    
    print(f'loss={info["loss"]:6.3f} ({info["loss_data"]:6.3f}+{info["loss_regu"]:6.3f})')
    
    
    if it in display_its:

        ax = axs[0,axi]
        plt_shape(ax, X, Y, g0, gL)
        plt_vectorfield(ax, X, params['p0'], color=cmap(.8))
        if it == display_its[0]: ax.set_ylabel('momentum')
            
        ax = axs[1,axi]
        plt_shape(ax, info['q1'], Y, info['g1'], gL)
        plt_vectorfield(ax, gX, k(gX, info['q1'])@info['p1'], scale=10, color='k')
        if it == display_its[0]: ax.set_ylabel('shooting')
        ax.set_title(f't={it}')
        
        axi += 1


fig.tight_layout()
plt_savefig(fig, 'summary/assets/plt_shooting.png')

