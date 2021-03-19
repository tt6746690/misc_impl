import numpy as np
from numpy.linalg import inv, det, cholesky
from numpy.linalg import solve as backsolve

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnp_linalg


def gp_regression_slow(X, y, Xt, k, σn):
    n = len(X)
    K = k(X, X)+(σn**2)*np.eye(n)
    Km = k(X, Xt)
    Kt = k(Xt, Xt)
    Kinv = inv(K)
    μ = Km.T@Kinv@y
    Σ = Kt - Km.T@Kinv@Km
    mll = -(1/2)*y.T@Kinv@y - (1/2)*np.log(det(K)) - (n/2)*np.log(2*np.pi)
    return μ, Σ, mll


def gp_regression_chol(X, y, Xt, k, logsn):
    n = len(X)
    sn2 = jnp.exp(2*logsn)
    if sn2.size == 1:
        sn2I = sn2*jnp.eye(n)
    else:
        # mtgp X[:,1] are task indices
        ind = np.asarray(X[:,1], np.int)
        sn2I = jnp.diag(sn2[ind])
    K = k(X, X)+sn2I
    Km = k(X, Xt)
    Kt = k(Xt, Xt)
    L = jnp_linalg.cholesky(K)
    α = jnp_linalg.solve(L.T, jnp_linalg.solve(L, y))
    μ = Km.T@α
    v = jnp_linalg.inv(L)@Km
    Σ = Kt - v.T@v
    mll = -(1/2)*y.T@α - jnp.sum(jnp.log(jnp.diag(L))) - \
        (n/2)*jnp.log(2*jnp.pi)
    return μ, Σ, mll[0, 0]


def run_sgd(f, params, lr=.002, num_steps=10, log_func=lambda i,x,params: None, optimizer='momentum'):
    import itertools
    from jax import jit, grad
    f = jit(f)
    g = jit(grad(f, argnums=0))
    opt_init, opt_update, get_params = get_optimizer(optimizer, lr=lr)
    opt_state = opt_init(params)
    itercount = itertools.count()
    for i in range(num_steps):
        params = get_params(opt_state)
        log_func(i, f, params)
        params_grad = g(params)
        opt_state = opt_update(next(itercount),params_grad, opt_state)
    return params


def get_optimizer(optimizer, lr=.002):
    from jax.experimental import optimizers
    if optimizer == 'sgd':
        return optimizers.sgd(lr)
    if optimizer == 'momentum':
        return optimizers.momentum(lr, .9)
    