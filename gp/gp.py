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


def gp_regression_chol(X, y, Xt, k, σn):
    n = len(X)
    K = k(X, X)+(σn**2)*jnp.eye(n)
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


def run_sgd(f, params, lr=.002, num_steps=10, log_func=None):
    import itertools
    from jax import jit, grad
    from jax.experimental import optimizers
    f = jit(f)
    g = jit(grad(f, argnums=0))
    opt_init, opt_update, get_params = optimizers.sgd(lr)
    opt_state = opt_init(params)
    itercount = itertools.count()
    for _ in range(num_steps):
        params = get_params(opt_state)
        if log_func: log_func(f, params)
        params_grad = g(params)
        opt_state = opt_update(next(itercount),params_grad, opt_state)
    return params
