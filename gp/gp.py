import numpy as onp

import jax.numpy as np
import jax.numpy.linalg as np_linalg


def gp_regression_slow(X, y, Xt, k, σn):
    n = len(X)
    K = k(X, X)+(σn**2)*np.eye(n)
    Km = k(X, Xt)
    Kt = k(Xt, Xt)
    Kinv = np_linalg.inv(K)
    μ = Km.T@Kinv@y
    Σ = Kt - Km.T@Kinv@Km
    mll = -(1/2)*y.T@Kinv@y - \
        (1/2)*np.log(np_linalg.det(K)) - \
        (n/2)*np.log(2*np.pi)
    return μ, Σ, mll[0, 0]

def gp_regression_chol(X, y, Xt, k, logsn):
    n = len(X)
    sn2 = np.exp(2*logsn)
    if sn2.size == 1:
        sn2I = sn2*np.eye(n)
    else:
        task_idx = np.asarray(X[:,1], np.int32)
        sn2I = np.diag(sn2[task_idx])
    K = k(X, X)+sn2I
    Km = k(X, Xt)
    Kt = k(Xt, Xt)
    L = np_linalg.cholesky(K)
    α = np_linalg.solve(L.T, np_linalg.solve(L, y))
    μ = Km.T@α
    v = np_linalg.inv(L)@Km
    Σ = Kt - v.T@v
    mll = -(1/2)*y.T@α - np.sum(np.log(np.diag(L))) - \
        (n/2)*np.log(2*np.pi)
    return μ, Σ, mll[0, 0]

def log_func_default(i, f, params, everyn=10):
    if (i+1)%everyn==0:
        S = []
        for k, v in params.items():
            if k.startswith('log'):
                if v.size == 1:
                    s = f'{k}={np.exp(v):.3f}'
                else:
                    s = f'{k}={np.asarray(np.exp(v))}'
                    s = s.replace('\n', ',')
            else:
                s = f'{k}={v}'
            S.append(s)
        S = '\t'.join(S)
        print(f'[{i:3}]:\tLoss={f(params):.3f}\t{S}')

def run_sgd(f, params, lr=.002, num_steps=10,
            log_func=None,
            optimizer='momentum'):
    import itertools
    from jax import jit, grad
    f = jit(f)
    g = jit(grad(f, argnums=0))
    opt_init, opt_update, get_params = get_optimizer(optimizer, lr=lr)
    opt_state = opt_init(params)
    itercount = itertools.count()
    for i in range(num_steps):
        params = get_params(opt_state)
        if log_func is not None: log_func(i, f, params)
        params_grad = g(params)
        opt_state = opt_update(next(itercount),params_grad, opt_state)
    return params


def get_optimizer(optimizer, lr=.002):
    from jax.experimental import optimizers
    if optimizer == 'sgd':
        return optimizers.sgd(lr)
    if optimizer == 'momentum':
        return optimizers.momentum(lr, .9)
    

