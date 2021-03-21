from typing import Any, Callable, Sequence, Optional, Tuple

import jax
import jax.numpy as np
import jax.numpy.linalg as linalg

import flax
from flax import linen as nn

from jaxkern import cov_se

class CovSE(nn.Module):
    
    def setup(self):
        init_fn = nn.initializers.zeros
        self.logℓ = self.param('logℓ', init_fn, (1,))
        self.logσ = self.param('logσ', init_fn, (1,))

    def __call__(self, X, Y=None):
        return cov_se(X, Y, logℓ=self.logℓ, logσ=self.logσ)

    
class Gpr(nn.Module):
    data: Tuple[np.ndarray, np.ndarray]
        
    def setup(self):
        self.k = CovSE()
        self.logσn = self.param('logσn', nn.initializers.zeros, (1,))
        
    def get_init_params(self, key):
        params = self.init(key, np.ones((1, self.data[0].shape[-1])),
                           method=self.pred_f)['params']
        return params

    def mll(self):
        X, y = self.data
        k = self.k
        n = len(X)
        σ2 = np.exp(2*self.logσn)
        
        K = k(X, X) + σ2*np.eye(n)
        L = linalg.cholesky(K)
        α = linalg.solve(L.T, linalg.solve(L, y))

        mll_quad  = -(1/2)*np.sum(y*α)
        mll_det   = - np.sum(np.log(np.diag(L)))
        mll_const = - (n/2)*np.log(2*np.pi)
        mll = mll_quad + mll_det + mll_const

        return mll
    
    def pred_f(self, Xs):
        X, y = self.data
        k = self.k
        n = len(X)
        σ2 = np.exp(2*self.logσn)
        
        K = k(X, X) + σ2*np.eye(n)
        Ks = k(X, Xs)
        Kss = k(Xs, Xs)
        L = linalg.cholesky(K)
        α = linalg.solve(L.T, linalg.solve(L, y))
        μ = Ks.T@α
        v = linalg.inv(L)@Ks
        Σ = Kss - v.T@v
        
        return μ, Σ

def gp_regression_slow(X, y, Xt, k, σn):
    n = len(X)
    K = k(X, X)+(σn**2)*np.eye(n)
    Km = k(X, Xt)
    Kt = k(Xt, Xt)
    Kinv = linalg.inv(K)
    μ = Km.T@Kinv@y
    Σ = Kt - Km.T@Kinv@Km
    mll = -(1/2)*y.T@Kinv@y - \
        (1/2)*np.log(linalg.det(K)) - \
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
    L = linalg.cholesky(K)
    α = linalg.solve(L.T, linalg.solve(L, y))
    μ = Km.T@α
    v = linalg.inv(L)@Km
    Σ = Kt - v.T@v
    mll = -(1/2)*y.T@α - np.sum(np.log(np.diag(L))) - \
        (n/2)*np.log(2*np.pi)
    return μ, Σ, mll[0, 0]


def log_func_simple(i, f, params):
    print(f'[{i:3}]\tLoss={f(params):.3f}')


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
    opt_init, opt_update, get_params = jax_opt_get_optimizer(optimizer, lr=lr)
    opt_state = opt_init(params)
    itercount = itertools.count()
    for i in range(num_steps):
        params = get_params(opt_state)
        if log_func is not None: log_func(i, f, params)
        params_grad = g(params)
        opt_state = opt_update(next(itercount), params_grad, opt_state)
    return params


def jax_opt_get_optimizer(optimizer, lr=.002):
    from jax.experimental import optimizers
    if optimizer == 'sgd':
        return optimizers.sgd(lr)
    if optimizer == 'momentum':
        return optimizers.momentum(lr, .9)


def flax_create_optimizer(params, lr, optimizer='GradientDescent'):
    from flax import optim
    optimizer_cls = getattr(optim, optimizer)
    return optimizer_cls(learning_rate=lr).create(params)


def flax_run_optim(f, params, lr=.002, num_steps=10,
                   log_func=None,
                   optimizer='GradientDescent'):
    import itertools
    f = jax.jit(f)
    fg_fn = jax.value_and_grad(f)
    opt = flax_create_optimizer(params, lr=lr, optimizer=optimizer)
    itercount = itertools.count()
    for i in range(num_steps):
        fx, grad = fg_fn(opt.target)
        if log_func is not None: log_func(i, f, opt.target)
        opt = opt.apply_gradient(grad)
    return opt.target
