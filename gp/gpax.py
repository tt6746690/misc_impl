from typing import Any, Callable, Sequence, Optional, Tuple

import jax
from jax import random
import jax.numpy as np
import jax.numpy.linalg as linalg
from jax.scipy.linalg import cho_solve, solve_triangular

import flax
from flax.core import unfreeze
from flax import optim
from flax import linen as nn

from jaxkern import cov_se


def cholesky_jitter(K, jitter=1e-6):
    L = linalg.cholesky(K+jitter*np.eye(len(K)))
    return L


class CovSE(nn.Module):

    def setup(self):
        init_fn = nn.initializers.zeros
        self.logℓ = self.param('logℓ', init_fn, (1,))
        self.logσ = self.param('logσ', init_fn, (1,))

    def __call__(self, X, Y=None, diag=False):
        if diag:
            return np.full((len(X),), np.exp(2*self.logσ)[0])
        else:
            return cov_se(X, Y, logℓ=self.logℓ, logσ=self.logσ)


class GPR(nn.Module):
    data: Tuple[np.ndarray, np.ndarray]
        
    def setup(self):
        self.k = CovSE()
        self.logσn = self.param('logσn', nn.initializers.zeros, (1,))
        
    def get_init_params(self, key):
        params = self.init(key, np.ones((1, self.data[0].shape[-1])),
                           method=self.pred_f)
        return params

    def mll(self):
        X, y = self.data
        k = self.k
        n = len(X)
        σ2 = np.exp(2*self.logσn)
        
        K = k(X) + σ2*np.eye(n)
        L = linalg.cholesky(K)
        α = cho_solve((L, True), y)

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
        
        K = k(X) + σ2*np.eye(n)
        Ks = k(X, Xs)
        Kss = k(Xs, Xs)
        L = linalg.cholesky(K)
        α = cho_solve((L, True), y)
        μ = Ks.T@α
        v = solve_triangular(L, Ks, lower=True)
        Σ = Kss - v.T@v
        
        return μ, Σ
    
    def pred_y(self, Xs):
        σ2 = np.exp(2*self.logσn)
        μf, Σf = self.pred_f(Xs)
        ns = len(Σf)
        μy, Σy = μf, Σf + σ2*np.diag(np.ones((ns,)))
        return μy, Σy


class GPRFITC(nn.Module):
    data: Tuple[np.ndarray, np.ndarray]
    n_inducing: int

    def setup(self):
        self.k = CovSE()
        self.logσn = self.param('logσn',
                                nn.initializers.zeros, (1,))
        X, y = self.data
        self.Xu = self.param('Xu', lambda k,s : X[:self.n_inducing],
                             (self.n_inducing, X.shape[-1]))

    def get_init_params(self, key):
        params = self.init(key, np.ones((1, self.data[0].shape[-1])),
                           method=self.pred_f)
        return params
    
    def precompute(self):
        X, y = self.data
        k = self.k
        σ2 = np.exp(2*self.logσn)
        Xu = self.Xu
        n, m = len(X), self.n_inducing
        
        Kdiag = k(X, diag=True)
        Kuu = k(Xu, Xu)
        Kuf = k(Xu, X)
        Luu = cholesky_jitter(Kuu, jitter=1e-4)
        
        V = solve_triangular(Luu, Kuf, lower=True)
        Qffdiag = np.sum(np.square(V), axis=0)
        Λ = Kdiag - Qffdiag + σ2
        Λ = Λ.reshape(-1,1)
        
        B = np.eye(m) + (V/Λ.T)@V.T
        LB = cholesky_jitter(B, jitter=1e-5)
        γ = solve_triangular(LB, V@(y/Λ), lower=True)

        return Luu, Λ, LB, γ
    
    def mll(self):
        X, y = self.data
        n = len(X)
        Luu, Λ, LB, γ = self.precompute()
        
        mll_quad  = -.5*( np.sum((y/Λ)*y) - np.sum(np.square(γ)) )
        mll_det   = -np.sum(np.log(np.diag(LB)))-.5*np.sum(np.log(Λ))
        mll_const = -(n/2)*np.log(2*np.pi)
        mll = mll_quad + mll_det + mll_const

        return mll
    
    def pred_f(self, Xs):
        X, y = self.data
        k = self.k
        Xu = self.Xu
        n = len(X)
        Luu, Λ, LB, γ = self.precompute()
        
        Kss = k(Xs, Xs)
        Kus = k(Xu, Xs)
        ω = solve_triangular(Luu, Kus, lower=True)
        ν = solve_triangular(LB, ω, lower=True)
        
        μ = ω.T@solve_triangular(LB.T, γ, lower=False)
        Σ = Kss - ω.T@ω + ν.T@ν
        
        return μ, Σ

    def pred_y(self, Xs):
        σ2 = np.exp(2*self.logσn)
        μf, Σf = self.pred_f(Xs)
        ns = len(Σf)
        μy, Σy = μf, Σf + σ2*np.diag(np.ones((ns,)))
        return μy, Σy


class VFE(nn.Module):
    data: Tuple[np.ndarray, np.ndarray]
    n_inducing: int

    def setup(self):
        self.k = CovSE()
        self.logσn = self.param('logσn',
                                nn.initializers.zeros, (1,))
        X, y = self.data
        self.Xu = self.param('Xu', lambda k,s : X[:self.n_inducing],
                             (self.n_inducing, X.shape[-1]))

    def get_init_params(self, key):
        params = self.init(key, np.ones((1, self.data[0].shape[-1])),
                           method=self.pred_f)
        return params
    
    def precompute(self):
        X, y = self.data
        k = self.k
        σ2 = np.exp(2*self.logσn)
        Xu = self.Xu
        n, m = len(X), self.n_inducing
        
        Kdiag = k(X, diag=True)
        Kuu = k(Xu, Xu)
        Kuf = k(Xu, X)
        Luu = cholesky_jitter(Kuu, jitter=1e-4)
        
        V = solve_triangular(Luu, Kuf, lower=True)
        Qffdiag = np.sum(np.square(V), axis=0)
        Λ = Kdiag - Qffdiag + σ2
        Λ = Λ.reshape(-1,1)
        
        B = np.eye(m) + (V/Λ.T)@V.T
        LB = cholesky_jitter(B, jitter=1e-5)
        γ = solve_triangular(LB, V@(y/Λ), lower=True)

        return Kdiag, Luu, V, Λ, LB, γ
    
    def mll(self):
        X, y = self.data
        n = len(X)
        σ2 = np.exp(2*self.logσn)[0]
        Kdiag, Luu, V, Λ, LB, γ = self.precompute()
        
        elbo_quad  = -.5*( np.sum((y/Λ)*y) - np.sum(np.square(γ)) )
        elbo_det   = -np.sum(np.log(np.diag(LB)))-.5*np.sum(np.log(Λ))
        elbo_const = -(n/2)*np.log(2*np.pi)
        elbo_trcc  = -(1/2/σ2)*( np.sum(Kdiag) - np.sum(np.square(V)) )
        
        elbo = elbo_quad + elbo_det + elbo_const + elbo_trcc
        return elbo
    
    def pred_f(self, Xs):
        X, y = self.data
        k = self.k
        Xu = self.Xu
        n = len(X)
        Kdiag, Luu, V, Λ, LB, γ = self.precompute()
        
        Kss = k(Xs, Xs)
        Kus = k(Xu, Xs)
        ω = solve_triangular(Luu, Kus, lower=True)
        ν = solve_triangular(LB, ω, lower=True)
        
        μ = ω.T@solve_triangular(LB.T, γ, lower=False)
        Σ = Kss - ω.T@ω + ν.T@ν
        
        return μ, Σ

    def pred_y(self, Xs):
        σ2 = np.exp(2*self.logσn)
        μf, Σf = self.pred_f(Xs)
        ns = len(Σf)
        μy, Σy = μf, Σf + σ2*np.diag(np.ones((ns,)))
        return μy, Σy


def randsub_init_fn(key, shape, dtype=np.float32, X=None):
    idx = random.choice(key, np.arange(len(X)),
                        shape=(shape[0],), replace=False)
    return X[idx]


proc_leaf_logscalar = lambda k, v: \
    (k.split('log')[1], np.exp(v[0])) if (k.startswith('log') and v.size==1) else (k, v)
proc_leaf_logvector = lambda k, v: \
    (k.split('log')[1], np.exp(v)) if (k.startswith('log') and v.size>1) else (k, v)
VEC_LENGTH_LIMIT = 5
prof_leaf_veclength = lambda k, v: \
    (f'{k}[:{VEC_LENGTH_LIMIT}]', v[:VEC_LENGTH_LIMIT]) if isinstance(v, np.ndarray) and v.size>1 else (k, v)
prof_leaf_vecsqueeze = lambda k, v: \
    (k, v.squeeze()) if isinstance(v, np.ndarray) else (k, v)
prof_leaf_fns = [proc_leaf_logscalar,
                 proc_leaf_logvector,
                 prof_leaf_veclength,
                 prof_leaf_vecsqueeze]

def log_func_simple(i, f, params, everyn=10):
    if i%everyn==0:
        print(f'[{i:3}]\tLoss={f(params):.3f}')


def log_func_default(i, f, params, everyn=20):
    if i%everyn == 0:
        flattened = flax.traverse_util.flatten_dict(unfreeze(params['params']))
        S = []
        for k, v in flattened.items():
            lk = k[-1]
            for proc in prof_leaf_fns:
                lk, v = proc(lk, v)
            k = list(k)
            k[-1] = lk
            k = '.'.join(k)
            S.append(f'{k}={v:.3f}' if v.size==1 else f'{k}={v}')

        S = '\t'.join(S)
        print(f'[{i:3}]\tLoss={f(params):.3f}\t{S}')


def flax_create_optimizer(params, optimizer, optimizer_kwargs):
    optimizer_cls = getattr(optim, optimizer)
    return optimizer_cls(**optimizer_kwargs).create(params)


def flax_run_optim(f, params, num_steps=10, log_func=None,
                   optimizer='GradientDescent',
                   optimizer_kwargs={'learning_rate': .002}):
    import itertools
    fg_fn = jax.value_and_grad(f)
    opt = flax_create_optimizer(
        params, optimizer=optimizer, optimizer_kwargs=optimizer_kwargs)
    itercount = itertools.count()
    for i in range(num_steps):
        fx, grad = fg_fn(opt.target)
        opt = opt.apply_gradient(grad)
        if log_func is not None: log_func(i, f, opt.target)
    return opt.target


def is_psd(x):
    return np.all(linalg.eigvals(x) > 0)