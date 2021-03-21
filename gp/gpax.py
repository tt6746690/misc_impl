from typing import Any, Callable, Sequence, Optional, Tuple

import jax
from jax import random
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
    
    def pred_y(self, Xs):
        σ2 = np.exp(2*self.logσn)
        μf, Σf = self.pred_f(Xs)
        ns = len(Σf)
        μy, Σy = μf, Σf + σ2*np.diag(np.ones((ns,)))
        return μy, Σy
        

class InducingPoints(nn.Module):
    
    def setup(self):
        init_fn = nn.initializers.zeros
        self.Xu = self.params('Xu', init_fn, ())
        
        
        self.logℓ = self.param('logℓ', init_fn, (1,))
        self.logσ = self.param('logσ', init_fn, (1,))

    def __call__(self, X, Y=None):
        return cov_se(X, Y, logℓ=self.logℓ, logσ=self.logσ)


    
    
class GPRFITC(nn.Module):
    data: Tuple[np.ndarray, np.ndarray]
    n_inducing: int

    def setup(self):
        self.k = CovSE()
        self.logσn = self.param('logσn',
                                nn.initializers.zeros, (1,))
        X, y = self.data
        def randsub_init_fn(key, shape, dtype=np.float32):
            idx = random.choice(key, np.arange(len(X)),
                                shape=(shape[0],), replace=False)
            return X[idx]
        self.Xu = self.param('Xu', randsub_init_fn,
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
        
        K = k(X)
        Kuu = k(Xu, Xu)
        Kuf = k(Xu, X)
        Luu = linalg.cholesky(Kuu)
        V = linalg.solve(Luu, Kuf)
        Qff = V.T@V
        Λ = np.diag(K) - np.diag(Qff) + σ2
        Λ = Λ.reshape(-1, 1)
        B = np.eye(m) + (V/Λ.T)@V.T
        LB = linalg.cholesky(B)
        γ = linalg.solve(LB, (V/Λ.T)@y)

        return Luu, Λ, LB, γ
    
    def mll(self):
        X, y = self.data
        n = len(X)
        Luu, Λ, LB, γ = self.precompute()
        
        mll_quad  = -.5*( sum(y/Λ*y) - sum(np.square(γ)) )[0]
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
        ω = linalg.solve(Luu, Kus)
        ν = linalg.solve(LB, ω)
        
        μ = ω.T@linalg.solve(LB.T, γ)
        Σ = Kss - ω.T@ω + ν.T@ν
        
        return μ, Σ

    def pred_y(self, Xs):
        σ2 = np.exp(2*self.logσn)
        μf, Σf = self.pred_f(Xs)
        ns = len(Σf)
        μy, Σy = μf, Σf + σ2*np.diag(np.ones((ns,)))
        return μy, Σy



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

