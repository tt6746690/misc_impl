import math
from typing import Any, Callable, Sequence, Optional, Tuple, Union, List
from dataclasses import dataclass, field

import jax
from jax import random, device_put
import jax.numpy as np
import jax.numpy.linalg as linalg
from jax.scipy.linalg import cho_solve, solve_triangular

import flax
from flax.core import freeze, unfreeze
from flax import optim, struct
from flax import linen as nn


class Mean(nn.Module):

    def __call__(self, X):
        m = self.m(X)
        return m.reshape(-1, 1) if self.flat else m


class MeanZero(Mean):
    output_dim: int = 1
    flat: bool = True

    def m(self, X):
        return np.zeros(X.shape[:-1] + (self.output_dim,))


class MeanConstant(Mean):
    output_dim: int = 1
    flat: bool = True
    init_val_m: float = 0.

    def setup(self):
        self.c = self.param('c', lambda k, s: np.full(s, self.init_val_m),
                            (self.output_dim,))

    def m(self, X):
        c = self.c.reshape(1, -1)
        m = np.tile(c, X.shape[:-1]+(1,))
        return m


def compose_kernel(k, l, op_reduce):

    class KernelComposition(nn.Module):
        op_reduce: Any

        def setup(self):
            self.ks = [k, l]

        def __call__(self, X, Y=None, full_cov=True):
            Ks = [k.__call__(X, Y, full_cov=full_cov) for k in self.ks]
            return op_reduce(*Ks)

    return KernelComposition(op_reduce)


def slice_to_array(s):
    """ Converts a slice `s` to np.ndarray 
        Returns (out, success) 
    """
    if s is None:
        return s, False
    elif isinstance(s, slice):
        if s.stop is not None:
            def ifnone(a, b): return b if a is None else a
            return np.array(list(range(
                ifnone(s.start, 0), s.stop, ifnone(s.step, 1)))), True
        else:
            return s, False
    elif isinstance(s, np.ndarray):
        return s, True
    elif isinstance(s, Sequence):
        return np.asarray(s), True
    else:
        raise ValueError('s must be Tuple[None, slice, list, np.ndarray]')


def kernel_active_dims_overlap(k1, k2):
    """Check if `k1,k2` active_dims overlap 
        Returns (overlaps, overlapped index)
    """
    a1, b1 = slice_to_array(k1.active_dims)
    a2, b2 = slice_to_array(k2.active_dims)
    if all([b1, b2]):
        o = np.intersect1d(a1, a2)
        return (len(o) > 0), o
    else:
        return True, None



def sqdist(X, Y=None):
    """ Returns D where D_ij = ||X_i - Y_j||^2 if Y is not `None`
            https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
            https://github.com/GPflow/GPflow/gpflow/utilities/ops.py#L84
        For Y!=None, similar speed to `jit(cdist_sqeuclidean)`
            jitting does not improve performance
        For Y==None, 10x faster than jitted `jit(cdist_sqeuclidean)`
        X   (n, d)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y is not None and Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if Y is None:
        D = X@X.T
        Xsqnorm = np.diag(D).reshape(-1, 1)
        D = - 2*D + Xsqnorm + Xsqnorm.T
        D = np.maximum(D, 0)
    else:
        Xsqnorm = np.sum(np.square(X), axis=-1).reshape(-1, 1)
        Ysqnorm = np.sum(np.square(Y), axis=-1).reshape(-1, 1)
        D = -2*np.tensordot(X, Y, axes=(-1, -1))
        D += Xsqnorm + Ysqnorm.T
        D = np.maximum(D, 0)
    return D


class Kernel(nn.Module):

    active_dims: Union[slice, list, np.ndarray] = None

    def slice(self, X):
        if self.active_dims is None:
            return X
        else:
            return X[..., self.active_dims] if X is not None else X

    def K(self, X, Y=None):
        raise NotImplementedError

    def Kdiag(self, X, Y=None):
        raise NotImplementedError

    def apply_mapping(self, X):
        if X is not None and hasattr(self, 'g'):
            return self.g(X)
        else:
            return X

    def __call__(self, X, Y=None, full_cov=True):
        X = self.slice(X)
        Y = self.slice(Y)
        X = self.apply_mapping(X)
        Y = self.apply_mapping(Y)
        if full_cov:
            return self.K(X, Y)
        else:
            if Y is not None:
                raise ValueError('full_cov=True & Y=None not compatible')
            return self.Kdiag(X, Y)

    def __add__(self, other):
        return compose_kernel(self, other, np.add)

    def __mul__(self, other):
        return compose_kernel(self, other, np.multiply)

    def check_ard_dims(self, ℓ):
        """Verify that ard ℓ size matches with `active_dims` 
            when `active_dims` is specified, `ard_len>1` """
        if ℓ.size > 1 and self.active_dims is not None:
            s, cvt = slice_to_array(self.active_dims)
            if cvt and s.size != ℓ.size:
                raise ValueError(
                    f'ardℓ {ℓ} does not match with active_dims={s}')


class CovSE(Kernel):
    # #ard lengthscales
    ard_len: int = 1
    # initialization
    init_val_l: float = 1.

    def setup(self):
        self.ℓ = BijSoftplus.forward(self.param(
            'ℓ', lambda k, s: BijSoftplus.reverse(
                self.init_val_l*np.ones(s, dtype=np.float32)), (self.ard_len,)))
        self.σ2 = BijSoftplus.forward(self.param(
            'σ2', lambda k, s: BijSoftplus.reverse(np.array([1.])), (1,)))
        self.check_ard_dims(self.ℓ)

    def scale(self, X):
        return X/self.ℓ if X is not None else X

    def K(self, X, Y=None):
        X = self.scale(X)
        Y = self.scale(Y)
        return self.σ2*np.exp(-sqdist(X, Y)/2)

    def Kdiag(self, X, Y=None):
        return np.tile(self.σ2, len(X))


class CovSEwithEncoder(Kernel):
    # #ard lengthscales
    ard_len: int = 1
    # initialization
    init_val_l: float = 1.
    # encoder
    g_feature_dims: int = 1

    def setup(self):
        self.g = nn.Dense(self.g_feature_dims)
        self.ℓ = BijSoftplus.forward(self.param(
            'ℓ', lambda k, s: BijSoftplus.reverse(
                self.init_val_l*np.ones(s, dtype=np.float32)), (self.ard_len,)))
        self.σ2 = BijSoftplus.forward(self.param(
            'σ2', lambda k, s: BijSoftplus.reverse(np.array([1.])), (1,)))
        self.check_ard_dims(self.ℓ)

    def scale(self, X):
        return X/self.ℓ if X is not None else X

    def K(self, X, Y=None):
        X = self.scale(X)
        Y = self.scale(Y)
        return self.σ2*np.exp(-sqdist(X, Y)/2)

    def Kdiag(self, X, Y=None):
        return np.tile(self.σ2, len(X))


class CovIndex(Kernel):
    """A kernel applied to indices over a lookup table B
            K[i,j] = B[i,j]
                where B = WWᵀ + diag[v]
    """
    # #rows of W
    output_dim: int = 1
    # #columns of W
    rank: int = 1
    # initialization
    init_val_W: float = 1.

    def setup(self):
        self.W = self.param('W', lambda k, s: self.init_val_W*random.normal(k, s),
                            (self.output_dim, self.rank))
        self.v = BijSoftplus.forward(
            self.param('v', lambda k, s: BijSoftplus.reverse(np.ones(s)),
                       (self.output_dim,)))

    def cov(self):
        return self.W@self.W.T + np.diag(self.v)

    def K(self, X, Y=None):
        Y = X if Y is None else Y
        X = np.asarray(X, np.int32).squeeze()
        Y = np.asarray(Y, np.int32).squeeze()
        B = self.cov()
        K = np.take(np.take(B, Y, axis=1), X, axis=0)
        return K

    def Kdiag(self, X, Y=None):
        Bdiag = np.sum(np.square(self.W), axis=1)+self.v
        X = np.asarray(X, np.int32).squeeze()
        return np.take(Bdiag, X)


class CovIndexSpherical(Kernel):
    """A kernela pplied to indices over a lookup table B
            K[i,j] = B[i,j]
                where   B = s*sᵀ + diag[l]
                        s = [1 cosθ0 cosθ1      cosθ3          ]
                            [0 sinθ0 sinθ1cosθ2 sinθ3cosθ4     ]
                            [0 0     sinθ1sinθ2 sinθ3sinθ4cosθ5]
                            [0 0     0          sinθ3sinθ4sinθ5]
                        output_dim = 4
        Note:
            Not psd, hard to optimize ...
    """
    output_dim: int = 1

    def n_θ(self, n):
        return int(n*(n-1)/2)

    def setup(self):
        if self.output_dim not in [1, 2, 3, 4]:
            raise ValueError(
                f'CovIndexSpherical not implemented for output_dim={self.output_dim}')

        self.θ = self.param('θ', lambda k, s: random.uniform(k, s, np.float32, 0, np.pi),
                            (self.n_θ(self.output_dim),))
        self.l = BijSoftplus.forward(
            self.param('l', lambda k, s: BijSoftplus.reverse(np.ones(s)),
                       (self.output_dim,)))

    def cov(self):
        d = self.output_dim
        θ = self.θ
        l = self.l

        if d == 1:
            s = np.array([[1.]])
        if d == 2:
            s = np.array([[1, np.cos(θ[0])],
                          [0, np.sin(θ[0])]])
        if d == 3:
            s = np.array([[1, np.cos(θ[0]), np.cos(θ[1])],
                          [0, np.sin(θ[0]), np.sin(θ[1])*np.cos(θ[2])],
                          [0, 0, np.sin(θ[1])*np.sin(θ[2])]])
        if d == 4:
            s = np.array([[1, np.cos(θ[0]), np.cos(θ[1]), np.cos(θ[3])],
                          [0, np.sin(θ[0]), np.sin(θ[1])*np.cos(θ[2]),
                           np.sin(θ[3])*np.cos(θ[4])],
                          [0, 0, np.sin(θ[1])*np.sin(θ[2]),
                           np.sin(θ[3])*np.sin(θ[4])*np.cos(θ[5])],
                          [0, 0, 0, np.sin(θ[3])*np.sin(θ[4])*np.sin(θ[5])]])
        A = s.T@s
        A = A + np.diag(l)
        return A

    def K(self, X, Y=None):
        Y = X if Y is None else Y
        X = np.asarray(X, np.int32).squeeze()
        Y = np.asarray(Y, np.int32).squeeze()
        B = self.cov()
        K = np.take(np.take(B, Y, axis=1), X, axis=0)
        return K

    def Kdiag(self, X, Y=None):
        Bdiag = self.l
        X = np.asarray(X, np.int32).squeeze()
        return np.take(Bdiag, X)


class CovICM(Kernel):
    """Intrinsic Coregionalization Kernel
            k(X,Y) = kx(X,Y) ⊗ B
                - fully observed multiple-outputs

    https://homepages.inf.ed.ac.uk/ckiw/postscript/multitaskGP_v22.pdf
    """
    # data kernel
    kx_cls: Kernel.__class__ = CovSE
    kx_kwargs: dict = field(default_factory=dict)
    # task kernel
    kt_cls: Kernel.__class__ = CovIndex
    kt_kwargs: dict = field(default_factory=dict)

    def setup(self):
        self.kx = self.kx_cls(**self.kx_kwargs)
        self.kt = self.kt_cls(**self.kt_kwargs)

    def K(self, X, Y=None):
        Kx = self.kx(X, Y, full_cov=True)
        return np.kron(Kx, self.kt.cov())

    def Kdiag(self, X, Y=None):
        Kx = self.kx(X, Y, full_cov=False)
        return np.kron(Kx, np.diag(self.kt.cov()))


class CovICMLearnable(Kernel):
    """ICM kernel where kt is formed from additional kernels

        mode: diag
            k(X,Y)= (kˣ(X,Y)⊗I) ◦ (Σᵢ kᵗⁱ(X,Y)⊗diag[eᵢ])
    """
    # diag, all
    mode: str = 'diag'
    # number of outputs/tasks
    output_dim: int = 1
    # data kernel
    kx_cls: Kernel.__class__ = CovSE
    kx_kwargs: dict = field(default_factory=dict)
    # task kernel
    kt_cls: Kernel.__class__ = CovSE
    kt_kwargs: dict = field(default_factory=dict)

    def setup(self):
        if self.mode not in ['diag', 'all']:
            raise ValueError(
                f'`mode` {self.mode} not allowed')
        self.kx = self.kx_cls(**self.kx_kwargs)
        self.kt = [self.kt_cls(**self.kt_kwargs)
                   for _ in range(self.n_kt())]

    def n_kt(self):
        if self.mode == 'diag':
            n_kt = self.output_dim
        if self.mode == 'all':
            n_kt = self.output_dim*self.output_dim
        return n_kt

    def K(self, X, Y=None):

        m = self.output_dim
        nx = len(X)
        ny = len(Y) if Y is not None else nx
        Kx = np.kron(self.kx(X, Y, full_cov=True),
                     np.ones((m, m)))
        for i in range(m):
            for j in range(m):
                ind = np.meshgrid(np.arange(m*nx, step=m),
                                  np.arange(m*ny, step=m))
                ind = tuple(x.T for x in ind)
                ind = (ind[0]+i, ind[1]+j)
                if self.mode == 'diag':
                    if i == j:
                        ktⁱ = self.kt[i](X, Y, full_cov=True)
                        M = Kx[ind]*ktⁱ
                    else:
                        M = 0
                if self.mode == 'all':
                    kti = i*m+j
                    ktⁱ = self.kt[kti](X, Y, full_cov=True)
                    M = Kx[ind]*ktⁱ
                Kx = jax.ops.index_update(Kx, ind, M)
        return Kx

    def Kdiag(self, X, Y=None):
        m = self.output_dim
        Kx = np.kron(self.kx(X, Y, full_cov=False),
                     np.ones((m,)))
        if self.mode == 'diag':
            Kt = [k(X, Y, full_cov=False) for k in self.kt]
        if self.mode == 'all':
            ind = [i*m+i for i in range(m)]
            Kt = [self.kt[kti](X, Y, full_cov=False) for kti in ind]
        Kt = np.vstack(Kt).T.flatten()
        K = Kx*Kt
        return K


class Lik(nn.Module):
    """ p(y|f) """

    def predictive_dist(μ, Σ, full_cov=True):
        """Computes predictive distribution for `y`
                E[y] = ∫∫ y  p(y|f)q(f) dfdy 
                V[y] = ∫∫ y² p(y|f)q(f) dfdy

            where q(f) = N(f; μ, Σ)

            `full_cov` if True implies Σ a vector
        """
        raise NotImplementedError

    def variational_log_prob(self, y, μf, σ2f):
        """Computes variational expectation of log density 
                E[log(p(y|f))] = ∫ log(p(y|f)) q(f) df

            where q(f) = N(f; μf, diag[σ2f])
        """
        raise NotImplementedError


class LikNormal(Lik):

    def setup(self):
        def init_fn(k, s): return BijSoftplus.reverse(np.repeat(1., 1))
        self.σ2 = BijSoftplus.forward(self.param('σ2', init_fn, (1,)))

    def predictive_dist(self, μ, Σ, full_cov=True):
        """ y ~ N(μ, K+σ²*I)
                where f~N(μ, Σ), y|f ~N(0,σ²I)
        """
        if full_cov:
            assert(Σ.shape[-1] == Σ.shape[-2])
            Σ = jax_add_to_diagonal(Σ, self.σ2)
        else:
            Σ = Σ.reshape(-1, 1)
            Σ = Σ + self.σ2
        return μ, Σ

    def variational_log_prob(self, y, μf, σ2f):
        """Computes E[log(p(y|f))] 
                where f ~ N(μ, diag[v]) and y = \prod_i p(yi|fi)

        E[log(p(y|f))] = Σᵢ E[ -.5log(2πσ²) - (.5/σ²) (yᵢ^2 - 2yᵢfᵢ + fᵢ^2) ]
                       = Σᵢ -.5log(2πσ²) - (.5/σ²) ((yᵢ-μᵢ)^2 + vᵢ)   by E[fᵢ]^2 = μᵢ^2 + vᵢ
        """
        # for multiple-output
        μf = μf.reshape(-1, 1)
        σ2f = σ2f.reshape(-1, 1)
        y = y.reshape(-1, 1)
        return np.sum(-.5*np.log(2*np.pi*self.σ2) -
                      (.5/self.σ2)*(np.square((y-μf)) + σ2f))


class LikMultipleNormal(Lik):
    output_dim: int = 1

    def setup(self):
        def init_fn(k, s): return BijSoftplus.reverse(
            np.repeat(1., self.output_dim))
        self.σ2 = BijSoftplus.forward(
            self.param('σ2', init_fn, (self.output_dim,)))

    def σ2s(self, ind):
        ind = np.asarray(ind, np.int32)
        return self.σ2[ind]

    def predictive_dist(self, μ, Σ, ind, full_cov=True):
        σ2s = self.σ2s(ind)
        if full_cov:
            assert(Σ.shape[-1] == Σ.shape[-2])
            Σ = jax_add_to_diagonal(Σ, σ2s)
        else:
            assert(Σ.size == σ2s.size)
            Σ = Σ + σ2s
        return μ, Σ

    def variational_log_prob(self, y, μf, σ2f, ind):
        y = y.reshape(-1, 1)
        ind = ind.reshape(y.shape)
        μf, σ2f = μf.reshape(y.shape), σ2f.reshape(y.shape)
        σ2s = self.σ2s(ind)
        return np.sum(-.5*np.log(2*np.pi*σ2s) -
                      (.5/σ2s)*(np.square((y-μf)) + σ2f))


class LikMultipleNormalKron(Lik):
    output_dim: int = 1

    def setup(self):
        def init_fn(k, s): return BijSoftplus.reverse(
            np.repeat(1., self.output_dim))
        self.σ2 = BijSoftplus.forward(
            self.param('σ2', init_fn, (self.output_dim,)))

    def σ2I(self, σ2I_size):
        n = σ2I_size/self.output_dim
        if not n.is_integer():
            raise ValueError(
                f'LikMultipleNormalKron.output_dim={self.output_dim}'
                f' not compatible with #data={n}')
        σ2I = np.kron(self.σ2, np.ones(int(n),))
        return σ2I

    def predictive_dist(self, μ, Σ, full_cov=True):
        """Computes μ, Σ -> μ, Σ+(D ⊗ I) where D = diag[σ2]"""
        if full_cov:
            assert(Σ.shape[-1] == Σ.shape[-2])
            σ2I = self.σ2I(Σ.shape[-1])
            Σ = jax_add_to_diagonal(Σ, σ2I)
        else:
            σ2I = self.σ2I(Σ.size)
            Σ = Σ + σ2I.reshape(Σ.shape)
        return μ, Σ

    def variational_log_prob(self, y, μf, σ2f):
        y = y.reshape(-1, 1)
        μf, σ2f = μf.reshape(y.shape), σ2f.reshape(y.shape)
        σ2I = self.σ2I(y.size).reshape(y.shape)
        return np.sum(-.5*np.log(2*np.pi*σ2I) -
                      (.5/σ2I)*(np.square((y-μf)) + σ2f))


class GPModel(object):

    def pred_y(self, Xs, full_cov=False):
        """ Assumes `self.lik` and implments `self.pred_f()` """
        μf, σ2f = self.pred_f(Xs, full_cov=full_cov)
        if isinstance(self.lik, LikMultipleNormal):
            μy, σ2y = self.lik.predictive_dist(
                μf, σ2f, Xs[:, -1], full_cov=full_cov)
        else:
            μy, σ2y = self.lik.predictive_dist(μf, σ2f, full_cov=full_cov)
        return μy, σ2y


class GPR(nn.Module, GPModel):
    data: Tuple[np.ndarray, np.ndarray]

    def setup(self):
        self.mean_fn = MeanZero()
        self.k = CovSE()
        self.lik = LikNormal()

    def get_init_params(self, key):
        Xs = np.zeros((1, self.data[0].shape[-1]))
        params = self.init(key, method=self.mll)
        return params

    def pred_cov(self, K, ind):
        """Computes K+σ2I"""
        if isinstance(self.lik, LikMultipleNormal):
            _, K = self.lik.predictive_dist(None, K, ind)
        else:
            _, K = self.lik.predictive_dist(None, K)
        return K

    def mll(self):
        X, y = self.data
        k = self.k
        n = len(X)

        K = self.pred_cov(k(X), X[:, -1])
        L = linalg.cholesky(K)

        m = self.mean_fn(X)
        mlik = MultivariateNormalTril(m, L)
        mll = mlik.log_prob(y)

        return mll

    def pred_f(self, Xs, full_cov=True):
        X, y = self.data
        k = self.k
        ns, output_dim = len(Xs), int(y.size/len(X))

        ms = self.mean_fn(Xs)
        mf = self.mean_fn(X)

        Kff = self.pred_cov(k(X), X[:, -1])
        Kfs = k(X, Xs)
        Kss = k(Xs, full_cov=full_cov)
        L = linalg.cholesky(Kff)

        μ, Σ = mvn_conditional_exact(Kss, Kfs, ms,
                                     L, mf, y, full_cov=full_cov)
        # μ = μ.reshape(ns, output_dim)
        return μ, Σ


class GPRFITC(nn.Module, GPModel):
    data: Tuple[np.ndarray, np.ndarray]
    n_inducing: int

    def setup(self):
        self.k = CovSE()
        self.lik = LikNormal()
        X, y = self.data
        self.Xu = self.param('Xu', lambda k, s: X[:self.n_inducing],
                             (self.n_inducing, X.shape[-1]))

    def get_init_params(self, key):
        Xs = np.ones((1, self.data[0].shape[-1]))
        params = self.init(key, Xs, method=self.pred_f)
        return params

    def precompute(self):
        X, y = self.data
        k = self.k
        Xu = self.Xu
        n, m = len(X), self.n_inducing

        Kdiag = k(X, full_cov=False)
        Kuu = k(Xu)
        Kuf = k(Xu, X)
        Luu = cholesky_jitter(Kuu, jitter=1e-5)

        V = solve_triangular(Luu, Kuf, lower=True)
        Qffdiag = np.sum(np.square(V), axis=0)
        Λ = Kdiag - Qffdiag + self.lik.σ2
        Λ = Λ.reshape(-1, 1)

        return Luu, V, Λ

    def mll(self):
        X, y = self.data
        n = len(X)
        Luu, V, Λ = self.precompute()

        mlik = MultivariateNormalInducing(np.zeros(n), V, Λ)
        mll = mlik.log_prob(y)

        return mll

    def pred_f(self, Xs, full_cov=True):
        X, y = self.data
        k = self.k
        Xu = self.Xu
        Luu, V, Λ = self.precompute()

        Kss = k(Xs, full_cov=full_cov)
        Kus = k(Xu, Xs)

        μ, Σ = mvn_conditional_sparse(Kss, Kus,
                                      Luu, V, Λ, y, full_cov=full_cov)
        return μ, Σ


class VFE(nn.Module, GPModel):
    data: Tuple[np.ndarray, np.ndarray]
    n_inducing: int

    def setup(self):
        self.k = CovSE()
        self.lik = LikNormal()
        X, y = self.data
        self.Xu = self.param('Xu', lambda k, s: X[:self.n_inducing],
                             (self.n_inducing, X.shape[-1]))

    def get_init_params(self, key):
        params = self.init(key, np.ones((1, self.data[0].shape[-1])),
                           method=self.pred_f)
        return params

    def precompute(self):
        X, y = self.data
        k = self.k
        Xu = self.Xu
        n, m = len(X), self.n_inducing

        Kdiag = k(X, full_cov=False)
        Kuu = k(Xu)
        Kuf = k(Xu, X)
        Luu = cholesky_jitter(Kuu, jitter=1e-5)

        V = solve_triangular(Luu, Kuf, lower=True)
        Λ = self.lik.σ2*np.ones(n)
        Λ = Λ.reshape(-1, 1)

        return Kdiag, Luu, V, Λ

    def mll(self):
        X, y = self.data
        n = len(X)

        Kdiag, Luu, V, Λ = self.precompute()

        mlik = MultivariateNormalInducing(np.zeros(n), V, Λ)
        elbo_mll = mlik.log_prob(y)
        elbo_trace = -(1/2/self.lik.σ2[0]) * \
            (np.sum(Kdiag) - np.sum(np.square(V)))
        elbo = elbo_mll + elbo_trace

        return elbo

    def pred_f(self, Xs, full_cov=True):
        X, y = self.data
        k = self.k
        Xu = self.Xu
        _, Luu, V, Λ = self.precompute()

        Kss = k(Xs, full_cov=full_cov)
        Kus = k(Xu, Xs)
        μ, Σ = mvn_conditional_sparse(Kss, Kus,
                                      Luu, V, Λ, y, full_cov=full_cov)

        return μ, Σ


class SVGP(nn.Module, GPModel):
    n_data: int
    Xu_initial: np.ndarray

    def setup(self):
        self.d = self.Xu_initial.shape[1]
        self.n_inducing = self.Xu_initial.shape[0]
        self.mean_fn = MeanZero()
        self.k = CovSE()
        self.lik = LikNormal()
        self.Xu = self.param('Xu', lambda k, s: self.Xu_initial,
                             (self.n_inducing, self.d))
        self.q = VariationalMultivariateNormal(self.n_inducing)

    def get_init_params(self, key, n_tasks=1):
        n = 1
        Xs = np.ones((n, self.Xu_initial.shape[-1]))
        ys = np.ones((n, n_tasks))
        params = self.init(key, (Xs, ys), method=self.mll)
        return params

    def mll(self, data):
        X, y = data
        k = self.k
        Xu, μq, Lq = self.Xu, self.q.μ, self.q.L

        mf = self.mean_fn(X)
        mu = self.mean_fn(Xu)

        Kff = k(X, full_cov=False)
        Kuf = k(Xu, X)
        Kuu = k(Xu)
        Luu = cholesky_jitter(Kuu, jitter=5e-5)

        α = self.n_data/len(X) \
            if self.n_data is not None else 1.

        μqf, σ2qf = mvn_marginal_variational(Kff, Kuf, mf,
                                             Luu, mu, μq, Lq, full_cov=False)
        if isinstance(self.lik, LikMultipleNormal):
            elbo_lik = α*self.lik.variational_log_prob(y, μqf, σ2qf, X[:, -1])
        else:
            elbo_lik = α*self.lik.variational_log_prob(y, μqf, σ2qf)
        elbo_nkl = -kl_mvn_tril(μq, Lq, mu, Luu)
        elbo = elbo_lik + elbo_nkl

        return elbo

    def pred_f(self, Xs, full_cov=True):
        k = self.k
        Xu, μq, Lq = self.Xu, self.q.μ, self.q.L

        ms = self.mean_fn(Xs)
        mu = self.mean_fn(Xu)

        Kss = k(Xs, full_cov=full_cov)
        Kus = k(Xu, Xs)
        Kuu = k(Xu)
        Luu = cholesky_jitter(Kuu, jitter=5e-5)

        μf, Σf = mvn_marginal_variational(Kss, Kus, ms,
                                          Luu, mu, μq, Lq, full_cov=full_cov)
        return μf, Σf


class MultivariateNormalTril(object):
    """N(μ, LLᵀ) """

    def __init__(self, μ, L):
        if μ.ndim != 1:
            μ = μ.flatten()
        d = μ.size
        if L.shape != (d, d):
            raise ValueError(
                f'L should have dim ({d},{d})')
        self.d = d
        self.μ = μ
        self.L = L

    def log_prob(self, x):
        """x: (n, m) -> (n*m,)"""
        x = x.reshape(self.μ.shape)
        α = solve_triangular(self.L, (x-self.μ), lower=True)
        mahan = -.5*np.sum(np.square(α))
        lgdet = -np.sum(np.log(np.diag(self.L)))
        const = -.5*self.d*np.log(2*np.pi)
        return mahan + const + lgdet

    def cov(self):
        return self.L@self.L.T

    def sample(self, key, shape=()):
        """Outputs μ+Lϵ where ϵ~N(0,I)"""
        shape = shape + self.μ.shape
        ϵ = random.normal(key, shape)
        return self.μ.T + np.tensordot(ϵ, self.L, [-1, 1])


class MultivariateNormalInducing(object):
    """N(μ, VᵀV + Λ) where V low rank, Λ diagonal

        Used to represent p(f|X) for sparse GPs
                - Q = VᵀV where V = inv(L)@Kuf, Q=LLᵀ
                - Λ_dic  = diag[σ2*I]
                - Λ_fitc = diag[K-Q+σ2*I]
    """

    def __init__(self, μ, V, Λ):
        self.μ = μ.reshape(-1, 1)
        self.V = V
        self.Λ = Λ.reshape(-1, 1)

    def log_prob(self, x):
        μ, Λ, V = self.μ, self.Λ, self.V
        d = μ.size
        x = x.reshape(μ.shape)
        e = x - μ

        B = np.eye(V.shape[0]) + (V/Λ.T)@V.T
        LB = cholesky_jitter(B, jitter=1e-5)
        γ = solve_triangular(LB, V@(e/Λ), lower=True)

        mahan = -.5*(np.sum((e/Λ)*e) - np.sum(np.square(γ)))
        lgdet = -np.sum(np.log(np.diag(LB)))-.5*np.sum(np.log(Λ))
        const = -(d/2)*np.log(2*np.pi)
        return mahan + const + lgdet

    def cov(self):
        return self.V.T@self.V + self.Λ


class VariationalMultivariateNormal(nn.Module):
    d: int = 1

    def setup(self):
        self.μ = self.param('μ', jax.nn.initializers.zeros, (self.d,))
        self.L = BijFillTril.forward(
            self.param('L', lambda k, s: BijFillTril.reverse(np.eye(self.d)),
                       (BijFillTril.reverse_shape(self.d),)))

    def __call__(self):
        return MultivariateNormalTril(self.μ, self.L)


class BijExp(object):

    @staticmethod
    def forward(x):
        """ x -> exp(x) \in \R+ """
        return np.exp(x)

    @staticmethod
    def reverse(y):
        return np.log(y)


class BijSoftplus(object):
    """
    Reference
    - https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Softplus
    - http://num.pyro.ai/en/stable/_modules/numpyro/distributions/transforms.html
    """
    @staticmethod
    def forward(x):
        return jax.nn.softplus(x)

    @staticmethod
    def reverse(y):
        return softplus_inv(y)


class BijFillTril(object):
    """Transofrms vector to lower triangular matrix
            v (n,) -> L (m,m)
                where `m = (-1+sqrt(1+8*n))/2`
                      `n = m*(m+1)/2`.`
    Reference
    - https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/FillTriangular
    - https://www.tensorflow.org/probability/api_docs/python/tfp/math/fill_triangular
    """
    @staticmethod
    def forward_shape(n):
        return int((-1+math.sqrt(1+8*n))/2)

    @staticmethod
    def reverse_shape(m):
        return int(m*(m+1)/2)

    @staticmethod
    def forward(v):
        m = BijFillTril.forward_shape(v.size)
        L = np.zeros((m, m))
        L = jax.ops.index_update(L, np.tril_indices(m), v.squeeze())
        return L

    @staticmethod
    def reverse(L):
        m = len(L)
        v = L[np.tril_indices(m)]
        v = v.squeeze()
        return v


class BijSoftplusFillTril(object):

    @staticmethod
    def forward_shape(n):
        return int((-1+math.sqrt(1+8*n))/2)

    @staticmethod
    def reverse_shape(m):
        return int(m*(m+1)/2)

    @staticmethod
    def forward(v):
        m = BijSoftplusFillTril.forward_shape(v.size)
        L = np.zeros((m, m))
        L = jax.ops.index_update(L, np.tril_indices(m, k=-1), v[:-m].squeeze())
        L = jax.ops.index_update(L, np.diag_indices(
            m), jax.nn.softplus(v[-m:].squeeze()))
        return L

    @staticmethod
    def reverse(L):
        m = len(L)
        v1 = L[np.tril_indices(m, k=-1)]
        v2 = softplus_inv(L[np.diag_indices(m)])
        v = np.concatenate((v1, v2), axis=-1)
        v = v.reshape(-1, 1)
        return v


def softplus_inv(y):
    """ y > log(exp(y)-1)
            log(1-exp(-y))+log(exp(y))
            log(1-exp(-y))+y
    """
    return np.log(-np.expm1(-y)) + y


def diag_indices_kth(n, k):
    rows, cols = np.diag_indices(n)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def mvn_conditional_exact(Kss, Ks, ms,
                          L, m, y, full_cov=False):
    """Computes p(fs|f) ~ N( ms  + Kfs*inv(K)*(y-m),
                             Kss - Ksᵀ*inv(K)*Ks )
        where 
                p(f,fs) ~ N([m ], [K    Ks ]
                            [ms], [Ksᵀ  Kss])            
                K=L@Lᵀ,
                Ks = k(X,Xs)

        when `full_cov=True`
            assume `K` is also the diagonal
    """
    if Kss.shape[0] != ms.size:
        raise ValueError(
            f'k(Xs), mean(Xs) does not agree in size '
            f'{Kss} vs {ms.shape}')
    if y.size != m.size:
        raise ValueError(
            f'y, mean(X)  does not agree in size '
            f'{y.shape} vs {m.shape}')
    # for multiple-output GP
    y = y.reshape(-1, 1)
    m = m.reshape(-1, 1)
    ms = ms.reshape(-1, 1)

    α = cho_solve((L, True), (y-m))
    μ = Ks.T@α + ms
    v = solve_triangular(L, Ks, lower=True)

    if full_cov:
        Σ = Kss - v.T@v
    else:
        Σ = Kss - np.sum(np.square(v), axis=0)

    return μ, Σ


def mvn_marginal_variational(Kff, Kuf, mf,
                             Luu, mu, μq, Lq, full_cov=False):
    """q(f) = \int p(f|u)q(u) du
            = N(mf  + Kfu Kuu^-1 (μq - mu),
                Kff - Qff + Kfu Kuu^-1 Σq Kuu^-1 Kuf)

        where   q(u)   ~ N(μq, Σq) w/  Σq := Lq@Lq.T
                p(u)   ~ N(0, Kuu) w/ Kuu := Luu@Luu.T
                p(f|u) ~ N(0, Kfu Kuu^-1 u, Kff - Qff)

                mf = mean_fn(X)
                mu = mean_fn(Xu)

        when `full_cov=True`
            assume `Kff` is also the diagonal
    """
    α = solve_triangular(Luu, Kuf, lower=True)
    β = solve_triangular(Luu.T, α, lower=False)
    γ = Lq.T@β
    if full_cov:
        Σf = Kff - α.T@α + γ.T@γ
    else:
        Σf = Kff - \
            np.sum(np.square(α), axis=0) + \
            np.sum(np.square(γ), axis=0)
    # for multiple-output
    μq = μq.reshape(-1, 1)
    mu = mu.reshape(-1, 1)
    mf = mf.reshape(-1, 1)
    μf = mf + β.T@(μq-mu)

    return μf, Σf


def mvn_conditional_sparse(Kss, Kus,
                           Luu, V, Λ, y, full_cov=False):
    """Computes q(fs|y) ~ N( Qsf*(Qff+Λ)^(-1)*y, 
                             Kss - Qsf(Qff+Λ)^(-1)*Qfs )
            where,
                q(f,fs) ~ N([0], [Qff + Λ, Qfs]
                            [0], [Qsf,     Kss])

        Qff = VᵀV
        Kuu = Luu*Luuᵀ

        when `full_cov=True`
            assume `Kss` is also the diagonal
    """
    Λ = Λ.reshape(-1, 1)

    B = np.eye(V.shape[0]) + (V/Λ.T)@V.T
    LB = cholesky_jitter(B, jitter=1e-5)
    γ = solve_triangular(LB, V@(y/Λ), lower=True)

    ω = solve_triangular(Luu, Kus, lower=True)
    ν = solve_triangular(LB, ω, lower=True)

    if full_cov:
        Σ = Kss - ω.T@ω + ν.T@ν
    else:
        Σ = Kss - \
            np.sum(np.square(ω), axis=0) + \
            np.sum(np.square(ν), axis=0)

    μ = ω.T@solve_triangular(LB.T, γ, lower=False)
    return μ, Σ


def rand_μΣ(key, d):
    k1, k2 = random.split(key)
    μ = random.normal(k1, (d,))
    L = random.normal(k2, (d, d))
    Σ = L@L.T
    return μ, Σ


def kl_mvn(μ0, Σ0, μ1, Σ1):
    """KL(q||p) where q~N(μ0,Σ0), p~N(μ1,Σ1) """
    μ0 = μ0.reshape(-1, 1)
    μ1 = μ1.reshape(-1, 1)
    n = μ0.size
    kl_trace = np.trace(linalg.solve(Σ1, Σ0))
    kl_mahan = np.sum((μ1-μ0).T@linalg.solve(Σ1, (μ1-μ0)))
    kl_const = -n
    kl_lgdet = np.log(linalg.det(Σ1)) - np.log(linalg.det(Σ0))
    kl = .5*(kl_trace + kl_mahan + kl_const + kl_lgdet)
    return kl


def kl_mvn_tril(μ0, L0, μ1, L1):
    """KL(q||p) where q~N(μ0,L0@L0.T), p~N(μ1,L1@L1.T) """
    μ0 = μ0.reshape(-1, 1)
    μ1 = μ1.reshape(-1, 1)
    n = μ0.size
    α = solve_triangular(L1, L0, lower=True)
    β = solve_triangular(L1, μ1 - μ0, lower=True)
    kl_trace = np.sum(np.square(α))
    kl_mahan = np.sum(np.square(β))
    kl_const = -n
    kl_lgdet = np.sum(np.log(np.diag(np.square(L1)))) - \
        np.sum(np.log(np.diag(np.square(L0))))
    kl = .5*(kl_trace + kl_mahan + kl_const + kl_lgdet)
    return kl


def kl_mvn_tril_zero_mean_prior(μ0, L0, L1):
    """KL(q||p) where q~N(μ0,L0@L0.T), p~N(0,L1@L1.T) """
    μ0 = μ0.reshape(-1, 1)
    n = μ0.size
    α = solve_triangular(L1,  L0, lower=True)
    β = solve_triangular(L1, -μ0, lower=True)
    kl_trace = np.sum(np.square(α))
    kl_mahan = np.sum(np.square(β))
    kl_const = -n
    kl_lgdet = np.sum(np.log(np.diag(np.square(L1)))) - \
        np.sum(np.log(np.diag(np.square(L0))))
    kl = .5*(kl_trace + kl_mahan + kl_const + kl_lgdet)
    return kl


def cholesky_jitter(K, jitter=1e-5):
    L = linalg.cholesky(jax_add_to_diagonal(K, jitter))
    return L


def jax_add_to_diagonal(A, v):
    """ Computes A s.t. diag[A] = diag[A] + v"""
    diag_idx = np.diag_indices(A.shape[-1])
    Adiag = A[diag_idx].squeeze()
    return jax.ops.index_update(A, diag_idx, Adiag+v)


def randsub_init_fn(key, shape, dtype=np.float32, X=None):
    idx = random.choice(key, np.arange(len(X)),
                        shape=(shape[0],), replace=False)
    return X[idx]


def proc_leaf_scalar_exponentiate(k, v): return \
    (k.split('log')[1], np.exp(v[0])) if (
        k.startswith('log') and v.size == 1) else (k, v)


def proc_leaf_vector_exponentiate(k, v): return \
    (k.split('log')[1], np.exp(v)) if (
        k.startswith('log') and v.size > 1) else (k, v)


PROC_LEAF_VECTOR_LENGTH_LIMIT = 5


def proc_leaf_vector_firstn(k, v): return \
    (f'{k}[:{PROC_LEAF_VECTOR_LENGTH_LIMIT}]', v[:PROC_LEAF_VECTOR_LENGTH_LIMIT]) \
    if isinstance(v, np.ndarray) and v.size > 1 else (k, v)


def proc_leaf_vector_squeeze(k, v): return \
    (k, v.squeeze()) if isinstance(v, np.ndarray) else (k, v)


prof_leaf_fns = [proc_leaf_scalar_exponentiate,
                 proc_leaf_vector_exponentiate,
                 proc_leaf_vector_firstn,
                 proc_leaf_vector_squeeze]


def log_func_simple(i, f, params, everyn=10):
    if i % everyn == 0:
        print(f'[{i:3}]\tLoss={f(params):.3f}')


def log_func_default(i, f, params, everyn=20):
    if i % everyn == 0:
        flattened = flax.traverse_util.flatten_dict(unfreeze(params['params']))
        S = []
        for k, v in flattened.items():
            lk = k[-1]
            for proc in prof_leaf_fns:
                lk, v = proc(lk, v)
            k = list(k)
            k[-1] = lk
            k = '.'.join(k)
            S.append(f'{k}={v:.3f}' if v.size == 1 else f'{k}={v}')

        S = '\t'.join(S)
        print(f'[{i:3}]\tLoss={f(params):.3f}\t{S}')


def get_data_stream(key, bsz, X, y):
    n = len(X)
    n_complete_batches, leftover = divmod(n, bsz)
    n_batches = n_complete_batches + bool(leftover)

    def data_stream(key):
        while True:
            key, permkey = random.split(key)
            perm = random.permutation(permkey, n)
            for i in range(n_batches):
                ind = perm[i*bsz:(i+1)*bsz]
                yield (X[ind], y[ind])

    return n_batches, data_stream(key)


def filter_contains(k, v, kwd, b=True):
    return b if kwd in k.split('/') else (not b)


def pytree_path_contains_keywords(path, kwds):
    # Check if any kwds appears in `path` of pytree
    return True if any(k in path for k in kwds) else False


def pytree_mutate(tree, kvs):
    """Mutate `tree` with `kvs: {path: value}` """
    aggregate = []
    for k, v in flax.traverse_util.flatten_dict(
            unfreeze(tree)).items():
        path = '/'.join(k)
        if path in kvs:
            assert(v.size == kvs[path].size)
            k, v = k, kvs[path]
        aggregate.append((k, v))
    tree = freeze(
        flax.traverse_util.unflatten_dict(dict(aggregate)))
    return tree


def pytree_leave(tree, name):
    for k, v in flax.traverse_util.flatten_dict(
            unfreeze(tree)).items():
        path = '/'.join(k)
        if path.endswith(name):
            return v


def pytree_leaves(tree, names):
    """Access `tree` leaves using `ks: [path]` """
    leafs = [None for _ in range(len(names))]
    for k, v in flax.traverse_util.flatten_dict(
            unfreeze(tree)).items():
        path = '/'.join(k)
        for i, name in enumerate(names):
            if path.endswith(name):
                leafs[i] = v
    return leafs


def flax_get_optimizer(optimizer_name):
    optimizer_cls = getattr(optim, optimizer_name)
    return optimizer_cls


def flax_create_optimizer(params, optimizer_name, optimizer_kwargs, optimizer_focus=None):
    return flax_get_optimizer(optimizer_name)(**optimizer_kwargs).create(params, optimizer_focus)


def flax_create_multioptimizer_3focus(params, optimizer_name, optimizer_kwargs, kwds, kwds_noopt):
    """3 distjoint set of parameters/traversals
            0. optimize parameters not mentioned in `kwds` or `kwds_noopt`
                - optimize using `optimizer_kwargs[0]`
            1. optimize `kwds` using `optimizer_kwargs[1]`
            2. optimize `kwds_noopt` using `optimizer_kwargs[2]`
    """
    focus0 = optim.ModelParamTraversal(
        lambda p, v: not pytree_path_contains_keywords(p, kwds+kwds_noopt))
    focus1 = optim.ModelParamTraversal(
        lambda p, v: pytree_path_contains_keywords(p, kwds))
    focus2 = optim.ModelParamTraversal(
        lambda p, v: pytree_path_contains_keywords(p, kwds_noopt))
    opt0 = flax_get_optimizer(optimizer_name)(**optimizer_kwargs[0])
    opt1 = flax_get_optimizer(optimizer_name)(**optimizer_kwargs[1])
    opt2 = flax_get_optimizer(optimizer_name)(**optimizer_kwargs[2])
    opt_def = optim.MultiOptimizer((focus0, opt0),
                                   (focus1, opt1),
                                   (focus2, opt2))
    opt = opt_def.create(params)
    return opt


def flax_create_multioptimizer(params, optimizer_name, optimizer_kwargs, kwds):
    focus0 = optim.ModelParamTraversal(
        lambda p, v: not pytree_path_contains_keywords(p, kwds))
    focus1 = optim.ModelParamTraversal(
        lambda p, v: pytree_path_contains_keywords(p, kwds))
    opt0 = flax_get_optimizer(optimizer_name)(**optimizer_kwargs[0])
    opt1 = flax_get_optimizer(optimizer_name)(**optimizer_kwargs[0])
    opt_def = optim.MultiOptimizer((focus0, opt0),
                                   (focus1, opt1))
    opt = opt_def.create(params)
    return opt


def flax_run_optim(f, params, num_steps=10, log_func=None,
                   optimizer='GradientDescent',
                   optimizer_kwargs={'learning_rate': .002},
                   optimizer_focus=None):
    import itertools
    fg_fn = jax.value_and_grad(f)
    opt = flax_create_optimizer(params,
                                optimizer_name=optimizer,
                                optimizer_kwargs=optimizer_kwargs,
                                optimizer_focus=optimizer_focus)
    itercount = itertools.count()
    for i in range(num_steps):
        fx, grad = fg_fn(opt.target)
        opt = opt.apply_gradient(grad)
        if log_func is not None:
            log_func(i, f, opt.target)
    return opt.target


def is_symm(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def is_pd(A):
    return np.all(linalg.eigvals(A) > 0)


def is_psd(A):
    return is_symm(A) and np.all(linalg.eigvalsh(A) > 0)


def jax_to_cpu(x, i=0):
    """e.g. over `pytree`, 
            jax.tree_util.tree_map(jax_to_cpu, params)
    """
    return device_put(x, jax.devices('cpu')[i])


def jax_to_gpu(x, i=0):
    return device_put(x, jax.devices('gpu')[i])


def torch_to_array(x):
    if isinstance(x, np.ndarray):
        return x
    else:
        return np.asarray(x.to('cpu').numpy())


def preproc_data(data, T):
    """Appends index info to `X` and onehot encode `y`"""
    X, y = data
    if X is not None:
        X = torch_to_array(X)
        n = len(X)
        X = np.repeat(X, T, axis=0)
        I = np.tile(np.arange(T), n).reshape(-1, 1)
        X = np.hstack((X, I))
    if y is not None:
        y = torch_to_array(y)
        y = jax.nn.one_hot(y.reshape(-1, 1), T).reshape(-1, 1)
        assert(y.shape[1] == 1)
    if X is not None and y is not None:
        assert(X.shape[0] == y.shape[0])
    return X, y
