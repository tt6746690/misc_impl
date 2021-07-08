import math
from typing import Any, Callable, Sequence, Optional, Tuple, Union, List, Iterable
import functools
import itertools
from functools import partial
from dataclasses import dataclass, field

import jax
from jax import random, device_put, vmap, vjp, jit
import jax.numpy as np
import jax.numpy.linalg as linalg
from jax.scipy.linalg import cho_solve, solve_triangular

import flax
from flax.core import freeze, unfreeze
from flax import optim, struct
from flax import linen as nn


class Mean(nn.Module):

    def flatten(self, m):
        return m.reshape(-1, 1) if self.flat else m

    def __call__(self, X):
        """ Applyes mean_fn to `x` or first element of `x` """
        if isinstance(X, tuple):
            X = X[0]
        return self.flatten(self.m(X))


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
        self.c = self.param('c', lambda k, s: np.repeat(self.init_val_m, s[0]),
                            (self.output_dim,))

    def m(self, X):
        c = self.c.reshape(1, -1)
        m = np.tile(c, (X.shape[0], 1))
        return m


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


def apply_fn_ndarray_or_tuple(fn, x):
    """ Computes `fn(x)` if `x` is `np.ndarray` o.w. (fn(x), ...)"""
    if isinstance(x, np.ndarray):
        return fn(x)
    elif isinstance(x, tuple):
        return (fn(x[0]),) + x[1:]


class Kernel(nn.Module):

    active_dims: Union[slice, list, np.ndarray] = None

    def slice(self, X):
        if self.active_dims is None:
            return X
        else:
            return X[..., self.active_dims] if X is not None else X

    def apply_mapping(self, X):
        if X is not None and hasattr(self, 'g'):
            return self.g(X)
        else:
            return X

    def slice_and_map(self, X):
        """Allows for `X,Y` as `np.ndarray` as well as tuple of form `(np.ndarray, ...)` """
        X = apply_fn_ndarray_or_tuple(self.slice, X)
        X = apply_fn_ndarray_or_tuple(self.apply_mapping, X)
        return X

    def check_full_cov(self, Y, full_cov):
        if full_cov == False and Y is not None:
            raise ValueError('full_cov=False & Y=None not compatible')

    def __call__(self, X, Y=None, full_cov=True):
        if full_cov:
            X = self.slice_and_map(X)
            Y = self.slice_and_map(Y)
            return self.K(X, Y)
        else:
            X = self.slice_and_map(X)
            self.check_full_cov(Y, full_cov)
            return self.Kdiag(X, Y)

    def K(self, X, Y=None):
        raise NotImplementedError

    def Kdiag(self, X, Y=None):
        raise NotImplementedError

    """ By default, `Kff,Kuf,Kuu` has `f,u ~ GP(0,k)`
        Methods which override `Kuf,Kuu` require handling of 
            slicing and application of encoder `self.g`, as well as 
            logic related to `full_cov`
    """

    def Kff(self, X, Y=None, full_cov=True):
        return self.__call__(X, Y, full_cov=full_cov)

    def Kuf(self, X, Y=None, full_cov=True):
        return self.__call__(X, Y, full_cov=full_cov)

    def Kuu(self, X, Y=None, full_cov=True):
        return self.__call__(X, Y, full_cov=full_cov)

    def __add__(self, other):
        raise NotImplemented

    def __mul__(self, other):
        return NotImplemented

    def check_ard_dims(self, ls):
        """Verify that ard ls size matches with `active_dims` 
            when `active_dims` is specified, `ard_len>1` """
        if ls.size > 1 and self.active_dims is not None:
            s, cvt = slice_to_array(self.active_dims)
            if cvt and s.size != ls.size:
                raise ValueError(
                    f'ardℓ {ls} does not match with active_dims={s}')


class CovConstant(Kernel):
    """ Constant kernel k(x, y) = σ2 """
    output_scaling: bool = True

    def setup(self):
        self.σ2 = BijSoftplus.forward(self.param(
            'σ2', lambda k, s: BijSoftplus.reverse(np.array([1.])), (1,)))

    def get_σ2(self):
        return self.σ2 if self.output_scaling else np.array([1.])

    def K(self, X, Y=None):
        Xshape = X.shape
        Yshape = Y.shape if Y is not None else X.shape
        return np.full(Xshape[:1]+Yshape[:1], self.get_σ2())

    def Kdiag(self, X, Y=None):
        return np.full(X.shape[:1], self.get_σ2())


class CovLin(Kernel):
    """Linear kernel k(x,y) = σ2xy"""
    # #ard σ2
    ard_len: int = 1

    def setup(self):
        self.σ2 = BijSoftplus.forward(self.param(
            'σ2', lambda k, s: BijSoftplus.reverse(
                1.*np.ones(s, dtype=np.float32)), (self.ard_len,)))
        self.check_ard_dims(self.σ2)

    def K(self, X, Y=None):
        if Y is None:
            Y = X
        return (X*self.σ2)@Y.T

    def Kdiag(self, X, Y=None):
        return np.sum(np.square(X)*self.σ2, axis=1)


class CovSE(Kernel):
    # #ard lengthscales
    ard_len: int = 1
    # initialization
    init_val_l: float = 1.
    # no scaling
    output_scaling: bool = True

    def setup(self):
        self.ls = BijSoftplus.forward(self.param(
            'ls', lambda k, s: BijSoftplus.reverse(
                self.init_val_l*np.ones(s, dtype=np.float32)), (self.ard_len,)))
        self.σ2 = BijSoftplus.forward(self.param(
            'σ2', lambda k, s: BijSoftplus.reverse(np.array([1.])), (1,)))
        self.check_ard_dims(self.ls)

    def scale(self, X):
        return X/self.ls if X is not None else X

    def K(self, X, Y=None):
        X = self.scale(X)
        Y = self.scale(Y)
        if self.output_scaling:
            return self.σ2*np.exp(-sqdist(X, Y)/2)
        else:
            return np.exp(-sqdist(X, Y)/2)

    def Kdiag(self, X, Y=None):
        if self.output_scaling:
            return np.tile(self.σ2, len(X))
        else:
            return np.tile(np.array([1.]), len(X))


class LayerIdentity(nn.Module):

    @nn.compact
    def __call__(self, X):
        return X

    def gz(self, X):
        return X


class CovSEwithEncoder(Kernel):
    # #ard lengthscales
    ard_len: int = 1
    # initialization
    init_val_l: float = 1.
    # encoder
    g_cls: Callable = LayerIdentity

    def setup(self):
        self.g = self.g_cls()
        self.ls = BijSoftplus.forward(self.param(
            'ls', lambda k, s: BijSoftplus.reverse(
                self.init_val_l*np.ones(s, dtype=np.float32)), (self.ard_len,)))
        self.σ2 = BijSoftplus.forward(self.param(
            'σ2', lambda k, s: BijSoftplus.reverse(np.array([1.])), (1,)))
        self.check_ard_dims(self.ls)

    def scale(self, X):
        return X/self.ls if X is not None else X

    def K(self, X, Y=None):
        X = self.scale(X)
        Y = self.scale(Y)
        return self.σ2*np.exp(-sqdist(X, Y)/2)

    def Kdiag(self, X, Y=None):
        return np.tile(self.σ2, len(X))


class CovPatchBase(Kernel):
    """ Kernel over image patches
            u ~ GP(0, kᵤ) where kᵤ = kᵧ·kℓ
                kᵧ(Z,Z') is the patch kernel
                kℓ(p,p') is the patch location kernel
    """
    kp_cls: Callable = CovSE
    kl_cls: Callable = partial(CovConstant, output_scaling=False)

    def setup(self):
        if getattr(self, 'g', None) is not None:
            raise ValueError('`CovPatch.g` should not be implemented')
        self.kp = self.kp_cls()
        self.kl = self.kl_cls()
        self.XL = self.get_XL()

    @property
    def use_loc_kernel(self):
        """ If True, when `kl_cls!=CovConstant`, then inducing patches
                is now a tuple of the form (Xp, XL)
                    where XL is location information
            Operationally, if `use_loc_kernel=True`, need to supply 
                (Xp, XL) instead of Xp as inducing inputs for `Kuf,Kuu` 
        """
        return not isinstance(self.kl, CovConstant)

    def K(self, X, Y=None):
        N, M = len(X), len(Y) if Y is not None else len(X)
        Xp, XL = self.proc_image(X)  # (N, P, L)
        P, L = Xp.shape[1], Xp.shape[2]
        Xp = Xp.reshape(-1, L)  # (N*P, L)
        if Y is not None:
            Yp, YL = self.proc_image(Y)  # (M, P, L)
            Yp = Yp.reshape(-1, L)  # (M*P, L)
        else:
            Yp, YL = None, None
        Kp = self.kp(Xp, Yp, full_cov=True)  # (N*P, M*P)
        Kp = Kp.reshape(N, P, M, P)  # (N, P, M, P)
        KL = self.kl(XL, YL, full_cov=True)  # (P, P)
        KL = np.expand_dims(KL, (0, 2))  # (1, P, 1, P)
        K = Kp*KL  # (N, P, M, P)
        return K

    def Kdiag(self, X, Y=None):
        Xp, XL = self.proc_image(X)  # (N, P, L)
        Kp = vmap(self.kp, (0, None, None), 0)(Xp, None, True)  # (N, P, P)
        KL = self.kl(XL, full_cov=True)  # (P, P)
        KL = np.expand_dims(KL, (0,))  # (1, P, P)
        K = Kp*KL  # (N, P, P)
        return K

    def Kuu(self, X, Y=None, full_cov=True):
        Xp, XL = self.proc_patch(X)  # (N, L)
        Yp, YL = self.proc_patch(
            Y) if Y is not None else (None, None)  # (M, L)
        if not full_cov:
            raise ValueError('Kuu(full_cov=False) not implemented')
        Kp = self.kp(Xp, Yp)  # (N, M)
        KL = self.kl(XL, YL, full_cov=True)  # (N, M)
        K = Kp*KL
        return K

    def Kuf(self, X, Y=None, full_cov=True):
        Xp, XL = self.proc_patch(X)  # (N, L)
        Yp, YL = self.proc_image(Y)  # (M, P, L)
        P, L = Yp.shape[1], Yp.shape[2]
        Yp = Yp.reshape(-1, L)  # (M*P, L)
        Kp = self.kp(Xp, Yp)  # (N, M*P)
        N, M = len(Xp), len(Y)
        Kp = Kp.reshape((N, M, P))  # (N, M, P)
        KL = self.kl(XL, YL, full_cov=True)  # (N, P)
        KL = np.expand_dims(KL, (1,))  # (N, 1, P)
        K = Kp*KL  # (N, M, P)
        return K

    def proc_image(self, X):
        raise NotImplementedError

    def proc_patch(self, X):
        raise NotImplementedError

    def get_XL(self):
        raise NotImplementedError


class CovPatch(CovPatchBase):
    """ Patch based kernel where all overlapping patches
            are extracted from the image and `kᵤ` applied """
    image_shape: Tuple[int] = (28, 28, 1)  # (H, W, C)
    patch_shape: Tuple[int] = (3, 3)      # (h, w)

    def proc_image(self, X):
        Xp = extract_patches_2d_vmap(X.reshape((-1, *self.image_shape)),
                                     self.patch_shape)  # (N, P, h, w)
        N, P = Xp.shape[:2]
        Xp = Xp.reshape(N, P, self.L)  # (N, P, L)
        return Xp, self.XL

    def proc_patch(self, X):
        if self.use_loc_kernel:
            Xp, XL = X
        else:
            Xp, XL = X, X
        Xp = Xp.reshape(-1, self.L)   # (N, L)
        return Xp, XL

    def get_XL(self):
        scal, transl = extract_patches_2d_scal_transl(self.image_shape,
                                                      self.patch_shape)
        scal = np.repeat(scal[np.newaxis, ...], len(transl), axis=0)
        scal_transl = np.column_stack([scal, transl])
        return scal_transl

    @property
    def L(self):
        """ Patch length / input dimension to `kp` """
        return self.patch_shape[0]*self.patch_shape[1]

    @property
    def P(self):
        """ #patches """
        H, W, C = self.image_shape
        h, w = self.patch_shape
        return (H-h+1)*(W-w+1)*C


class CovPatchEncoder(CovPatchBase):
    """ Patch based kernel where patch based responses
            are computed using an encoder, e.g. bagnet """
    encoder: 'str' = 'CNNMnist'
    XL_init_fn: Callable = None

    def setup(self):
        if self.encoder not in self.available_encoders:
            raise ValueError(
                f'CovPatchEncoder.encoder should be in {list(patch_encoders.keys())}')
        if getattr(self, 'g', None) is not None:
            raise ValueError('`CovPatchEncoder.g` should not be implemented')
        self.kp = self.kp_cls()
        self.kl = self.kl_cls()

        if self.XL_init_fn is not None:
            self.XL = self.XL_init_fn()
        else:
            self.XL = self.available_encoders[self.encoder].get_XL()

    def proc_image(self, X):
        # (N, Px, Py, L)
        if X.ndim != 4:
            raise ValueError(f'Patch should have 4-dims, '
                             f'but got {X.ndim}-dims')
        N, Py, Px, L = X.shape
        P = Py*Px
        Xp = X.reshape((N, P, L))  # (N, P, L)
        if self.XL is None:
            return Xp, np.ones((P, 4))
        else:
            return Xp, self.XL

    def proc_patch(self, X):
        if (isinstance(X, np.ndarray) and X.ndim != 2) or \
                (isinstance(X, tuple) and X[0].ndim != 2):
            raise ValueError(f'Patch should have 2-dims, '
                             f'but got {X.ndim}-dims')
        if self.use_loc_kernel:
            Xp, XL = X
        else:
            Xp, XL = X, X
        return Xp, XL

    @property
    def available_encoders(self):
        return CovPatchEncoder.get_available_encoders()

    @classmethod
    def get_available_encoders(self):
        return {'CNNMnist': CNNMnistTrunk, }


class CovConvolutional(Kernel):
    """ Convolutional Kernel f~GP(0,k)
            where k(X,X') = (1/P^2) ΣpΣp' kᵧ(X[p], X[p'])·kℓ(p,p')
        The additive kernel over patches models a linear 
            function of patch response, e.g.  f(X) = (1/P) Σp u(Xp)
                for u ~ GP(0, kᵤ) where kᵤ = kᵧ·kℓ
                    kᵧ(Z,Z') is the patch kernel
                    kℓ(p,p') is the patch location kernel
        Note we apply average of patch kernel since doing so
            requires no change to the mean function 
                m(X) = E[f(X)] = (1/P) Σp m(Xp)
                    the choice of m(Xp) = m(X) works fine
        `inducing_patch`
            If True, u~GP(0, kᵧ), f~GP(0, k)
                Kuu = kᵧ(Z)
                Kuf = (1/P) Σp kᵧ(Zp, X[p])
                Treat X as patches, Y as images
            otherwise, u,f~GP(0, k)
                Kuu, Kuf fall back to brute force evaluation of k (NMP^2 kᵧ call)
                Treat both X, Y as images 
        `kl_cls`
    """
    kg_cls: Callable = CovPatch
    inducing_patch: bool = False

    def setup(self):
        self.kg = self.kg_cls()

    def K(self, X, Y=None):
        Kg = self.kg(X, Y, full_cov=True)  # (N, P, M, P)
        K = np.mean(Kg, axis=(1, 3))  # (N, M)
        return K

    def Kdiag(self, X, Y=None):
        Kg = self.kg(X, full_cov=False)  # (N, P, P)
        K = np.mean(Kg, axis=(1, 2))  # (N,)
        return K

    def Kuf(self, X, Y=None, full_cov=True):
        if not self.inducing_patch:
            return self.__call__(X, Y, full_cov=full_cov)
        if not full_cov:
            raise ValueError('Kuf(full_cov=False) not valid')
        Kg = self.kg.Kuf(X, Y)  # (N, M, P)
        K = np.mean(Kg, axis=(2,))  # (N, M)
        return K

    def Kuu(self, X, Y=None, full_cov=True):
        if not self.inducing_patch:
            return self.__call__(X, Y, full_cov=full_cov)
        if not full_cov:
            raise ValueError('Kuu(full_cov=False) not implemented')
        Kg = self.kg.Kuu(X, Y)  # (N, M)
        return Kg


def get_init_patches(key, X, M, image_shape, patch_shape, init_method='unique'):
    """ `random` might result in many duplicate initializations 
        Currently, cannot be jitted, since requires calling to numpy.unique
    """
    import numpy as onp
    k1, k2, k3 = random.split(key, 3)
    ind = random.randint(k2, (M,), 0, len(X))
    X = np.take(X, ind, axis=0).reshape((-1, *image_shape))
    jit_extract_patches_2d_vmap = jit(extract_patches_2d_vmap,
                                      static_argnums=(1,))
    patches = jit_extract_patches_2d_vmap(X, patch_shape)  # (N, P, h, w)
    patches = patches.reshape(-1, patch_shape[0]*patch_shape[1])  # (N*P, h*w)
    if init_method == 'random':
        ind = random.randint(k3, (M,), 0, len(patches))
        patches = np.take(patches, ind, axis=0)
        patches += random.normal(key, (*patches.shape,))*.001
    elif init_method == 'unique':
        patches = onp.unique(patches, axis=0)
        ind = random.randint(k3, (M,), 0, len(patches))
        patches = np.take(patches, ind, axis=0)
    return patches


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


class CovAdd1(Kernel):
    """Order 1 additive kernel
            k(x,y) = Σᵢ kᵢ(xᵢ,yᵢ)
    """
    # base kernel
    k_cls: Callable = partial(CovSE, output_scaling=False)

    def setup(self):
        s, cvt = slice_to_array(self.active_dims)
        if not cvt:
            raise ValueError(
                "CovAdditive1.active_dims must be convertible to array")
        # `active_dims` for children kernel should start from 0
        #     since `__call__` has inputs that are pre-sliced
        self.ks = [self.k_cls(active_dims=[d]) for d in range(len(s))]

    def K(self, X, Y=None):
        K = None
        for i, k in enumerate(self.ks):
            Ki = k(X, Y, full_cov=True)
            K = Ki if (K is None) else jax.ops.index_update(
                K, jax.ops.index[:, :], K+Ki)
        return K

    def Kdiag(self, X, Y=None):
        Ks = [k(X, Y, full_cov=False) for k in self.ks]
        return np.sum(np.stack(Ks), axis=0)


class CovProduct(Kernel):
    """Hadamard product of two kernel matrices
            K = K0∘K1
    """
    # k0 & k1
    k0_cls: Callable = CovSE
    k1_cls: Callable = CovSE

    def setup(self):
        self.k0 = self.k0_cls()
        self.k1 = self.k1_cls()

    def K(self, X, Y=None):
        K0 = self.k0(X, Y, full_cov=True)
        K1 = self.k1(X, Y, full_cov=True)
        return K0*K1

    def Kdiag(self, X, Y=None):
        K0diag = self.k0(X, Y, full_cov=False)
        K1diag = self.k1(X, Y, full_cov=False)
        return K0diag*K1diag


class CovICM(Kernel):
    """Intrinsic Coregionalization Kernel
            k(X,Y) = kx(X,Y) ⊗ B
                - fully observed multiple-outputs

    https://homepages.inf.ed.ac.uk/ckiw/postscript/multitaskGP_v22.pdf
    """
    # data kernel
    kx_cls: Callable = CovSE
    # task kernel
    kt_cls: Callable = CovIndex

    def setup(self):
        self.kx = self.kx_cls()
        self.kt = self.kt_cls()

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
    kx_cls: Callable = CovSE
    # task kernel
    kt_cls: Callable = CovSE

    def setup(self):
        if self.mode not in ['diag', 'all']:
            raise ValueError(
                f'`mode` {self.mode} not allowed')
        self.kx = self.kx_cls()
        self.kt = [self.kt_cls() for _ in range(self.n_kt())]

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


class CovMultipleOutputIndependent(Kernel):
    """Represents multiple output GP with a shared encoder
            f = (f1,...,fᴾ)
            fᵖ ~ GP(0, kᵖ(NeuralNetTrunk(x)))
                kᵖ are convolutional kernels
    """
    output_dim: int = 1
    k_cls: Callable = CovSE
    g_cls: Callable = LayerIdentity

    def setup(self):
        self.ks = [self.k_cls() for d in range(self.output_dim)]
        self.g = self.g_cls()

        if isinstance(self.ks[0], CovConvolutional) and \
                self.ks[0].inducing_patch == True and \
                not hasattr(self.g, 'gz'):
            raise ValueError('`CovConvolutional(inducing_patch=True)`'
                             'requires `g.gz` implemented')

    def K(self, X, Y=None):
        Ks = np.stack([k(X, Y, full_cov=True) for k in self.ks])   # (P, N, N')
        return Ks

    def Kdiag(self, X, Y=None):
        Ks = np.stack([k(X, Y, full_cov=False) for k in self.ks])  # (P, N)
        return Ks

    def Kuf(self, X, Y=None, full_cov=True):
        X = self.slice_and_map_inducing(X)
        Y = self.slice_and_map(Y)
        if not full_cov:
            raise ValueError('Kuf(full_cov=False) not valid')
        Ks = np.stack([k.Kuf(X, Y, full_cov=full_cov)
                      for k in self.ks])  # (P, N, N')
        return Ks

    def Kuu(self, X, Y=None, full_cov=True):
        X = self.slice_and_map_inducing(X)
        Y = self.slice_and_map_inducing(Y)
        if not full_cov:
            raise ValueError('Kuu(full_cov=False) not implemented')
        Ks = np.stack([k.Kuu(X, Y, full_cov=full_cov)
                      for k in self.ks])  # (P, N, N)
        return Ks

    def slice_and_map_inducing(self, X):
        X = apply_fn_ndarray_or_tuple(self.slice, X)
        X = apply_fn_ndarray_or_tuple(self.apply_mapping_inducing, X)
        return X

    def apply_mapping_inducing(self, X):
        if (X is None) or (not hasattr(self, 'g')):
            return X
        if isinstance(self.ks[0], CovConvolutional) and \
                self.ks[0].inducing_patch == True:
            return self.g.gz(X)
        else:
            return self.g(X)


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
        σ2I = np.kron(np.ones(int(n),), self.σ2)
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


class LikMulticlassSoftmax(Lik):
    """ Represents p(y|f) = Cat(π) where π = softmax(f) """
    output_dim: int = 1
    n_mc_samples: int = 20
    # Normalize posterior process `f`
    #     s.t.`exp(f) ~ Gamma(1/σ2, 1)`
    apx_gamma: bool = False

    def predictive_dist(self, μf, σ2f, full_cov=False):
        """ Computes posterior distribution via MC samples of `f` 
                     E[y] = ∫∫ y p(y|f)q(f) df dy
                          = ∫ softmax(f) q(f) df
                     V[y] = ∫∫ (y-E[y]) p(y|f)q(f) df dy
                          = ∫ softmax(f)*(1-softmax(f)) q(f) df
        """
        if full_cov:
            raise ValueError('`LikMulticlassSoftmax.predictive_dist`'
                             'full covariance not implemented!')
        D = self.output_dim
        μf, σ2f = μf.reshape((-1, D)), σ2f.reshape((-1, D))

        def predictive_mean(f):
            return jax.nn.softmax(f, axis=-1)

        def predictive_variance(f):
            p = jax.nn.softmax(f, axis=-1)
            return p - p**2

        μf = self.get_μf(μf, σ2f)
        Ey, Vy = reparam_mc_integration([predictive_mean, predictive_variance],
                                        self.make_rng('lik_mc_samples'),
                                        self.n_mc_samples,
                                        μf, σ2f)
        return Ey, Vy

    def logprob(self, f, y):
        """Computes log p(y|f) = Σᵢ yᵢ log( eᶠⁱ/(Σⱼeᶠʲ) ) 

            f,y    (N*L, D)
            logp   (N*L, 1)
        """
        logp = jax.nn.log_softmax(f)
        logp = np.sum(logp*y, axis=-1)
        return logp

    def variational_log_prob(self, y, μf, σ2f):
        """ Computes variational log prob with MC samples of `f`
                ∫ log p(y|p) q(f) df 

            Assumes `y` is one hot encoded 
        """
        N, D = y.shape
        μf, σ2f = μf.reshape((N, D)), σ2f.reshape((N, D))
        if D != self.output_dim:
            raise ValueError('`LikMulticlassSoftmax`: dimension mismatch')

        μf = self.get_μf(μf, σ2f)
        logp = reparam_mc_integration(self.logprob,
                                      self.make_rng('lik_mc_samples'),
                                      self.n_mc_samples,
                                      μf, σ2f,
                                      y=y)
        logp = np.sum(logp)
        return logp

    def get_μf(self, μf, σ2f):
        if self.apx_gamma:
            β = 1/(σ2f*np.exp(μf+σ2f/2)+1e-10)
            μf = μf + np.log(β+1e-10)
        return μf


def reparam_mc_integration(fns, key, L, μ, σ2, **kwargs):
    """Computes
            fn(S, **kwargs) for fn in fns
                where S_1,...,S_L ~ N(μ, σ2)
    """
    N, D = μ.shape
    ϵ = random.normal(key, (L, *μ.shape))
    S = np.sqrt(σ2)*ϵ + μ
    S = np.reshape(S, (L*N, D))

    for k, v in kwargs.items():
        D_v = v.shape[1]
        V = np.tile(v[np.newaxis, ...], (L, 1, 1))
        kwargs[k] = np.reshape(V, (L*N, D_v))

    def eval_fn(fn):
        fn_eval = fn(S, **kwargs)
        fn_eval = fn_eval.reshape((L, N, -1))
        fn_eval = np.mean(fn_eval, axis=0)
        return fn_eval

    if isinstance(fns, Iterable):
        return [eval_fn(fn) for fn in fns]
    else:
        return eval_fn(fns)


class LikMulticlassDirichlet(Lik):
    """ Represents p(y|f) = Cat(π) where π ~ Dir(α)
            where Dir(α) is approximated using LogNormals
                e.g. (eᶠ, ... eᶠᴷ) ~ Dir(α)
                      eᶠᵏ ~ LogNormal(μ, σ2) approx Gamma(α, 1)
                      fk  ~ Normal(μ, σ2)
    """
    output_dim: int = 1
    init_val_α_ϵ: float = .1
    init_val_α_δ: float = 10.
    approx_type: str = 'kl'
    n_mc_samples: int = 20

    def setup(self):
        self.α_ϵ = self.init_val_α_ϵ
        self.α_δ = self.init_val_α_δ
        self.α = np.array([self.α_ϵ, self.α_δ+self.α_ϵ])
        self.lognorm_y, self.lognorm_σ2 = gamma_to_lognormal(
            self.α, self.approx_type)

    def to_lognorm(self, y_onehot):
        y = np.asarray(y_onehot, np.int32)
        ỹ = self.lognorm_y[y]
        σ̃2 = self.lognorm_σ2[y]
        return ỹ, σ̃2

    def predictive_dist(self, μf, σ2f, full_cov=False):
        if full_cov:
            raise ValueError('`LikMulticlassDirichlet.predictive_dist`'
                             'full covariance not implemented!')
        D = self.output_dim
        μf, σ2f = μf.reshape((-1, D)), σ2f.reshape((-1, D))

        def predictive_mean(f):
            return jax.nn.softmax(f, axis=-1)

        def predictive_variance(f):
            p = jax.nn.softmax(f, axis=-1)
            return p - p**2

        Ey, Vy = reparam_mc_integration([predictive_mean, predictive_variance],
                                        self.make_rng('lik_mc_samples'),
                                        self.n_mc_samples,
                                        μf, σ2f)
        return Ey, Vy

    def variational_log_prob(self, y, μf, σ2f):
        """ Handles multiple response `y` """
        y = y.reshape(-1, 1)
        μf, σ2f = μf.reshape(y.shape), σ2f.reshape(y.shape)
        ỹ, σ̃2 = self.to_lognorm(y)
        return np.sum(-.5*np.log(2*np.pi*σ̃2) -
                      (.5/σ̃2)*(np.square(ỹ- μf) + σ2f))


def gamma_to_lognormal(α, approx_type='kl'):
    """Computes lognormal approximation to Gamma(α,1)
        Two types of `approx_type` possible,
            - 'kl'
                \min_{μ,σ2} = KL(LogNormal(μ,σ2)||Gamma(α,1))
                    implies α -> (ln(α)-.5*σ2, 1/α)

            - `moment`    match first & second moment
                    implies α -> (ln(α)-.5*σ2, log(1/α + 1))
    """
    if approx_type == 'kl':
        σ2 = 1/α
        μ = np.log(α) - σ2/2
    elif approx_type == 'moment':
        σ2 = np.log(1/α + 1)
        μ = np.log(α) - σ2/2
    else:
        raise ValueError(f'approx_type={approx_type} not implemented!')
    return μ, σ2


def gamma_to_lognormal_inv(μ, σ2,
                           approx_type='kl',
                           mc_key=random.PRNGKey(0),
                           mc_n_samples=1000):
    """Computes inverse of lognormal approximation to Gamma(α,1) 
    """
    import scipy

    if approx_type == 'kl':
        α = np.real(1/(scipy.special.lambertw(1/(2*np.exp(μ)))*2))
    elif approx_type == 'moment':
        """via interpolation"""
        y = np.logspace(np.log(1e-5), np.log(40), 2000, base=np.ℯ)
        α = np.exp(y-30)   # α = np.linspace(1e-10, 100, 2000)
        σ2 = np.log(1/α + 1)
        y = np.log(α) - σ2/2
        interp_fn = scipy.interpolate.interp1d(y, α)
        α = interp_fn(μ)
    elif approx_type == 'mc1':
        α = reparam_mc_integration(
            lambda f: np.exp(f), mc_key, mc_n_samples, μ, σ2)
    elif approx_type == 'mc2':
        μ = μ + np.log(1/(σ2*np.exp(μ+σ2/2)))
        α = reparam_mc_integration(
            lambda f: np.exp(f), mc_key, mc_n_samples, μ, σ2)
    else:
        raise ValueError(f'approx_type={approx_type} not implemented!')
    return α


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
    mean_fn_cls: Callable
    k_cls: Callable
    lik_cls: Callable
    data: Tuple[np.ndarray, np.ndarray]

    def setup(self):
        self.mean_fn = self.mean_fn_cls()
        self.k = self.k_cls()
        self.lik = self.lik_cls()

    @classmethod
    def get_init_params(self, model, key):
        Xs = np.zeros((1, model.data[0].shape[-1]))
        params = model.init(key, method=model.mll)
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
        mll = log_prob_mvn_tril(m, L, y)

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

    @classmethod
    def get_init_params(self, model, key):
        Xs = np.ones((1, model.data[0].shape[-1]))
        params = model.init(key, Xs, method=model.pred_f)
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

    @classmethod
    def get_init_params(self, model, key):
        params = model.init(key, np.ones((1, model.data[0].shape[-1])),
                            method=model.pred_f)
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
    mean_fn_cls: Callable
    k_cls: Callable
    lik_cls: Callable
    inducing_loc_cls: Callable
    n_data: int
    output_dim: int

    def setup(self):
        self.mean_fn = self.mean_fn_cls()
        self.k = self.k_cls()
        self.lik = self.lik_cls()
        self.Xu = self.inducing_loc_cls()
        self.n_inducing = self.Xu.shape[0]

        # Two types of variational distribution
        # (1) shared inducing points & independent multiple-output `q`
        #     only usable with multiple output kernel `CovMultipleOutputIndependent`
        #     equivalent to (2) when `CovIndex` is fixed to be identity
        # (2) shared inducing points & correlated multiple-output `q`
        #     use in conjunction with `CovICM`, e.g. len(Xu)=100, then 400 outputs ...
        #     In https://arxiv.org/pdf/1705.09862.pdf
        #     Assume its a large mvn where Σ models both input&output
        if isinstance(self.k_cls(), CovMultipleOutputIndependent):
            D, P = self.n_inducing, self.output_dim
        else:
            D, P = self.n_inducing*self.output_dim, 1
        self.q = VariationalMultivariateNormal(D=D, P=P)

    @classmethod
    def get_init_params(self, model, key, X_shape=None):
        """ If `X_shape` is not None, implies using interdomain inducing variables
                where `inducing_loc.shape` is different from `X.shape`
                so requires supplying shape of inputs for jax tracing
        """
        if X_shape is None:
            X_shape = model.inducing_loc_cls().shape[1:]
        k1, k2 = random.split(key)
        rngs = {'params': k1, 'lik_mc_samples': k2}
        n = 2
        # for interdomain inducing points in different domain, e.g. patches
        # might give error if inducing_loc.shape is not shape of inputs in original domain
        Xs = np.ones((n, *X_shape))
        ys = np.ones((n, model.output_dim))
        params = model.init(rngs, (Xs, ys), method=model.mll)
        return params

    def mll(self, data):
        X, y = data
        k = self.k
        Xu = self.Xu()               # (M,...)
        μq, Lq = self.q.μ, self.q.L  # (P, M) & (P, M, M)
        if μq.shape[0] == 1:
            μq, Lq = μq.squeeze(0), Lq.squeeze(0)

        mf = self.mean_fn(X)        # (N, P)
        mu = self.mean_fn(Xu)       # (M, P)

        Kff = k.Kff(X, full_cov=False)  # (P, N)
        Kuf = k.Kuf(Xu, X)              # (P, M, N)
        Kuu = k.Kuu(Xu)                 # (P, M, M)
        Luu = cholesky_jitter_vmap(Kuu, jitter=5e-5)  # (P, M, M)

        α = self.n_data/len(X) \
            if self.n_data is not None else 1.

        if isinstance(k, CovMultipleOutputIndependent):
            mvn_marginal_variational_fn = vmap(
                mvn_marginal_variational, (0, 0, 1, 0, 1, 0, 0, None), -1)  # along P-dim
            kl_mvn_tril_fn = vmap(kl_mvn_tril, (0, 0, 1, 0))
        else:
            mvn_marginal_variational_fn = mvn_marginal_variational
            kl_mvn_tril_fn = kl_mvn_tril

        μqf, σ2qf = mvn_marginal_variational_fn(Kff, Kuf, mf,
                                                Luu, mu, μq, Lq, False)
        μqf = μqf.reshape(σ2qf.shape)  # (N, P)

        if isinstance(self.lik, LikMultipleNormal):
            elbo_lik = α*self.lik.variational_log_prob(y, μqf, σ2qf, X[:, -1])
        else:
            elbo_lik = α*np.sum(self.lik.variational_log_prob(y, μqf, σ2qf))
        elbo_nkl = -np.sum(kl_mvn_tril_fn(μq, Lq, mu, Luu))
        elbo = elbo_lik + elbo_nkl

        return elbo

    def pred_f(self, Xs, full_cov=True):
        k = self.k
        Xu = self.Xu()               # (M,...)
        μq, Lq = self.q.μ, self.q.L  # (P, M) & (P, M, M)
        if μq.shape[0] == 1:
            μq, Lq = μq.squeeze(0), Lq.squeeze(0)

        ms = self.mean_fn(Xs)
        mu = self.mean_fn(Xu)

        Kss = k.Kff(Xs, full_cov=full_cov)
        Kus = k.Kuf(Xu, Xs)
        Kuu = k.Kuu(Xu)
        Luu = cholesky_jitter_vmap(Kuu, jitter=5e-5)  # (P, M, M)

        if isinstance(k, CovMultipleOutputIndependent):
            mvn_marginal_variational_fn = vmap(
                mvn_marginal_variational, (0, 0, 1, 0, 1, 0, 0, None), -1)  # along P-dim
        else:
            mvn_marginal_variational_fn = mvn_marginal_variational

        μf, Σf = mvn_marginal_variational_fn(Kss, Kus, ms,
                                             Luu, mu, μq, Lq, full_cov)
        if not full_cov:
            μf = μf.reshape(Σf.shape)  # (N, P)
        return μf, Σf


def SVGP_pred_f_details(self, Xs, output_gh=False):
    k = self.k
    Xu = self.Xu()               # (M,...)
    μq, Lq = self.q.μ, self.q.L  # (P, M) & (P, M, M)
    if μq.shape[0] == 1:
        μq, Lq = μq.squeeze(0), Lq.squeeze(0)

    ms = self.mean_fn(Xs)
    mu = self.mean_fn(Xu)

    Kss = k.Kff(Xs, full_cov=False)
    Kus = k.Kuf(Xu, Xs)
    Kuu = k.Kuu(Xu)
    Luu = cholesky_jitter_vmap(Kuu, jitter=5e-5)  # (P, M, M)
    
    def mvn_marginal_variational_details(Kff, Kuf, mf,
                                         Luu, mu, μq, Lq, full_cov=False):
        α = solve_triangular(Luu, Kuf, lower=True)
        β = solve_triangular(Luu.T, α, lower=False)
        γ = Lq.T@β
        if full_cov:
            Σg = γ.T@γ       # completely dependent on u, within data uncertainty
            Σh = Kff - α.T@α # completely indp of u, distributional uncertainty
            Σf = Σg + Σh
        else:
            Σg = np.sum(np.square(γ), axis=0)
            Σh = Kff - np.sum(np.square(α), axis=0)
            Σf = Σg + Σh
        # for multiple-output
        μq = μq.reshape(-1, 1)
        mu = mu.reshape(-1, 1)
        mf = mf.reshape(-1, 1)
        A = β.T; δ = (μq-mu)
        μf = mf + A@δ
        return μf, Σg, Σh, Σf, A, δ, mf

    if isinstance(k, CovMultipleOutputIndependent):
        mvn_marginal_variational_fn = vmap(
            mvn_marginal_variational_details, (0, 0, 1, 0, 1, 0, 0, None), -1)  # along P-dim
    else:
        mvn_marginal_variational_fn = mvn_marginal_variational_details

    μf, Σg, Σh, Σf, A, δ, mf = mvn_marginal_variational_fn(Kss, Kus, ms,
                                         Luu, mu, μq, Lq, False)
    # μf,      Σf/Σg/Σh,  A,        δ,       mf
    # (N, D), (N, D), (N, M, D), (M, D), (N, D)
    N, D = Σf.shape
    μf = μf.reshape((N,D))
    A = A.reshape((N,-1,D))
    δ = δ.reshape((-1,D))
    mf = mf.reshape((N,D))
    if output_gh:
        return μf, Σg, Σh, Σf, A, δ, mf
    else:
        return μf, Σf, A, δ, mf


class InducingLocations(nn.Module):
    shape: Tuple[int]  # shape output by init_fn
    init_fn: Callable
    transform_cls: Callable = LayerIdentity

    def setup(self):
        self.X = self.param('X', self.init_fn, self.shape)
        self.transform = self.transform_cls()

    def __call__(self):
        X = self.transform(self.X)
        return X


def transform_to_matrix(θ, T_type, A_init_val):
    """Given trainable parameter `θ`,  
        convert to 2x3 affine change of coordinate matrix `A` """
    assert(θ.ndim == 1)
    if T_type == 'transl':
        ind = jax.ops.index[[0, 1], [2, 2]]
    elif T_type == 'transl+isot_scal':
        ind = jax.ops.index[[0, 1, 0, 1], [0, 1, 2, 2]]
        θ = np.array([θ[0], θ[0], θ[1], θ[2]])
    elif T_type == 'transl+anis_scal':
        ind = jax.ops.index[[0, 1, 0, 1], [0, 1, 2, 2]]
    elif T_type == 'affine':
        ind = jax.ops.index[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]]
    A = jax.ops.index_update(A_init_val, ind, θ)
    return A


def spatial_transform_bound_init_fn(in_shape, out_shape):
    scal, transl = extract_patches_2d_scal_transl(in_shape, out_shape)
    bnd_transly = np.array((np.min(transl[:, 0]), np.max(transl[:, 0])))
    bnd_translx = np.array((np.min(transl[:, 1]), np.max(transl[:, 1])))
    bnd_scal = np.array([np.max(np.array((0.5*np.min(scal), np.array(0.)))),
                         np.min(np.array((1.5*np.max(scal), np.array(1.))))])
    return bnd_scal, bnd_transly, bnd_translx


class SpatialTransform(nn.Module):
    shape: Tuple[int]  # (h, w) output shape
    n_transforms: int
    T_type: str
    T_init_fn: Callable = None
    A_init_val: np.ndarray = np.array([[1, 0, 0],
                                       [0, 1, 0]], dtype=np.float32)
    output_transform: bool = False
    bound_init_fn: Callable = None

    def setup(self):
        if len(self.shape) != 2:
            raise ValueError(
                '`SpatialTransform.shape` should have dim of 2')
        if self.T_type not in ['transl', 'transl+isot_scal',
                               'transl+anis_scal', 'affine']:
            raise ValueError(
                f'`self.T_type`={self.T_type} not Implemented')

        if self.bound_init_fn is not None:
            self.bij = self.get_bij()

        T_init_shape, T_init_fn = self.default_T_init()
        if self.T_init_fn is not None:
            T_init_fn = self.T_init_fn
        self.T = self.params_to_matrix(
            self.param('T', T_init_fn, T_init_shape))

    def default_T_init(self):
        """Get initial shape and init_fn for spatial transformation θ 
                If `bound_fn` used, then these initial values lives in 
                the bouned space, e.g. init_scal=1 -> middle of scale bound """
        T_type, n = self.T_type, self.n_transforms
        if T_type == 'transl':
            init_shape, init_val = (n, 2), np.array([0, 0.])  # (tx, ty)
        elif T_type == 'transl+isot_scal':
            init_shape, init_val = (n, 3), np.array([1, 0, 0.])  # (s, tx, ty)
        elif T_type == 'transl+anis_scal':
            init_shape, init_val = (n, 4), np.array(
                [1, 1, 0, 0.])  # (sx, sy, tx, ty)
        elif T_type == 'affine':
            init_shape, init_val = (n, 6), np.array([1, 0, 0, 0, 1, 0.])

        def init_fn(k, s):
            return np.tile(init_val, (s[0], 1))
        return init_shape, init_fn

    def get_bij(self):
        T_type = self.T_type
        bnd_scal, bnd_transly, bnd_translx = self.bound_init_fn()
        if T_type == 'transl':
            bnds = [bnd_translx, bnd_transly]
        elif T_type == 'transl+isot_scal':
            bnds = [bnd_scal, bnd_translx, bnd_transly]
        elif T_type == 'transl+anis_scal':
            bnds = [bnd_scal, bnd_scal, bnd_translx, bnd_transly]
        elif T_type == 'affine':
            # for now just standard Sigmoid applied to free-form transform
            bnds = [np.array([0, 1]) for _ in range(6)]
        bound = np.column_stack(bnds)
        return BijSigmoid(bound)

    def param_bij(self, θ, direction):
        if self.bound_init_fn is None:
            return θ
        return getattr(self.bij, direction)(θ)

    def param_bij_forward(self, θ):
        return self.param_bij(θ, 'forward')

    def param_bij_reverse(self, θ):
        return self.param_bij(θ, 'reverse')

    def params_to_matrix(self, θ):
        """ θ -> A """
        θ = self.param_bij_forward(θ)
        fn = vmap(transform_to_matrix, (0, None, None), 0)
        return fn(θ, self.T_type, self.A_init_val)

    @property
    def scal(self):
        return self.T[:, [1, 0], [1, 0]]  # (sy, sx)

    @property
    def transl(self):
        return self.T[:, [1, 0], 2]  # (ty, tx)

    def __call__(self, X):
        """Spatially transforms batched image X (N, H, W, 1)
                to target image of size (N, h, w, 1) """
        h, w = self.shape
        A = self.T
        fn = vmap(spatial_transform, (0, 0, None), 0)
        X = fn(A, X, (h, w))
        if self.output_transform:
            return X, np.column_stack([self.scal, self.transl])
        else:
            return X


class MultivariateNormalTril(object):
    """N(μ, LLᵀ) representing multiple independent mvn
            where μ (P, D)
                  L (P, D, D)
    """

    def __init__(self, μ, L):
        P, D = μ.shape
        if L.shape != (P, D, D):
            raise ValueError(
                f'L should have dim ({P}, {D}, {D})')
        self.μ = μ
        self.L = L

    def log_prob(self, x):
        """x: (n, m) -> (n*m,)"""
        P, D = self.μ.shape
        # Note `x` usually (d, P) previous impl
        x = x.reshape((D, P))
        logp_vmap = vmap(log_prob_mvn_tril, (0, 0, 1), 0)
        logp = logp_vmap(self.μ, self.L, x)
        return np.sum(logp)

    def cov(self):
        return vmap(lambda X: X@X.T, (0,), 0)(self.L).squeeze()


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
    D: int = 1
    P: int = 1

    def setup(self):
        """ Variational Normal distribution with 
                μ (P, D) 
                L (P, D, D)
        """
        P, D = self.P, self.D
        self.μ = self.param('μ', jax.nn.initializers.zeros, (P, D))
        init_L_shape = (P, BijFillTril.reverse_shape(D))

        def init_L_fn(k, s):
            return np.repeat(BijFillTril.reverse(np.eye(D))[np.newaxis, ...], P, axis=0)
        self.L = vmap(BijFillTril.forward)(
            self.param('L', init_L_fn, init_L_shape))

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


class BijSigmoid(object):
    """ Logistic sigmoid bijector g(X) = 1/(1+exp(-X))
            if `bound` specified, then computes
                Y = lo + (hi-lo)*g(X)
                X = σ(Y) where σ is logistic function

        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/sigmoid.py
    """

    def __init__(self, bound=None):
        if bound is None:
            bound = np.array([0, 1.])
        self.bound = bound

    @property
    def lo(self):
        return self.bound[0]

    @property
    def hi(self):
        return self.bound[1]

    def forward(self, x):
        return self.hi*jax.nn.sigmoid(x) + self.lo*jax.nn.sigmoid(-x)

    def reverse(self, y):
        x = np.log(y-self.lo) - np.log(self.hi-y)
        return x


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


def log_prob_mvn_tril(μ, L, x):
    """ μ (d,)   L (d,d)   x (d,) """
    d = μ.size
    x = x.reshape((d,))
    μ = μ.reshape((d,))
    α = solve_triangular(L, (x-μ), lower=True)
    mahan = -.5*np.sum(np.square(α))
    lgdet = -np.sum(np.log(np.diag(L)))
    const = -.5*d*np.log(2*np.pi)
    return mahan + const + lgdet


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


class CNN(nn.Module):

    @nn.compact
    def __call__(self, x):
        conv = partial(nn.Conv, kernel_size=(4, 4), strides=(2, 2))
        assert(x.shape[1] == 224 and x.shape[2] == 224)
        x = x.reshape(-1, 224, 224, 1)
        # (1, 224, 224, 1)
        x = conv(features=16)(x)
        x = nn.relu(x)
        x = conv(features=32)(x)
        x = nn.relu(x)
        x = conv(features=64)(x)
        x = nn.relu(x)
        x = conv(features=128)(x)
        x = nn.relu(x)
        # (1, 14, 14, 128)
        x = np.mean(x, axis=(1, 2))
        # (1, 128)
        return x


class CNNMnist(nn.Module):
    output_dim: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x



class CNNMnistTrunk(nn.Module):
    # in_shape: (1, 28, 28, 1)
    # receptive field: (10, 10)

    @nn.compact
    def __call__(self, x):
        # (N, H, W, C)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # (N, Px, Py, L)
        return x

    def gz(self, x):
        if x.ndim == 3:
            x = x[..., np.newaxis]  # add color channel
        x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1), (0, 0)),
                   mode='constant',
                   constant_values=0)
        x = self.__call__(x)
        if x.shape[1:3] != (3, 3):
            raise ValueError(
                'CNNMnistTrunk.gz(x) does not have correct shape')
        x = x[:, 1, 1, :].squeeze()
        return x
    
    @classmethod
    def get_start_ind(self, image_shape):
        if image_shape == (14, 14, 1):
            start_ind = np.array([[0, 0], [0, 1], [0, 5], [1, 0], [
                                 1, 1], [1, 5], [5, 0], [5, 1], [5, 5]])
        elif image_shape == (28, 28, 1):
            start_ind = np.array(
                [[0, 0], [0, 1], [0, 5], [0, 9], [0, 13], [0, 17], [0, 21],
                 [1, 0], [1, 1], [1, 5], [1, 9], [1, 13], [1, 17], [1, 21],
                 [5, 0], [5, 1], [5, 5], [5, 9], [5, 13], [5, 17], [5, 21],
                 [9, 0], [9, 1], [9, 5], [9, 9], [9, 13], [9, 17], [9, 21],
                 [13, 0], [13, 1], [13, 5], [13, 9], [13, 13], [13, 17],
                 [13, 21], [17, 0], [17, 1], [17, 5], [17, 9], [17, 13],
                 [17, 17], [17, 21], [21, 0], [21, 1], [21, 5], [21, 9],
                 [21, 13], [21, 17], [21, 21]])
        else:
            start_ind, _ = compute_receptive_fields_start_ind_extrap(
                CNNMnistTrunk, (1,)+image_shape)
        return start_ind

    @classmethod
    def get_XL(self, image_shape=(28, 28, 1)):
        if image_shape not in [(28, 28, 1), (14, 14, 1)]:
            raise ValueError('CNNMnistTrunk.getXL() invalid image shape')
        start_ind = self.get_start_ind(image_shape)
        scal, transl = startind_to_scal_transl(
            image_shape[:2], (10, 10), start_ind)
        scal = np.repeat(scal[np.newaxis, ...], len(transl), axis=0)
        XL = np.column_stack([scal, transl])
        return XL
    



def compute_receptive_fields(model_def, in_shape, spike_loc=None):
    """Computes receptive fields using gradients
        For images, returns receptive fields for (h, w)
    """
    x = np.ones(in_shape)
    model = model_def()
    params = model.init(random.PRNGKey(0), x)
    params = freeze(jax.tree_map(lambda w: np.ones(w.shape),
                                 unfreeze(params)))
    # vjp (𝑥,𝑣)↦∂𝑓(𝑥)ᵀv
    # vjp :: (a -> b) -> a -> (b, CT b -> CT a)
    #     vjp: (f, x) -> (f(x), vjp_fn) where vjp_fn: u -> v
    def f(x): return model.apply(params, x)
    y, vjp_fn = vjp(f, x)
    S = y.shape
    gy = np.zeros(S)
    if spike_loc is not None:
        ind = jax.ops.index[0, spike_loc[:, 0], spike_loc[:, 1], ...]
    else:
        ind = jax.ops.index[0, S[1]//2, S[2]//2, ...]
    gy = jax.ops.index_update(gy, ind, 1)
    gx = vjp_fn(gy)[0]
    I = np.where(gx != 0)
    rf = np.array([np.max(idx)-np.min(idx)+1
                   for idx in I])[np.array([1, 2])]  # (y, x)
    return rf, gx, gy


def compute_receptive_fields_start_ind(model_def, in_shape):
    """Computes start indices for patches to get transformation 
            parameters (in start indices of patches)

        ```
        g_cls = CNNMnistTrunk; image_shape = (28, 28, 1)
        ind_start, rf = compute_receptive_fields_start_ind(
            g_cls, (1, *image_shape))
        fig,ax = plt.subplots(1,1,figsize=(5,5))
        ax.imshow(np.zeros(image_shape), cmap='Greys', origin='upper')
        ax.scatter(ind_start[:,0], ind_start[:,1])
        ax.grid()
        ``` 
    """
    if len(in_shape) != 4:
        raise ValueError('`in_shape` has dims (N, H, W, C)')

    image_shape = in_shape[1:1+2]  # ndim=2
    rf, _, gy = compute_receptive_fields(model_def, in_shape)
    Py, Px = gy.shape[1:3]
    P = Py*Px
    spike_locs = list(itertools.product(np.arange(Py), np.arange(Px)))
    spike_locs = np.array(spike_locs, dtype=np.int32)

    x = np.ones(in_shape)
    model = model_def()
    params = model.init(random.PRNGKey(0), x)
    params = freeze(jax.tree_map(lambda w: np.ones(w.shape),
                                 unfreeze(params)))

    def f(x): return model.apply(params, x)
    y, vjp_fn = vjp(f, x)

    def construct_gy(spike_loc):
        if len(spike_loc) != 2:
            raise ValueError(f'len(spike_loc)={len(spike_loc)}')
        gy = np.zeros(y.shape)
        ind = jax.ops.index[0, spike_loc[0], spike_loc[1], ...]
        gy = jax.ops.index_update(gy, ind, 1)
        gx = vjp_fn(gy)[0]
        return gx
    # (P, *image_shape)  squeeze batch-dim
    gx = vmap(construct_gy)(spike_locs).squeeze(1)
    ind = []
    for p in range(len(gx)):
        gxp = gx[p]
        I = np.where(gxp > np.mean(gxp)*.1)
        ind.append([(np.min(idx), np.max(idx)) for idx in I])

    # (P, 3, 2)
    ind = np.array(ind)[:, [0, 1]]
    # (P, hi/wi, min/max)
    if not np.all((ind[:, :, 1]-ind[:, :, 0]+1) <= rf):
        # Note patches on boundary have < receptive_field!
        raise ValueError('leaky gradient, `gx` has'
                         'more nonzero entries than possible')
    # (P, hi/wi)
    ind_start = ind[:, :, 0]

    return ind_start, rf


def compute_receptive_fields_start_ind_extrap(model_def, in_shape):
    """ Just 3 evalution of `vjp`, rest extrapolated ... """

    if len(in_shape) != 4:
        raise ValueError('`in_shape` has dims (N, H, W, C)')

    image_shape = in_shape[1:1+2]  # ndim=2
    rf, _, gy = compute_receptive_fields(model_def, in_shape)
    Py, Px = gy.shape[1:3]
    P = Py*Px
    spike_locs = list(itertools.product([0], np.arange(3)))
    spike_locs = np.array(spike_locs, dtype=np.int32)

    x = np.ones(in_shape)
    model = model_def()
    params = model.init(random.PRNGKey(0), x)
    params = freeze(jax.tree_map(lambda w: np.ones(w.shape),
                                 unfreeze(params)))

    def f(x): return model.apply(params, x)
    y, vjp_fn = vjp(f, x)

    def construct_gy(spike_loc):
        if len(spike_loc) != 2:
            raise ValueError(f'len(spike_loc)={len(spike_loc)}')
        gy = np.zeros(y.shape)
        ind = jax.ops.index[0, spike_loc[0], spike_loc[1], ...]
        gy = jax.ops.index_update(gy, ind, 1)
        gx = vjp_fn(gy)[0]
        return gx
    # (P, *image_shape)  squeeze batch-dim
    gx = vmap(construct_gy)(spike_locs).squeeze(1)
    ind = []
    for p in range(len(gx)):
        gxp = gx[p]
        I = np.where(gxp > np.mean(gxp)*.1)
        ind.append([(np.min(idx), np.max(idx)) for idx in I])

    # (P, 3, 2)
    ind = np.array(ind)[:, [0, 1]]

    # (P, hi/wi, min/max)
    if not np.all((ind[:, :, 1]-ind[:, :, 0]+1) <= rf):
        # Note patches on boundary have < receptive_field!
        raise ValueError('leaky gradient, `gx` has'
                         'more nonzero entries than possible')
    # (P, wi)
    ind_start = ind[:, 1, 0]
    offset_border = ind_start[1]-ind_start[0]
    step = ind_start[2]-ind_start[1]

    ind_start = list(itertools.product(np.arange(-1, Py-1),
                                       np.arange(-1, Px-1)))
    ind_start = np.array(ind_start)*step + offset_border
    ind_start = np.maximum(0, ind_start)
    
    return ind_start, rf




def cholesky_jitter(K, jitter=1e-5):
    L = linalg.cholesky(jax_add_to_diagonal(K, jitter))
    return L


def cholesky_jitter_vmap(K, jitter=1e-5):
    """ Handles batched kernel matrix """
    if K.ndim == 2:
        return cholesky_jitter(K, jitter)
    else:
        return vmap(cholesky_jitter, (0, None), 0)(K, jitter)


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


def get_data_stream(key, bsz, dataset):
    n = len(dataset[0] if isinstance(dataset, (list, tuple)) else dataset)
    n_complete_batches, leftover = divmod(n, bsz)
    n_batches = n_complete_batches + bool(leftover)

    def data_stream(key):
        while True:
            key, permkey = random.split(key)
            perm = random.permutation(permkey, n)
            for i in range(n_batches):
                ind = perm[i*bsz:(i+1)*bsz]
                if isinstance(dataset, np.ndarray):
                    yield dataset[ind]
                elif isinstance(dataset, (list, tuple)):
                    yield tuple((X[ind] for X in dataset))
                else:
                    # isinstance(dataset, torchvision.datasets.VisionDataset)
                    data = [dataset[i] for i in ind]
                    data_batched = tuple(np.stack(x) for x in list(zip(*data)))
                    yield data_batched

    return n_batches, data_stream(key)


def filter_contains(k, v, kwd, b=True):
    return b if kwd in k.split('/') else (not b)


def pytree_path_contains_keywords(path, kwds):
    # Check if any kwds appears in `path` of pytree
    return True if any(k in path for k in kwds) else False


def pytree_get_kvs(tree):
    """Gets `tree['params']` to {path: np.ndarray, ...} """
    kvs = {}
    for k, v in flax.traverse_util.flatten_dict(
            unfreeze(tree['params'])).items():
        kvs['/'.join(k)] = v
    return kvs


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


def pytree_mutate_with_fn(params, path, mutate_fn):
    """ mutates `param` at `path` with 
            `mutate_fn`: np.ndarray -> np.ndarray """
    a = pytree_leaf(params, path)
    a = mutate_fn(a)
    return pytree_mutate(params, {path: a})


def pytree_leave(tree, name):
    for k, v in flax.traverse_util.flatten_dict(
            unfreeze(tree)).items():
        path = '/'.join(k)
        if path.endswith(name):
            return v


def pytree_leaves(tree, names):
    """Access `tree` leaves using `ks: [path]` """
    leafs = [np.nan for _ in range(len(names))]
    for k, v in flax.traverse_util.flatten_dict(
            unfreeze(tree)).items():
        path = '/'.join(k)
        for i, name in enumerate(names):
            if path.endswith(name):
                leafs[i] = v
    return leafs


def pytree_leaf(tree, path):
    import operator
    try:
        a = functools.reduce(operator.getitem, path.split('/'), tree)
    except:
        a = np.nan
    return a


def pytree_keys(tree):
    kvs = flax.traverse_util.flatten_dict(unfreeze(tree))
    return ['/'.join(k) for k, v in kvs.items()]


def flax_check_traversal(params, traversal):

    if isinstance(params, (dict, flax.core.FrozenDict)):
        state = optim.GradientDescent().create(
            params, traversal).state
        kvs = flax.traverse_util.flatten_dict(unfreeze(state.param_states))
        return ['/'.join(k) for k, v in kvs.items() if v is not None]
    else:
        def flax_optim_get_params_dict(inputs):
            if isinstance(inputs, flax.nn.base.Model):
                return inputs.params
            elif isinstance(inputs, (dict, flax.core.FrozenDict)):
                return flax.core.unfreeze(inputs)
            else:
                raise ValueError(
                    'Can only traverse a flax Model instance or a nested dict, not '
                    f'{type(inputs)}')

        def flax_optim_sorted_items(x):
            """Returns items of a dict ordered by keys."""
            return sorted(x.items(), key=lambda x: x[0])

        def iterate_path(traversal, inputs):
            params = flax_optim_get_params_dict(inputs)
            flat_dict = flax.traverse_util.flatten_dict(params)
            for key, value in flax_optim_sorted_items(flat_dict):
                path = '/' + '/'.join(key)
                if traversal._filter_fn(path, value):
                    yield path

        return list(iterate_path(traversal, params.params))


def flax_check_multiopt(params, opt):
    for hyper_param, traversal in zip(opt.optimizer_def.hyper_params,
                                      opt.optimizer_def.traversals):
        print(hyper_param)
        print(flax_check_traversal(params, traversal))


def pytree_check_device(tree):
    return jax.tree_util.tree_map(lambda x: x.device_buffer.device(), tree)


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


def flax_create_multioptimizer(params, optimizer_name, optimizer_kwargs, traversal_filter_fns):
    traversals_and_optimizers = []
    for filter_fn, opt_kwargs in zip(traversal_filter_fns,
                                     itertools.cycle(optimizer_kwargs)):
        traversal = optim.ModelParamTraversal(filter_fn)
        opt_def = flax_get_optimizer(optimizer_name)(**opt_kwargs)
        traversals_and_optimizers.append((traversal, opt_def))
    opt_def = optim.MultiOptimizer(*traversals_and_optimizers)
    opt = opt_def.create(params)
    return opt


def flax_create_multioptimizer_2focus(params, optimizer_name, optimizer_kwargs, filter_fn_kwds):
    """ Create multiopimizer with 2 focus, where 1st focus uses `filter_fn_kwds` """
    traversal_filter_fns = [lambda p, v: pytree_path_contains_keywords(p, filter_fn_kwds),
                            lambda p, v: not pytree_path_contains_keywords(p, filter_fn_kwds)]
    return flax_create_multioptimizer(params, optimizer_name, optimizer_kwargs, traversal_filter_fns)


def flax_run_optim(f, params, num_steps=10, log_func=None,
                   optimizer='GradientDescent',
                   optimizer_kwargs={'learning_rate': .002},
                   optimizer_focus=None):
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


def pytree_save(tree, path):
    import pickle
    with open(path, 'wb') as file:
        pickle.dump(tree, file)


def pytree_load(tree, path):
    import pickle
    from flax import serialization
    with open(path, 'rb') as file:
        output = pickle.load(file)  # onp.ndarray
    new_tree = serialization.from_state_dict(tree, output)
    return new_tree


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


def flax_model2params(target):
    if isinstance(target, flax.nn.base.Model):
        params = flax.core.freeze(target.params)
    else:
        params = target
    return params


def flax_params2model(model, params):
    return flax.nn.base.Model(model, unfreeze(params))


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


def extract_patches_2d_nojit(im, patch_size):
    if im.ndim == 3 and im.shape[2] != 1:
        raise ValueError('`extract_patches_2d` only supports C=1')
    h, w = patch_size
    H, W = im.shape[0], im.shape[1]
    P = (H-h+1)*(W-w+1)
    patches = []
    for hi in range(H-h+1):
        for wi in range(W-w+1):
            patches.append(im[hi:hi+h, wi:wi+w, ...])
    patches = np.stack(patches)
    return patches


def extract_patches_2d(im, patch_size):
    """Extract patches with size `patch_size` from image `im` 
            (H, W, 1) -> (P, h, w) where P=#patches
    """
    if im.ndim == 3 and im.shape[2] != 1:
        raise ValueError('`extract_patches_2d` only supports C=1')
    h, w = patch_size
    H, W = im.shape[0], im.shape[1]
    im = im.reshape((H, W))
    P = (H-h+1)*(W-w+1)
    hi = np.arange(H-h+1)
    wi = np.arange(W-w+1)
    hwi = np.array(list(itertools.product(hi, wi)))
    def f(hwi): return jax.lax.dynamic_slice(im, (hwi[0], hwi[1]), (h, w))
    patches = jax.lax.map(f, hwi)
    return patches


def extract_patches_2d_vmap(ims, patch_size):
    return vmap(extract_patches_2d, (0, None), 0)(ims, patch_size)


def extract_patches_2d_scal_transl(image_shape, patch_shape):
    """Computes scaling `s` and translation `t` 
            in the sense of spatial transforms impl 
            for patches as output of `extract_patches_2d` """
    H, W = image_shape[:2]
    h, w = patch_shape
    hi = np.arange(H-h+1)
    wi = np.arange(W-w+1)
    hwi = np.array(list(itertools.product(hi, wi)))
    s, t = startind_to_scal_transl(image_shape, patch_shape, hwi)
    return s, t


def startind_to_scal_transl(image_shape, patch_shape, start_ind):
    image_shape = np.array(image_shape[:2])
    patch_shape = np.array(patch_shape)
    s = patch_shape/image_shape
    t = 2*start_ind/image_shape - 1 + s
    return s, t


def trans2x3_from_scal_transl(s, t):
    """ s = [sy, sx]; t = [ty, tx] """
    return np.array([[s[1], 0, t[1]], [0, s[0], t[0]]])


def make_im_grid(ims, im_per_row=8, padding=1, pad_value=.2):
    """Makes a grid of image from batched images

        ims    (N, H, W, C)
        grid   (H, N*W, C) if no padding
    """
    if ims.ndim == 3:
        ims = ims[..., np.newaxis]
    N, H, W, C = ims.shape
    n_col = min(im_per_row, N)
    n_row = int(math.ceil(N/n_col))
    H = int(H+padding)
    W = int(W+padding)
    k = 0
    grid = np.full((H*n_row+padding, W*n_col+padding, C),
                   pad_value)
    for ri in range(n_row):
        for ci in range(n_col):
            if k > N:
                break
            grid = jax.ops.index_update(grid,
                                        jax.ops.index[(ri*H+padding):(ri*H+H),
                                                      (ci*W+padding):(ci*W+W), ...],
                                        ims[k])
            k += 1
    return grid


def rotated_ims(x, rot_range=(0, 180), n_ims=10):
    import scipy
    degs = np.linspace(*rot_range, n_ims)
    ims = []
    for deg in degs:
        im = scipy.ndimage.rotate(x, angle=deg,
                                  reshape=False,
                                  order=1,
                                  mode='constant', cval=0.)
        ims.append(im)
    ims = np.stack(ims)
    return ims


def homogeneous_grid(height, width):
    """ Returns target grid in homogeneous coordinate
            grid    (3, width*height)
    """
    Xt, Yt = np.meshgrid(np.linspace(-1, 1, width),
                         np.linspace(-1, 1, height))
    ones = np.ones(Xt.size)
    grid = np.vstack([Xt.flatten(), Yt.flatten(), ones])
    return grid


def grid_sample(S, G):
    """Use bilinear interpolation to interpolate values 
        of `S` at source grid `G`'s location
    '"""
    w, h, c = S.shape
    X, Y = G

    # (-1, 1) -> (0, height/width)
    X = w*(X+1)/2
    Y = h*(Y+1)/2

    X0 = np.floor(X).astype(np.int32)
    Y0 = np.floor(Y).astype(np.int32)
    X1 = X0+1
    Y1 = Y0+1

    Xmax = w-1
    Ymax = h-1
    X0 = np.clip(X0, 0, Xmax)
    X1 = np.clip(X1, 0, Xmax)
    Y0 = np.clip(Y0, 0, Ymax)
    Y1 = np.clip(Y1, 0, Ymax)

    #
    # a -- c
    # |  x |
    # b -- d
    #
    wa = (X1-X)*(Y1-Y)
    wb = (X1-X)*(Y-Y0)
    wc = (X-X0)*(Y1-Y)
    wd = (X-X0)*(Y-Y0)

    S_flat = S.reshape(h*w, c)
    Sa = S_flat[np.ravel_multi_index((Y0, X0), dims=(h, w), mode='wrap'), ...]
    Sb = S_flat[np.ravel_multi_index((Y1, X0), dims=(h, w), mode='wrap'), ...]
    Sc = S_flat[np.ravel_multi_index((Y0, X1), dims=(h, w), mode='wrap'), ...]
    Sd = S_flat[np.ravel_multi_index((Y1, X1), dims=(h, w), mode='wrap'), ...]

    T = wa[..., np.newaxis]*Sa + \
        wb[..., np.newaxis]*Sb + \
        wc[..., np.newaxis]*Sc + \
        wd[..., np.newaxis]*Sd

    return T


def spatial_transform(A, S, Tsize):
    """Differentiable transformation of `S` via
        application of homogeneous matrix `A` 
            to get target image `T` of size `Tsize`

        Following allows for (uniform) stretching by `s`
            and translation of `tx` and `ty`
            A = [s  0  tx]
                [0  s  ty]

       A     (2, 3)
       S     (h, w, c)
       Tsize (2,)
    """
    height, width = Tsize
    Gt = homogeneous_grid(height, width)
    Gs = A@Gt
    Xs_flat = Gs[0, :]
    Ys_flat = Gs[1, :]
    Xs = Xs_flat.reshape(*Tsize)
    Ys = Ys_flat.reshape(*Tsize)
    T = grid_sample(S, (Xs, Ys))
    return T


@partial(jax.jit, static_argnums=(2,))
def spatial_transform_details(A, S, Tsize):
    height, width = Tsize
    Gt = homogeneous_grid(height, width)
    Gs = A@Gt
    Xs_flat = Gs[0, :]
    Ys_flat = Gs[1, :]
    Xs = Xs_flat.reshape(*Tsize)
    Ys = Ys_flat.reshape(*Tsize)
    T = grid_sample(S, (Xs, Ys))
    return T, Gs


def plt_spatial_transform(axs, Gs, S, T):
    """ Given `axs` of size 2, source grid `Gs` 
            draw source image `S` with source grid `Gs` and
            target spatially transformed image `T`
    """
    h, w = T.shape[0], T.shape[1]
    Gt = homogeneous_grid(h, w)
    Xt, Yt = np.meshgrid(np.linspace(-1, 1, h),
                         np.linspace(-1, 1, w))
    Xs_flat = Gs[0, :]
    Ys_flat = Gs[1, :]
    Xs = Xs_flat.reshape((h, w))
    Ys = Ys_flat.reshape((h, w))

    ax = axs[1]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(Xs, Ys, marker='+', c='r', s=40, lw=1)
    ax.imshow(S, cmap='Greys', extent=(-1, 1, 1, -1), origin='upper')

    ax = axs[0]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(Xt, Yt, marker='+', c='r', s=40, lw=1)
    ax.imshow(T, cmap='Greys', extent=(-1, 1, 1, -1), origin='upper')


def plt_inducing_inputs_spatial_transform(params, model, max_show=10):
    
    ind = np.arange(max_show)
    
    m = model.bind(params)
    A = m.Xu.transform.T
    S = pytree_leaf(params, 'params/Xu/X')
    A = A[ind]; S = S[ind]

    fn = vmap(spatial_transform_details, (0, 0, None), 0)
    T, Gs = fn(A, S, patch_shape)
    fig, axs = plt.subplots(2, len(A), figsize=(3*len(A),3*2))
    for i in range(len(T)):
        plt_spatial_transform(axs[:,i], Gs[i], S[i], T[i])
    fig.tight_layout()
    
    return fig
    