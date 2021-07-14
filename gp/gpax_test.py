import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

from functools import partial
import sys
import unittest
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import torch
import numpy as onp

has_tfp = False
try:
    import tensorflow_probability as tfp
    has_tfp = True
except:
    pass
import jax
jax.config.update('jax_platform_name', 'cpu') # for multi-cpu pytest s!

from jax.scipy.linalg import cho_solve, solve_triangular
import jax.numpy.linalg as linalg
from jax import random
from jax import numpy as np

from gpax import *


class TestNumpyBehavior(unittest.TestCase):

    def test_ravel_multi_index(self):
        m, n = 4, 5
        for i in range(m):
            for j in range(n):
                a = np.ravel_multi_index((i, j), (m, n))
                b = i*n+j
                test_same = a == b
                self.assertTrue(test_same)

        indtrue = np.ravel_multi_index(
            np.tile(np.arange(m), (2, 1)), (m, m))
        ind = [i*m+i for i in range(m)]
        test_diagonal_indices = np.array_equal(indtrue, ind)
        self.assertTrue(test_diagonal_indices)


class TestJaxUtilities(unittest.TestCase):

    def test_add_to_diagonal(self):

        n = 100
        key = random.PRNGKey(0)
        A = random.normal(key, (n, n))
        jitter = 10

        a = jax_add_to_diagonal(A, jitter)
        b = A+jitter*np.eye(len(A))
        self.assertTrue(np.array_equal(a, b))

        v = np.ones(n)*jitter
        a = jax_add_to_diagonal(A, v)
        b = A+np.diag(v)
        self.assertTrue(np.array_equal(a, b))


class TestReceptiveFields(unittest.TestCase):

    def test_cnnmnist(self):
        g_cls = CNNMnistTrunk; image_shape = (28, 28, 1)
        rf_true = np.array([10, 10], np.int32)
        ind_start, rf = compute_receptive_fields_start_ind(
            g_cls, (1, *image_shape))
        ind_start_extrap, rf = compute_receptive_fields_start_ind_extrap(
            g_cls, (1, *image_shape))
        ind_start_true = np.array([[0, 0], [0, 1], [0, 5], [0, 9], [0, 13], [0, 17], [0, 21], [1, 0], [1, 1], [1, 5], [1, 9], [1, 13], [1, 17], [1, 21], [5, 0], [5, 1], [5, 5], [5, 9], [5, 13], [5, 17], [5, 21], [9, 0], [9, 1], [9, 5], [9, 9], [9, 13], [9, 17], [9, 21], [13, 0], [13, 1], [13, 5], [13, 9], [13, 13], [13, 17], [13, 21], [17, 0], [17, 1], [17, 5], [17, 9], [17, 13], [17, 17], [17, 21], [21, 0], [21, 1], [21, 5], [21, 9], [21, 13], [21, 17], [21, 21]])
        rf_correct = np.all(rf == rf_true)
        ind_start_correct = np.all(ind_start==ind_start_true)
        ind_start_extrap_correct = np.all(ind_start_extrap==ind_start_true)
        self.assertTrue(ind_start_correct)
        self.assertTrue(ind_start_extrap_correct)
        self.assertTrue(rf_correct)

    @unittest.skip("CNNCxrTrunk yet to be impl")
    def test_cnn224x224(self):
        g_cls = CNNCxrTrunk; image_shape = (224, 224, 1)
        rf_true = np.array([46, 46], np.int32)
        ind_start_extrap, rf = compute_receptive_fields_start_ind_extrap(
            g_cls, (1, *image_shape))
        ind_start_true = [[0, 0], [0, 1], [0, 17], [0, 33], [0, 49], [0, 65], [0, 81], [0, 97], [0, 113], [0, 129], [0, 145], [0, 161], [0, 177], [0, 193], [1, 0], [1, 1], [1, 17], [1, 33], [1, 49], [1, 65], [1, 81], [1, 97], [1, 113], [1, 129], [1, 145], [1, 161], [1, 177], [1, 193], [17, 0], [17, 1], [17, 17], [17, 33], [17, 49], [17, 65], [17, 81], [17, 97], [17, 113], [17, 129], [17, 145], [17, 161], [17, 177], [17, 193], [33, 0], [33, 1], [33, 17], [33, 33], [33, 49], [33, 65], [33, 81], [33, 97], [33, 113], [33, 129], [33, 145], [33, 161], [33, 177], [33, 193], [49, 0], [49, 1], [49, 17], [49, 33], [49, 49], [49, 65], [49, 81], [49, 97], [49, 113], [49, 129], [49, 145], [49, 161], [49, 177], [49, 193], [65, 0], [65, 1], [65, 17], [65, 33], [65, 49], [65, 65], [65,81], [65, 97], [65, 113], [65, 129], [65, 145], [65, 161], [65, 177], [65, 193], [81, 0], [81, 1], [81, 17], [81, 33], [81, 49], [81, 65], [81, 81], [81, 97], [81, 113], [81, 129], [81, 145], [81, 161], [81, 177], [81, 193], [97, 0], [97, 1], [97, 17], [97, 33], [97, 49], [97, 65], [97, 81], [97, 97], [97, 113], [97, 129], [97, 145], [97, 161], [97, 177], [97, 193], [113, 0], [113, 1], [113, 17], [113, 33], [113, 49], [113, 65], [113, 81], [113, 97], [113, 113], [113, 129], [113, 145], [113, 161], [113, 177], [113, 193], [129, 0], [129, 1], [129, 17], [129, 33], [129, 49], [129, 65], [129, 81], [129, 97], [129, 113], [129, 129], [129, 145], [129, 161], [129, 177], [129, 193], [145, 0], [145, 1], [145, 17], [145, 33], [145, 49], [145, 65], [145, 81], [145, 97], [145, 113], [145, 129], [145, 145], [145, 161], [145, 177], [145, 193], [161, 0], [161, 1], [161, 17], [161, 33], [161, 49], [161, 65], [161, 81], [161, 97], [161, 113], [161, 129], [161, 145], [161, 161], [161, 177], [161, 193], [177, 0], [177, 1], [177, 17], [177, 33], [177, 49], [177, 65], [177, 81], [177, 97], [177, 113], [177, 129], [177, 145], [177, 161], [177, 177], [177, 193], [193, 0], [193, 1], [193, 17], [193, 33], [193, 49], [193, 65], [193, 81], [193, 97], [193, 113], [193, 129], [193, 145], [193, 161], [193, 177], [193, 193]]

        rf_correct = np.all(rf == rf_true)
        ind_start_extrap_correct = np.all(ind_start_extrap==ind_start_true)
        self.assertTrue(ind_start_extrap_correct)
        self.assertTrue(rf_correct)


class TestBijectors(unittest.TestCase):

    def test_BijFillTril(self):
        n = 6
        v = np.arange(n, dtype=np.float32)
        L = BijFillTril.forward(v)
        w = BijFillTril.reverse(L)
        K = np.array([[0, 0, 0],
                      [1, 2, 0],
                      [3, 4, 5]])
        same_vec = np.array_equal(v, w)
        same_mat = np.array_equal(K, L)
        self.assertTrue(same_vec)
        self.assertTrue(same_mat)

    def TestSigmoid(self):

        bound = np.array([-2, .25])
        bij = BijSigmoid(bound)
        x = np.linspace(-10,10,100)
        cycleconsistent = np.allclose(x, bij.reverse(bij.forward(x)), rtol=1e-4)
        self.assertTrue(cycleconsistent)

        # same code works for batch bound/inputs as well
        key = random.PRNGKey(0)
        bnd_scal, bnd_transly, bnd_translx = np.array([0,1]), np.array([1,2]), np.array([.2,.3])
        bnds = [bnd_scal, bnd_translx, bnd_transly]
        x = random.normal(key, (10,3))
        bijs = [BijSigmoid(bnd) for bnd in bnds]
        y = np.column_stack([bij.forward(x[:,i])
                            for i, bij in enumerate(bijs)])
        bound = np.column_stack(bnds)
        bij = BijSigmoid(bound)
        same = np.all(y == bij.forward(x))
        self.assertTrue(same)


class TestMeanFn(unittest.TestCase):

    def test_MeanConstant(self):

        key = random.PRNGKey(0)
        n = 3
        X = np.linspace(0, 1, n).reshape(-1, 1)
        c = np.array([0, .5, 1])
        output_dim = 3

        mean_fn = MeanConstant(output_dim=output_dim,
                               flat=True)
        params = {'params': {'c': c}}
        mean_fn = mean_fn.bind(params)

        m = mean_fn(X)
        mtrue = np.repeat(c.reshape(1, -1), 3, axis=0).reshape(n*output_dim, 1)
        test_entries = np.array_equal(m, mtrue)

        self.assertTrue(test_entries)


class TestLikelihoods(unittest.TestCase):

    def test_LikNormal(self):

        key = random.PRNGKey(0)
        lik = LikNormal()
        μ = np.ones((10, 1)).astype(np.float32)
        Σ = np.eye(10)
        params = lik.init(key, μ, Σ, method=lik.predictive_dist)
        μ1, Σ1 = lik.apply(params, μ, Σ, full_cov=True,
                           method=lik.predictive_dist)
        self.assertTrue(np.array_equal(μ, μ1))
        self.assertTrue(np.array_equal(Σ+np.eye(10), Σ1))

    def test_LikMultipleNormalKron(self):

        n, T = 10, 2
        lik = LikMultipleNormalKron(output_dim=T)
        lik = lik.bind({'params': {'σ2': np.arange(T)}})
        Σ = np.zeros((n, n))
        Σy = lik.predictive_dist(None, Σ)[1]
        test_diagonal_entries = np.allclose(np.diag(Σy),
                                            np.kron(lik.σ2, np.ones((n//2,))))


class TestKernel(unittest.TestCase):

    def test_CovConstant(self):

        key = random.PRNGKey(0)
        x = random.normal(key, (10,3))
        y = random.normal(key, (20,3))
        k = CovConstant()
        params = k.init(key, x)
        k = k.bind(params)
        Kdiag = k.Kdiag(x)
        K = k.K(x)
        test_Kdiagshape = ( Kdiag.shape == (len(x),) )
        test_Kshape = ( K.shape == (len(x), len(x)) )

        self.assertTrue(test_Kdiagshape)
        self.assertTrue(test_Kshape)


    def test_CovLin(self):
        key = random.PRNGKey(0)
        X = random.normal(key, (3, 10))
        Y = random.normal(key, (3, 10))+2

        k = CovLin(ard_len=1)
        k = k.bind(k.init(key, X))

        diagK = np.diag(k(X))
        Kdiag = k(X, full_cov=False)
        Kdiag_agrees = np.allclose(diagK, Kdiag)
        self.assertTrue(Kdiag_agrees)

    def test_CovConvolutional(self):
        
        key = random.PRNGKey(0)
        N, M = 1, 2
        Np, Mp = 3, 4
        image_shape = (3, 3, 1)
        patch_shape = (2, 2)
        x = random.normal(key, (N, *image_shape))
        y = random.normal(key, (M, *image_shape))
        xp = random.normal(key, (Np, *patch_shape))
        yp = random.normal(key, (Mp, *patch_shape))
        xl = random.normal(key, (Np, 4))
        yl = random.normal(key, (Mp, 4))

        # test shape when u,f ~ GP(0,k)
        kg_cls = partial(CovPatch, image_shape=image_shape,
                                   patch_shape=patch_shape,
                                   kp_cls=CovSE)
        k = CovConvolutional(kg_cls=kg_cls,
                            inducing_patch=False)
        params = k.init(key, x)

        for X, Y, shape, method in [(x, y, (N, M), k.Kff),
                                    (x, y, (N, M), k.Kuf),
                                    (x, y, (N, M), k.Kuu)]:
            K = k.apply(params, X, Y, method=method)
            Kshape_correct = K.shape == shape
            self.assertTrue(Kshape_correct)



        # test shape when u ~ GP(0,ku) f ~ GP(0, kf)
        k = CovConvolutional(kg_cls=kg_cls,
                            inducing_patch=True)
        params = k.init(key, x)
        for X, Y, shape, method in [(x, None, (N, N), k.Kff),
                                    (x, y, (N, M), k.Kff),
                                    (y, y, (M, M), k.Kff),
                                    (xp, x, (Np, N), k.Kuf),
                                    (xp, y, (Np, M), k.Kuf),      # k(Xu, X)
                                    (xp, None, (Np, Np), k.Kuu),  # k(Xu)
                                    (xp, xp, (Np, Np), k.Kuu),
                                    (xp, yp, (Np, Mp), k.Kuu)]:
            K = k.apply(params, X, Y, method=method)
            Kshape_correct = K.shape == shape
            self.assertTrue(Kshape_correct)
            

        # test shape  when u ~ GP(0,ku) f ~ GP(0, kf)
        #      and use non-constant location kernel kl
        xpl = (xp, xl)
        ypl = (yp, yl)
        kg_cls = partial(CovPatch, image_shape=image_shape,
                                   patch_shape=patch_shape,
                                   kp_cls=CovSE,
                                   kl_cls=CovSE)
        k = CovConvolutional(kg_cls=kg_cls,
                             inducing_patch=True)
        params = k.init(key, x)
        for X, Y, shape, method in [(x, None, (N, N), k.Kff),
                                    (x, y, (N, M), k.Kff),
                                    (y, y, (M, M), k.Kff),
                                    (xpl, x, (Np, N), k.Kuf),
                                    (xpl, y, (Np, M), k.Kuf),      # k(Xu, X)
                                    (xpl, None, (Np, Np), k.Kuu),  # k(Xu)
                                    (xpl, xpl, (Np, Np), k.Kuu),
                                    (xpl, ypl, (Np, Mp), k.Kuu)]:
            K = k.apply(params, X, Y, method=method)
            Kshape_correct = K.shape == shape
            self.assertTrue(Kshape_correct)


    def test_CovConvolutionalwithMultipleIndependentOutput(self):

        key = random.PRNGKey(0)
        O = 2
        N, M = 1, 2
        Np, Mp = 3, 4
        image_shape = (28, 28, 1)
        patch_shape = (10, 10)
        x = random.normal(key, (N, *image_shape))
        y = random.normal(key, (M, *image_shape))
        xp = random.normal(key, (Np, *patch_shape))
        yp = random.normal(key, (Mp, *patch_shape))
        xl = random.normal(key, (Np, 4))
        yl = random.normal(key, (Mp, 4))

        g_cls = CNNMnistTrunk
        kg_cls = CovPatchEncoder
        kf_cls = partial(CovConvolutional, kg_cls=kg_cls,
                                           inducing_patch=False)
        k_cls = partial(CovMultipleOutputIndependent, k_cls=kf_cls,
                                                    output_dim=O,
                                                    g_cls=g_cls)
        k = k_cls()
        params = k.init(key, x)


        for X, Y, shape, method in [(x, y, (O, N, M), k.Kff),
                                    (x, y, (O, N, M), k.Kuf),
                                    (x, y, (O, N, M), k.Kuu)]:
            K = k.apply(params, X, Y, method=method)
            Kshape_correct = K.shape == shape
            self.assertTrue(Kshape_correct)

        # test shape when u ~ GP(0,ku) f ~ GP(0, kf)
        kf_cls = partial(CovConvolutional, kg_cls=kg_cls,
                                           inducing_patch=True)
        k_cls = partial(CovMultipleOutputIndependent, k_cls=kf_cls,
                                                    output_dim=O,
                                                    g_cls=g_cls)
        k = k_cls()
        params = k.init(key, x)

        for X, Y, shape, method in [(x, None, (O, N, N), k.Kff),
                                    (x, y, (O, N, M), k.Kff),
                                    (y, y, (O, M, M), k.Kff),
                                    (xp, x, (O, Np, N), k.Kuf),
                                    (xp, y, (O, Np, M), k.Kuf),      # k(Xu, X)
                                    (xp, None, (O, Np, Np), k.Kuu),  # k(Xu)
                                    (xp, xp, (O, Np, Np), k.Kuu),
                                    (xp, yp, (O, Np, Mp), k.Kuu)]:
            K = k.apply(params, X, Y, method=method)
            Kshape_correct = K.shape == shape
            self.assertTrue(Kshape_correct)
            
        # test shape  when u ~ GP(0,ku) f ~ GP(0, kf)
        #      and use non-constant location kernel kl
        xpl = (xp, xl)
        ypl = (yp, yl)
        kg_cls = partial(CovPatchEncoder, kl_cls=CovSE)
        kf_cls = partial(CovConvolutional, kg_cls=kg_cls,
                                           inducing_patch=True)
        k_cls = partial(CovMultipleOutputIndependent, k_cls=kf_cls,
                                                    output_dim=O,
                                                    g_cls=g_cls)
        k = k_cls()
        params = k.init(key, x)

        for X, Y, shape, method in [(x, None, (O, N, N), k.Kff),
                                    (x, y, (O, N, M), k.Kff),
                                    (y, y, (O, M, M), k.Kff),
                                    (xpl, x, (O, Np, N), k.Kuf),
                                    (xpl, y, (O, Np, M), k.Kuf),      # k(Xu, X)
                                    (xpl, None, (O, Np, Np), k.Kuu),  # k(Xu)
                                    (xpl, xpl, (O, Np, Np), k.Kuu),
                                    (xpl, ypl, (O, Np, Mp), k.Kuu)]:
            K = k.apply(params, X, Y, method=method)
            Kshape_correct = K.shape == shape
            self.assertTrue(Kshape_correct)


    def test_CovIndex(self):

        def distmat(func, x, y):
            if y == None: y = x
            return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)

        def LookupKernel(X, Y, A):
            return distmat(lambda x, y: A[x, y], X, Y)

        key = random.PRNGKey(0)
        for d in [2]:
            X = random.randint(key, (3, 2), 0, 2)
            Y = random.randint(key, (3, 2), 0, 2)
            k = CovIndex(active_dims=[d], output_dim=3, rank=1)
            params = k.init(key, X)
            W = params['params']['W']
            v = BijSoftplus.forward(params['params']['v'])
            B = W@W.T+np.diag(v)
            K1 = k.apply(params, X, Y, full_cov=True)
            K2 = LookupKernel(X[:, d], Y[:, d], B)
            self.assertTrue(np.array_equal(K1, K2))
            K1diag = k.apply(params, X, full_cov=False)
            K2diag = np.diag(K2)
            # self.assertTrue(np.array_equal(K1diag, K2diag))


    def test_CovIndexSpherical(self):

        key = random.PRNGKey(0)
        k = CovIndexSpherical(output_dim=4)
        X = random.randint(key, (3, 1), 0, 3)
        params = k.init(key, X)
        k = k.bind(params)
        B = k.cov()
        test_diagonal_entries = np.array_equal(np.diag(B), np.full((4,), 1))

    def test_CovICM(self):

        T, n, d = 2, 2, 1
        key = random.PRNGKey(0)
        X = random.normal(key, (n, d))
        kt_cls = partial(CovIndex, output_dim=T, rank=1)
        k = CovICM(kt_cls=kt_cls)
        params = k.init(key, X)
        K = k.apply(params, X)
        Kdiag = k.apply(params, X, full_cov=False)

        test_output_dim = (K.shape[0] == n*T) and (K.shape[1] == n*T)
        test_diag_entries = np.allclose(
            Kdiag, np.diag(K), rtol=1e-6).item()

        self.assertTrue(test_output_dim)
        self.assertTrue(test_diag_entries)

    def test_CovICMLearnable(self):

        key = random.PRNGKey(0)
        m = 2
        nr = 3

        for nc in [nr, 4]:
            X = random.normal(key, (nr, 2))
            Y = random.normal(key, (nc, 2))
            k = CovICMLearnable(mode='all', output_dim=m)
            k = k.bind(k.init(key, X))
            K = k(X, Y)
            Kdiag = k(X, full_cov=False)

            lhs = np.kron(k.kx(X, Y), np.ones((m, m)))
            rhs = []
            for i in range(m):
                for j in range(m):
                    Eij = np.zeros((m, m))
                    ind = (np.array([i]), np.array([j]))
                    v = np.array([1.])
                    Eij = jax.ops.index_update(Eij, ind, v)
                    kti = np.ravel_multi_index((i, j), (m, m))
                    Kti = np.kron(k.kt[kti](X, Y), Eij)
                    rhs.append(Kti)
            rhs = np.sum(np.stack(rhs), axis=0)
            Ktrue = lhs*rhs

            test_K = np.allclose(Ktrue, K)
            test_Kdiag = (nr != nc) or np.allclose(np.diag(Ktrue), Kdiag)
            test_size_K = K.size == (m*nr)*(m*nc)

            self.assertTrue(test_K)
            self.assertTrue(test_Kdiag)
            self.assertTrue(test_size_K)

    def test_CovICMLearnableMeshgrid(self):
        # symmetric A
        i, j = 1, 0
        m = 2
        n = 2
        A = np.array([[0,  1,  2,  3],
                      [4,  5,  6,  7],
                      [8,  9, 10, 11],
                      [12, 13, 14, 15]],)
        ind = np.arange(m*n, step=m)
        ind = tuple(x.T for x in np.meshgrid(ind, ind))
        ind = (ind[0]+i, ind[1]+j)
        v = np.array([1, 2, 3, 4]).reshape(2, 2)*1000
        A = jax.ops.index_update(A, ind, v)
        Atrue = np.array([[0,    1,    2,    3],
                          [1000,    5, 2000,    7],
                          [8,    9,   10,   11],
                          [3000,   13, 4000,   15]])
        self.assertTrue(np.allclose(Atrue, A))

        # asymmetric A
        i, j = 1, 0
        T = 2
        n = 3
        m = 2
        A = np.arange(24).reshape(4, 6)
        ind = np.meshgrid(np.arange(T*m, step=T),
                          np.arange(T*n, step=T))
        ind = tuple(x.T for x in ind)
        ind = (ind[0]+i, ind[1]+j)
        v = np.arange(6).reshape(2, 3)*1000
        A = jax.ops.index_update(A, ind, v)
        Atrue = np.array([[0,    1,    2,    3,    4,    5],
                          [0,    7, 1000,    9, 2000,   11],
                          [12,   13,   14,   15,   16,   17],
                          [3000,   19, 4000,   21, 5000,   23]])
        self.assertTrue(np.allclose(Atrue, A))


class TestKL(unittest.TestCase):

    def test_kl_mvn(self):

        i, m = 0, 10
        μ0, Σ0 = rand_μΣ(random.PRNGKey(i), m)
        μ1, Σ1 = rand_μΣ(random.PRNGKey(i*2), m)
        μ1 = np.zeros((m,))
        L0 = linalg.cholesky(Σ0)
        L1 = linalg.cholesky(Σ1)

        kl1 = kl_mvn(μ0, Σ0, μ1, Σ1)
        kl2 = kl_mvn_tril(μ0, L0, μ1, L1)
        kl3 = kl_mvn_tril_zero_mean_prior(μ0, L0, L1)
        kl4 = torch.distributions.kl_divergence(
            torch.distributions.MultivariateNormal(
                loc=torch.tensor(onp.array(μ0)),
                scale_tril=torch.tensor(onp.array(L0))),
            torch.distributions.MultivariateNormal(
                loc=torch.tensor(onp.array(μ1)),
                scale_tril=torch.tensor(onp.array(L1)))).mean().item()

        if not np.isnan(kl1):
            self.assertTrue(np.allclose(kl1, kl2, rtol=1e-3))
        self.assertTrue(np.allclose(kl2, kl3))
        self.assertTrue(np.allclose(kl2, kl4))

        if has_tfp:
            kl5 = tfp.distributions.kl_divergence(
                tfp.distributions.MultivariateNormalTriL(
                    loc=μ0, scale_tril=L0),
                tfp.distributions.MultivariateNormalTriL(
                    loc=μ1, scale_tril=L1)).numpy()
            self.assertTrue(np.allclose(kl2, kl5))



class TestMvnConditional(unittest.TestCase):

    def test_mvn_conditional_exact(self):

        n, ns = 5, 3
        key = random.PRNGKey(0)
        k1, k2, k3 = random.split(key, 3)
        X = random.normal(k1, (n, 3))
        Xs = random.normal(k2, (ns, 3))
        y = random.uniform(k3, (n, 1))

        k = CovSE()
        k = k.bind(k.init(key, X))
        mean_fn = MeanConstant(output_dim=1)
        mean_fn = mean_fn.bind({'params': {'c': np.array([.2])}})

        Kff = k(X)
        Kfs = k(X, Xs)
        Kss = k(Xs, Xs)
        L = linalg.cholesky(Kff)

        mf = mean_fn(X)
        ms = mean_fn(Xs)

        α = cho_solve((L, True), (y-mf))
        μt = Kfs.T@α + ms
        v = solve_triangular(L, Kfs, lower=True)
        Σt = Kss - v.T@v

        μ, Σ = mvn_conditional_exact(
            Kss, Kfs, ms, L, mf, y, full_cov=True)
        _, Σd = mvn_conditional_exact(
            np.diag(Kss), Kfs, ms, L, mf, y, full_cov=False)

        test_μ_entries = np.allclose(μt, μ)
        test_Σ_entries = np.allclose(Σt, Σ)
        test_Σ_diagonal_entries = np.allclose(np.diag(Σt), Σd)

        self.assertTrue(test_μ_entries)
        self.assertTrue(test_Σ_entries)
        self.assertTrue(test_Σ_diagonal_entries)

    def test_mvn_conditional_exact_multipleoutput(self):

        T = 2
        n, ns = 4, 2
        key = random.PRNGKey(0)
        k1, k2, k3 = random.split(key, 3)
        X = random.normal(k1, (n, 2))
        Xs = random.normal(k2, (ns, 2))
        y = random.uniform(k3, (n, T))

        kt_cls = partial(CovIndex, output_dim=T)
        k = CovICM(kt_cls=kt_cls)
        k = k.bind(k.init(key, X))
        mean_fn = MeanConstant(output_dim=T)
        mean_fn = mean_fn.bind({'params': {'c': np.full((T,), .2)}})

        Kff = k(X)
        Kfs = k(X, Xs)
        Kss = k(Xs)
        L = linalg.cholesky(Kff)

        mf = mean_fn(X)
        ms = mean_fn(Xs)

        α = cho_solve((L, True), (y.reshape(-1, 1)-mf))
        μt = Kfs.T@α + ms
        v = solve_triangular(L, Kfs, lower=True)
        Σt = Kss - v.T@v

        μ, Σ = mvn_conditional_exact(
            Kss, Kfs, ms, L, mf, y, full_cov=True)
        _, Σd = mvn_conditional_exact(
            np.diag(Kss), Kfs, ms, L, mf, y, full_cov=False)

        test_μ_entries = np.allclose(μt, μ)
        test_Σ_entries = np.allclose(Σt, Σ)
        test_Σ_diagonal_entries = np.allclose(np.diag(Σt), Σd)

        self.assertTrue(test_μ_entries)
        self.assertTrue(test_Σ_entries)
        self.assertTrue(test_Σ_diagonal_entries)

    def test_mvn_marginal_variational(self):

        from jax.scipy.linalg import solve_triangular

        def mvn_marginal_variational_unstable(
                Kff, Kuf, Kuu, μq, Σq, mf, mu, full_cov=False):
            """ Unstable version of `mvn_marginal_variational` """
            # for multiple-output
            μq = μq.reshape(-1, 1)
            mu = mu.reshape(-1, 1)
            mf = mf.reshape(-1, 1)

            Lq = linalg.cholesky(Σq)
            Luu = linalg.cholesky(Kuu)
            μf = mf + Kuf.T@linalg.solve(Kuu, (μq-mu))
            α = solve_triangular(Luu, Kuf, lower=True)
            Qff = α.T@α
            β = linalg.solve(Kuu, Kuf)
            if full_cov:
                Σf = Kff - Qff + β.T@Σq@β
            else:
                Σf = np.diag(Kff - Qff + β.T@Σq@β)
            return μf, Σf

        for T in [1, 2]:
            n, m, l = 10, 3, 5
            key = random.PRNGKey(0)
            X, Xu = random.normal(key, (n, 2)), random.normal(key, (m, 2))
            kt_cls = partial(CovIndex, output_dim=T)
            k = CovICM(kt_cls=kt_cls)
            k = k.bind(k.init(key, X))
            mean_fn = MeanConstant(init_val_m=.2, output_dim=T)
            mean_fn = mean_fn.bind(mean_fn.init(key, X))
            mf, mu = mean_fn(X), mean_fn(Xu)
            Kff = k(X)
            Kuf = k(Xu, X)
            Kuu = k(Xu)+1*np.eye(m*T)
            μq, Σq = rand_μΣ(key, m*T)
            Lq = linalg.cholesky(Σq)
            μf1, Σf1 = mvn_marginal_variational_unstable(
                Kff, Kuf, Kuu, μq, Σq, mf, mu, full_cov=True)
            μf2, Σf2 = mvn_marginal_variational(
                Kff, Kuf, mf, linalg.cholesky(Kuu), mu, μq, Lq, full_cov=True)
            test_μ_entries = np.allclose(μf1, μf1)
            test_Σ_entries = np.allclose(Σf1, Σf2, rtol=1e-5, atol=1e-5)

            self.assertTrue(test_μ_entries)
            self.assertTrue(test_Σ_entries)


class TestDistributions(unittest.TestCase):

    def test_MultivariateNormalTril(self):

        if has_tfp:

            key = random.PRNGKey(0)
            n = 10
            μ, Σ = rand_μΣ(key, n)
            L = linalg.cholesky(Σ)
            y = random.uniform(key, (n,))

            p = tfp.distributions.MultivariateNormalTriL(
                loc=μ, scale_tril=L, validate_args=True)
            log_prob_true = p.log_prob(y).numpy()
            log_prob = log_prob_mvn_tril(μ, L, y)
            test_log_prob = np.allclose(log_prob_true, log_prob)

            self.assertTrue(test_log_prob)


class TestSpatialTransformations(unittest.TestCase):

    def test_SpatialTransforms(self):

        key = random.PRNGKey(0)
        n = 1
        x = random.normal(key, (n,14,14,1))
        target_shape = (5, 5)
        T_type = 'transl'
        A_init_val = np.array([[.25,0,0], [0,.25,0]])
        st = SpatialTransform(target_shape, n, T_type)
        params = st.init(key, x)
        t = st.apply(params, x)
        self.assertTrue(t.shape == (n, *target_shape, 1))

        θs = pytree_leaf(params, 'params/θ')
        self.assertTrue(θs.shape == st.apply(params, method=st.default_T_init)[0])

    def test_SpatialTransformBounds(self):

        key = random.PRNGKey(0)
        image_shape = (14,14)
        patch_shape = (7,7)
        T_type = 'transl+isot_scal'
        bound_init_fn = partial(spatial_transform_bound_init_fn,
                                in_shape=image_shape,
                                out_shape=patch_shape)
        bnd_scal, bnd_transly, bnd_translx = bound_init_fn()
        st = SpatialTransform(shape=patch_shape,
                              n_transforms=3,
                              T_type=T_type,
                              output_transform=True,
                              bound_init_fn=bound_init_fn)

        x = random.normal(key, (3,*image_shape,1))
        params = st.init(key, x)
        params = pytree_mutate(params, {'params/θ': random.normal(key, (3,3))})
        st = st.bind(params)
        Tx, scal_transl = st(x)
        θ = pytree_leaf(params, 'params/θ')
        transly_correct = np.all(BijSigmoid(bnd_transly).forward(θ[:,1]) == scal_transl[:,0+2])
        translx_correct = np.all(BijSigmoid(bnd_translx).forward(θ[:,2]) == scal_transl[:,1+2])
        scaly_correct = np.all(BijSigmoid(bnd_scal).forward(θ[:,0]) == scal_transl[:,0])
        scalx_correct = np.all(BijSigmoid(bnd_scal).forward(θ[:,0]) == scal_transl[:,1])

        self.assertTrue(transly_correct)
        self.assertTrue(translx_correct)
        self.assertTrue(scaly_correct)
        self.assertTrue(scalx_correct)



if __name__ == '__main__':
    unittest.main()
