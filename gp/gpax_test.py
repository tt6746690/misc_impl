import unittest

import sys
sys.path.append('../kernel')

import numpy as onp

import jax
from jax import numpy as np
from jax import random
import jax.numpy.linalg as linalg

import torch

from gpax import *
from jaxkern import *


class TestJaxUtilities(unittest.TestCase):

    def test_add_to_diagonal(self):

        n = 100
        key = jax.random.PRNGKey(0)
        A = random.normal(key, (n,n))
        jitter = 10

        a = jax_add_to_diagonal(A, jitter)
        b = A+jitter*np.eye(len(A))
        self.assertTrue(np.array_equal(a,b))

        v = np.ones(n)*jitter
        a = jax_add_to_diagonal(A, v)
        b = A+np.diag(v)
        self.assertTrue(np.array_equal(a,b))


class TestBijectors(unittest.TestCase):

    def test_BijFillTril(self):
        n = 6
        v = np.arange(n, dtype=np.float32)
        L = BijFillTril.forward(v)
        w = BijFillTril.reverse(L)
        K = np.array([[0,0,0],
                      [1,2,0],
                      [3,4,5]])
        same_vec = np.array_equal(v,w)
        same_mat = np.array_equal(K,L)
        self.assertTrue(same_vec)
        self.assertTrue(same_mat)


class TestLikelihoods(unittest.TestCase):

    def test_LikNormal(self):

        key = jax.random.PRNGKey(0)
        lik = LikNormal()
        μ = np.ones((10,1)).astype(np.float32)
        Σ = np.eye(10)
        params = lik.init(key, μ, Σ, method=lik.predictive_dist)
        μ1, Σ1 = lik.apply(params, μ, Σ, full_cov=True, method=lik.predictive_dist)
        self.assertTrue(np.array_equal(μ, μ1))
        self.assertTrue(np.array_equal(Σ+np.eye(10), Σ1))


class TestKernel(unittest.TestCase):

    def test_CovIndex(self):
        for d in [1,3]: # active_dims
            for i in range(2):
                key = jax.random.PRNGKey(i)
                X = random.randint(key, (10,5), 0, 3)
                Y = random.randint(key, (10,5), 0, 3)
                k = CovIndex(active_dims=[d], output_dim=4, rank=2)
                params = k.init(key, X)
                W = params['params']['W']
                v = BijSoftplus.forward(params['params']['v'])
                B = W@W.T+np.diag(v)
                K1 = k.apply(params, X, Y, full_cov=True)
                K2 = LookupKernel(X[:,d],Y[:,d], B)
                self.assertTrue(np.array_equal(K1, K2))
                K1diag = k.apply(params, X, full_cov=False)
                K2diag = np.diag(LookupKernel(X[:,d],X[:,d], B))
                self.assertTrue(np.array_equal(K1diag, K2diag))


def rand_μΣ(key, m):
    μ = random.normal(key, (m, 1))
    Σ = random.normal(key, (m, m))
    Σ = Σ@Σ.T
    return μ, Σ


class TestKL(unittest.TestCase):

    def test_kl_mvn(self):

        for i, m in enumerate([30,50]):
            μ0,Σ0 = rand_μΣ(random.PRNGKey(i), m)
            μ1,Σ1 = rand_μΣ(random.PRNGKey(i*2), m)
            μ1 = np.zeros((m,1))
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


class TestMvnConditional(unittest.TestCase):

    def test_mvn_conditional_variational(self):

        from jax.scipy.linalg import solve_triangular

        def mvn_conditional_variational_unstable(Kff, Kuf, Kuu, μq, Σq, full_cov=False):
            """ Unstable version of `mvn_conditional_variational` """
            Lq = linalg.cholesky(Σq)
            Luu = linalg.cholesky(Kuu)
            μf = Kuf.T@linalg.solve(Kuu, μq)
            α = solve_triangular(Luu, Kuf, lower=True)
            Qff = α.T@α
            β = linalg.solve(Kuu, Kuf)
            if full_cov:
                Σf = Kff - Qff + β.T@Σq@β
            else:
                Σf = np.diag(Kff - Qff + β.T@Σq@β)
            return μf, Σf


        n,m,l = 100,30,5
        key = jax.random.PRNGKey(0)
        X, Xu = random.normal(key, (n,2)), random.normal(key, (m,2))
        k = CovSE()
        Kff = k.apply(k.init(key, Xu), X)
        Kuf = k.apply(k.init(key, Xu), Xu, X)
        Kuu = k.apply(k.init(key, Xu), Xu)+1*np.eye(m)
        μq, Σq = rand_μΣ(key, m)
        Lq = linalg.cholesky(Σq)
        μf1, Σf1 = mvn_conditional_variational_unstable(Kff, Kuf, Kuu, μq, Σq, full_cov=True)
        μf2, Σf2 = mvn_conditional_variational(Kff, Kuf, linalg.cholesky(Kuu), μq, Lq, full_cov=True)
        self.assertTrue(np.allclose(μf1, μf1))
        self.assertTrue(np.allclose(Σf1, Σf2, rtol=1e-5, atol=1e-5))



if __name__ == '__main__':
    unittest.main()