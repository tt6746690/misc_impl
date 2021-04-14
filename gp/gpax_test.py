import unittest
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import sys
sys.path.append('../kernel')

import numpy as onp

import jax
from jax import numpy as np
from jax import random
import jax.numpy.linalg as linalg
from jax.scipy.linalg import cho_solve, solve_triangular

import torch

from gpax import *
from jaxkern import *


class TestJaxUtilities(unittest.TestCase):

    def test_add_to_diagonal(self):

        n = 100
        key = random.PRNGKey(0)
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


class TestMeanFn(unittest.TestCase):

    def test_MeanConstant(self):

        key = random.PRNGKey(0)
        n = 3
        X = np.linspace(0,1,n).reshape(-1,1)
        c = np.array([0,.5,1])
        output_dim = 3

        mean_fn = MeanConstant(output_dim=output_dim,
                            flat=True)
        params = {'params': {'c': c}}
        mean_fn = mean_fn.bind(params)

        m = mean_fn(X)
        mtrue = np.repeat(c.reshape(1,-1),3,axis=0).reshape(n*output_dim,1)
        test_entries = np.array_equal(m, mtrue)

        self.assertTrue(test_entries)


class TestLikelihoods(unittest.TestCase):

    def test_LikNormal(self):

        key = random.PRNGKey(0)
        lik = LikNormal()
        μ = np.ones((10,1)).astype(np.float32)
        Σ = np.eye(10)
        params = lik.init(key, μ, Σ, method=lik.predictive_dist)
        μ1, Σ1 = lik.apply(params, μ, Σ, full_cov=True, method=lik.predictive_dist)
        self.assertTrue(np.array_equal(μ, μ1))
        self.assertTrue(np.array_equal(Σ+np.eye(10), Σ1))


    def test_LikMultipleNormalKron(self):

        n, T = 10, 2
        lik = LikMultipleNormalKron(output_dim=T)
        lik = lik.bind({'params': {'σ2': np.arange(T)}})
        Σ = np.zeros((n,n))
        Σy = lik.predictive_dist(None, Σ)[1] 
        test_diagonal_entries = np.allclose(np.diag(Σy),
                                            np.kron(lik.σ2, np.ones((n//2,))))


class TestKernel(unittest.TestCase):

    def test_CovIndex(self):
        for d in [1,3]: # active_dims
            for i in range(2):
                key = random.PRNGKey(i)
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


    def test_CovIndexSpherical(self):

        key = random.PRNGKey(0)
        k = CovIndexSpherical(output_dim=4)
        X = random.randint(key, (3,1), 0, 3)
        params = k.init(key, X)
        k = k.bind(params)
        B = k.cov()
        test_diagonal_entries = np.array_equal(np.diag(B), np.full((4,), 1))


    def test_CovICM(self):

        for T in [3,5]:
            n, d = 12, 2

            key = random.PRNGKey(0)
            X = random.normal(key, (n,d))
            k = CovICM(kx_cls=CovSE,
                       kt_kwargs={'output_dim': T, 'rank': 1})
            params = k.init(key, X)
            k = k.bind(params)
            K = k(X)
            Kdiag = k(X, full_cov=False)

            test_output_dim = (K.shape[0] == n*T) and (K.shape[1] == n*T)
            test_diag_entries = np.allclose(Kdiag, np.diag(K), rtol=1e-6).item()

            self.assertTrue(test_output_dim)
            self.assertTrue(test_diag_entries)

    def test_CovICMLearnable(self):
        
        key = random.PRNGKey(0)
        m = 3
        nr = 5

        for nc in [nr,10]:
            X = random.normal(key, (nr,5))
            Y = random.normal(key, (nc,5))
            k = CovICMLearnable(output_dim=m)
            k = k.bind(k.init(key, X))
            K = k(X, Y)
            Kdiag = k(X, full_cov=False)

            lhs = np.kron(k.kx(X, Y), np.ones((m,m)))
            rhs = []
            for i in range(m):
                Eii = np.zeros((m,m))
                ind = (np.array([i]), np.array([i]))
                v = np.array([1.])
                Eii = jax.ops.index_update(Eii, ind, v)
                Kti = np.kron(k.kt[i](X, Y), Eii)
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
        i,j = 1,0
        m = 2
        n = 2
        A = np.array([[ 0,  1,  2,  3],
                      [ 4,  5,  6,  7],
                      [ 8,  9, 10, 11],
                      [12, 13, 14, 15]],)
        ind = np.arange(m*n, step=m)
        ind = tuple(x.T for x in np.meshgrid(ind, ind))
        ind = (ind[0]+i, ind[1]+j)
        v = np.array([1,2,3,4]).reshape(2,2)*1000
        A = jax.ops.index_update(A, ind, v)
        Atrue = np.array([[   0,    1,    2,    3],
                          [1000,    5, 2000,    7],
                          [   8,    9,   10,   11],
                          [3000,   13, 4000,   15]])
        self.assertTrue(np.allclose(Atrue, A))

        # asymmetric A 
        i,j = 1,0
        T = 2
        n = 3
        m = 2
        A = np.arange(24).reshape(4,6)
        ind = np.meshgrid(np.arange(T*m, step=T),
                          np.arange(T*n, step=T))
        ind = tuple(x.T for x in ind)
        ind = (ind[0]+i, ind[1]+j)
        v = np.arange(6).reshape(2,3)*1000
        A = jax.ops.index_update(A, ind, v)
        Atrue = np.array([[   0,    1,    2,    3,    4,    5],
                          [   0,    7, 1000,    9, 2000,   11],
                          [  12,   13,   14,   15,   16,   17],
                          [3000,   19, 4000,   21, 5000,   23]])
        self.assertTrue(np.allclose(Atrue, A))



class TestKL(unittest.TestCase):

    def test_kl_mvn(self):

        import tensorflow_probability as tfp

        i, m = 0, 50
        μ0,Σ0 = rand_μΣ(random.PRNGKey(i), m)
        μ1,Σ1 = rand_μΣ(random.PRNGKey(i*2), m)
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
        kl5 = tfp.distributions.kl_divergence(
            tfp.distributions.MultivariateNormalTriL(
                loc=μ0, scale_tril=L0),
            tfp.distributions.MultivariateNormalTriL(
                loc=μ1, scale_tril=L1)).numpy()

        if not np.isnan(kl1):
            self.assertTrue(np.allclose(kl1, kl2, rtol=1e-3))
        self.assertTrue(np.allclose(kl2, kl3))
        self.assertTrue(np.allclose(kl2, kl4))
        self.assertTrue(np.allclose(kl2, kl5))


class TestMvnConditional(unittest.TestCase):

    def test_mvn_conditional_exact(self):

        n, ns = 5, 3
        key = random.PRNGKey(0)
        k1, k2, k3 = random.split(key, 3)
        X  = random.normal(k1, (n, 3))
        Xs = random.normal(k2, (ns,3))
        y  = random.uniform(k3, (n,1))

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

        μ,Σ = mvn_conditional_exact(
            Kss, Kfs, ms, L, mf, y, full_cov=True)
        _,Σd = mvn_conditional_exact(
            np.diag(Kss), Kfs, ms, L, mf, y, full_cov=False)

        test_μ_entries = np.allclose(μt, μ)
        test_Σ_entries = np.allclose(Σt, Σ)
        test_Σ_diagonal_entries = np.allclose(np.diag(Σt), Σd)

        self.assertTrue(test_μ_entries)
        self.assertTrue(test_Σ_entries)
        self.assertTrue(test_Σ_diagonal_entries)

    def test_mvn_conditional_exact_multipleoutput(self):
        
        T = 2
        n, ns = 5, 3
        key = random.PRNGKey(0)
        k1, k2, k3 = random.split(key, 3)
        X  = random.normal(k1, (n, 3))
        Xs = random.normal(k2, (ns,3))
        y  = random.uniform(k3, (n, T))

        k = CovICM(kt_kwargs={'output_dim': T})
        k = k.bind(k.init(key, X))
        mean_fn = MeanConstant(output_dim=T)
        mean_fn = mean_fn.bind({'params': {'c': np.full((T,),.2)}})

        Kff = k(X)
        Kfs = k(X, Xs)
        Kss = k(Xs)
        L = linalg.cholesky(Kff)

        mf = mean_fn(X)
        ms = mean_fn(Xs)

        α = cho_solve((L, True), (y.reshape(-1,1)-mf))
        μt = Kfs.T@α + ms
        v = solve_triangular(L, Kfs, lower=True)
        Σt = Kss - v.T@v

        μ,Σ = mvn_conditional_exact(
            Kss, Kfs, ms, L, mf, y, full_cov=True)
        _,Σd = mvn_conditional_exact(
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
            μq = μq.reshape(-1,1)
            mu = mu.reshape(-1,1)
            mf = mf.reshape(-1,1)

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

        for T in [1,2]:
            n,m,l = 10,3,5
            key = random.PRNGKey(0)
            X, Xu = random.normal(key, (n,2)), random.normal(key, (m,2))
            k = CovICM(kt_kwargs={'output_dim':T})
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

        import tensorflow_probability as tfp

        key = random.PRNGKey(0)
        n = 10
        μ, Σ = rand_μΣ(key, n)
        L = linalg.cholesky(Σ)
        y = random.uniform(key, (n,))

        p = tfp.distributions.MultivariateNormalTriL(
            loc=μ, scale_tril=L, validate_args=True)
        log_prob_true = p.log_prob(y).numpy()

        p = MultivariateNormalTril(μ, L)
        log_prob = p.log_prob(y)
        log_prob_unflattened = p.log_prob(y.reshape(-1,2))

        test_log_prob = np.allclose(log_prob_true, log_prob)
        test_log_prob_unflattened = np.allclose(log_prob_true, log_prob_unflattened)

        self.assertTrue(test_log_prob)
        self.assertTrue(test_log_prob_unflattened)





if __name__ == '__main__':
    unittest.main()