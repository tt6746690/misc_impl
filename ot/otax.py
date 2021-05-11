from functools import partial

import jax
import jax.numpy as np
from jax.scipy.special import logsumexp


def sinkhorn_knopp(a, b, C, ϵ, ρ, n_iters):
    """Matrix scaling for entropic regularized optimal transport

        Reference
            - https://arxiv.org/pdf/1306.0895.pdf
    """

    n, m = C.shape
    a = np.reshape(a, (n, -1))
    b = np.reshape(b, (m, -1))
    K = np.exp(-C/ϵ)
    λ = ρ / (ρ + ϵ)

    def body_fn(i, val):
        u, v = val
        u = (a/(K  @v))**λ
        v = (b/(K.T@u))**λ
        return u, v
    
    init_val = (np.ones((n,1))/n, np.ones((m,1))/m)
    u, v = jax.lax.fori_loop(0, n_iters, body_fn, init_val)

    P = u*K*v.flatten()

    return P, np.sum(P*C)


def sinkhorn_log_stabilized(a, b, C, ϵ, ρ, n_iters):
    """ Sinkhorn iteration for (unbalanced) transport in log domain

        Reference
            - https://arxiv.org/pdf/1610.06519.pdf
            - https://arxiv.org/pdf/1506.05439.pdf
                Learning with Wasserstein Loss
    """

    n, m = C.shape
    a = a.flatten()
    b = b.flatten()
    λ = ρ / (ρ + ϵ)

    def S(f, g):
        """ Sᵢⱼ = Cᵢⱼ - fᵢ - gⱼ """
        return (C - np.expand_dims(f, 1) - np.expand_dims(g, 0))
    
    def body_fn(i, val):
        f, g = val
        f = λ*( -ϵ*logsumexp(-S(f,g)/ϵ, axis=1) + f + ϵ*np.log(a) )
        g = λ*( -ϵ*logsumexp(-S(f,g)/ϵ, axis=0) + g + ϵ*np.log(b) )
        return f, g
    
    init_val = (np.ones((n,))/n, np.ones((m,))/m)
    f, g = jax.lax.fori_loop(0, n_iters, body_fn, init_val)

    P = np.exp(-S(f, g)/ϵ)

    # Ignore H(P) = KL(P||α⊗β) + const since
    #     - numerically unstable due to taking logs
    #     - gradient wrt a,b,x,y does not depend on the regularizer term
    return P, np.sum(P*C)


@partial(jax.jit, static_argnums=(4, 5,))
def sinkhorn_divergence(a, b, x, y, c, sink):
    """ S(ϵ,α,β) = OT(ϵ,α,β) - .5*OT(ϵ,α,α) - .5*OT(ϵ,β,β)
            - S is differentiabl wrt a,b,x,y !

        Reference
            - https://arxiv.org/pdf/1810.08278.pdf
            - https://arxiv.org/pdf/1706.00292.pdf
    """
    Pxy, Lxy = sink(a, b, c(x, y))
    Pxx, Lxx = sink(a, a, c(x, x))
    Pyy, Lyy = sink(b, b, c(y, y))
    return Pxy, Lxy - .5*(Lxx + Lyy)