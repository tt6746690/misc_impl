from functools import partial

import jax
import jax.numpy as np
from jax.scipy.special import logsumexp


@partial(jax.jit, static_argnums=(4,))
def sinkhorn_knopp(a, b, C, ϵ, n_iters):
    """Matrix scaling for entropic regularized optimal transport

        Reference
            - https://arxiv.org/pdf/1306.0895.pdf
    """

    n, m = C.shape
    a = np.reshape(a, (n, -1))
    b = np.reshape(b, (m, -1))
    K = np.exp(-C/ϵ)

    def body_fn(i, val):
        u, v = val
        u = a/(K  @v)
        v = b/(K.T@u)
        return u, v
    
    init_val = (np.ones((n,1))/n, np.ones((m,1))/m)
    u, v = jax.lax.fori_loop(0, n_iters, body_fn, init_val)

    P = u*K*v.flatten()

    return P


@partial(jax.jit, static_argnums=(3,4,))
def sinkhorn_log_stabilized(a, b, C, ϵ, n_iters):

    n, m = C.shape
    a = a.flatten()
    b = b.flatten()

    def S(f, g):
        """ Sᵢⱼ = Cᵢⱼ - fᵢ - gⱼ """
        return (C - np.expand_dims(f, 1) - np.expand_dims(g, 0))
    
    def body_fn(i, val):
        f, g = val
        f = -ϵ*logsumexp(-S(f,g)/ϵ, axis=1) + f + ϵ*np.log(a)
        g = -ϵ*logsumexp(-S(f,g)/ϵ, axis=0) + g + ϵ*np.log(b)
        return f, g
    
    init_val = (np.ones((n,))/n, np.ones((m,))/m)
    f, g = jax.lax.fori_loop(0, n_iters, body_fn, init_val)

    P = np.exp(-S(f, g)/ϵ)
    return P
