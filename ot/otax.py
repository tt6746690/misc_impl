
from functools import partial

import jax
import jax.numpy as np


@partial(jax.jit, static_argnums=(4,))
def sinkhorn_knopp(a, b, C, ϵ, n_iters):
    """Matrix scaling for entropic regularized optimal transport

        Reference
            - https://arxiv.org/pdf/1306.0895.pdf
    """
    if (np.sum(a) != 1) or (np.sum(b) != 1):
        raise ValueError('α, β must lie in simplex')

    n, m = C.shape
    a = np.reshape(a, (n, -1))
    b = np.reshape(b, (m, -1))
    u = np.ones((n,1))/n
    v = np.ones((m,1))/m
    K = np.exp(-C/ϵ)

    for it in range(n_iters):
        u = a/(K@v)
        v = b/(K.T@u)

    P = u*K*v.flatten()
    cost = np.sum(P*C)

    return u,v,cost,P


def lse(x, axis=None, keepdims=False):
    """y = c + logΣᵢexp(xᵢ-c) where c = max(x) 
        Reference
            - https://github.com/scipy/scipy/blob/v1.6.3/scipy/special/_logsumexp.py#L7-L127
    """
    c = np.amax(x, axis=axis, keepdims=True)
    y = np.log(np.sum(np.exp(x - c), axis=axis, keepdims=keepdims))
    if not keepdims: c = np.squeeze(c, axis=axis)
    y += c
    return y

