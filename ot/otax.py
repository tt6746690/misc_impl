from functools import partial

import jax
import jax.numpy as np
from jax.scipy.special import logsumexp

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def sinkhorn_knopp(a, b, C, ϵ, ρ, n_iters):
    """Matrix scaling for entropic regularized optimal transport

        Reference
            - Sinkhorn Distances: Lightspeed Computations of Optimal Transport Distances
                https://arxiv.org/pdf/1306.0895.pdf
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
    """ Sinkhorn iteration for entropic regularized transport in log domain

        Reference
            - Stabilized Sparse Scaling Algorithms for Entropic Regularized OT
                https://arxiv.org/pdf/1610.06519.pdf
            - Learning with Wasserstein Loss
                https://arxiv.org/pdf/1506.05439.pdf

        ϵ   Coefficient to entropy term,
            Smaller values of ϵ yield sparser transport plan and slower convergence
        ρ   Coefficient to soft marginal regularization term,
            Use a large ρ, e.g. ρ = 1e5, for balanced optimal transport
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
            - Interpolating between Optimal Transport and MMD using Sinkhorn Divergence
                https://arxiv.org/pdf/1810.08278.pdf
            - Learning Generative Models with Sinkhorn Divergences
                https://arxiv.org/pdf/1706.00292.pdf
    """
    Pxy, Lxy = sink(a, b, c(x, y))
    Pxx, Lxx = sink(a, a, c(x, x))
    Pyy, Lyy = sink(b, b, c(y, y))
    return Pxy, Lxy - .5*(Lxx + Lyy)


def plt_transport_plan(ax, π, x, y, thresh=.001, scale=5):
    π = jax.ops.index_update(π, jax.ops.index[np.abs(π)<thresh], 0)
    ind = np.nonzero(π)
    T = np.column_stack(ind)
    width = π[ind]
    width = scale*(width - np.min(width))/(np.max(width)-np.min(width))
    segs = np.stack((x[T[:,0]], y[T[:,1]]), axis=1)
    ax.add_collection(LineCollection(
        segs, color=plt.cm.get_cmap('Pastel1')(3), linewidths=width,
        linestyle='solid', zorder=0))