from functools import partial

import jax
from jax import grad
import jax.numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections  import LineCollection


def GridData(nlines=11, xlim=(0,1), ylim=(0,1)):
    ranges = [ xlim, ylim ]
    np_per_lines = (nlines-1) * 4 + 1
    x_l = [np.linspace(min_r, max_r, nlines      ) for (min_r,max_r) in ranges]
    x_d = [np.linspace(min_r, max_r, np_per_lines) for (min_r,max_r) in ranges]

    v = [] ; c = [] ; i = 0
    for x in x_l[0] :                    # One vertical line per x :
        v += [ [x, y] for y in x_d[1] ]  # Add points to the list of vertices.
        c += [ [i+j,i+j+1] for j in range(np_per_lines-1)] # + appropriate connectivity
        i += np_per_lines
    for y in x_l[1] :                    # One horizontal line per y :
        v += [ [x, y] for x in x_d[1] ]  # Add points to the list of vertices.
        c += [ [i+j,i+j+1] for j in range(np_per_lines-1)] # + appropriate connectivity
        i += np_per_lines

    return ( np.vstack(v), np.vstack(c) ) # (vertices, connectivity)


def plt_grid(ax, X, L, color='lightgrey'):
    """X,L are position, lines of grid 

        ```
        X, L = GridData()
        fig, ax = plt.subplots(figsize=(10,10))
        plt_grid(ax, X, L)
        ```
    """
    segs = line_get_segments(X, L)
    ax.add_collection(LineCollection(
        segs, color=color, linewidths=(2,), linestyle='solid', zorder=0))


def plt_vectorfield(ax, X, V, color='b', scale=1, **kwargs):
    ax.quiver(X[:,0], X[:,1], V[:,0], V[:,1], scale=scale, color=color, **kwargs)


@jax.jit
def line_get_segments(X, L):
    return np.array([[X[l[0]], X[l[1]]] for l in L])


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


def cov_se(X, Y=None, σ2=1., ℓ=1.):
    scal = lambda X: X/ℓ if X is not None else X
    X = scal(X)
    Y = scal(Y)
    return σ2*np.exp(-sqdist(X, Y)/2)


def Hqp(q, p, k):
    """Note we use the same kernel for d = 1,...,D
            i.e. k(qᵢ,qᵢ) I_D \in \R^DxD
    """
    H = k(q,q)*(p@p.T)
    return .5 * np.sum(H)

dq_Hqp = grad(Hqp, argnums=0)
dp_Hqp = grad(Hqp, argnums=1)

@partial(jax.jit, static_argnums=(3,))
def HamiltonianShooting(q, p, g, k, euler_steps=10, δt = .1):
    for t in range(euler_steps):
        q, p, g = [q + δt*dp_Hqp(q,p,k),
                   p - δt*dq_Hqp(q,p,k),
                   g + δt*k(g,q)@p]
        # note we also use the same kernel to interpolate
        # momenum across the dimensions of `p`
    return q, p, g

