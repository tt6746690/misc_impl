import jax
import jax.numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def plt_subplots_1x1_if_not_exists(fig, ax, gridspec_kw=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, gridspec_kw=gridspec_kw)
        fig.set_size_inches(5, 5)
    return fig, ax

def plt_subplots_1x2_if_not_exists(fig, axs, s=5, gridspec_kw=None):
    if fig is None and axs is None:
        fig, axs = plt.subplots(1, 2, gridspec_kw=gridspec_kw)
        fig.set_size_inches(s*2, s)
    return fig, axs


def plt_scaled_colobar_ax(ax):
    """ `fig.colorbar(im, cax=plt_scaled_colobar_ax(ax))` """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return cax

    
def plt_savefig(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=100)


@jax.jit
def cartesian(points):
    """ Barycentric (2-simplex) -> Cartesian (1st quadrant)
    
        points    (N, 3)
        Returns   (N, 2)
        
    e.g. (1,0,0) -> (0,0)
         (0,1,0) -> (0,1)
         (0,0,1) -> (0.5, np.sqrt(3)/2)
    """
    points = np.asarray(points)
    ndim = points.ndim 
    if ndim == 1:
        points = points.reshape((1, points.size))
    d = points.sum(axis=1)  # in case values aren't normalized
    x = 0.5 * (2 * points[:, 1] + points[:, 2]) / d
    y = (np.sqrt(3.0) / 2) * points[:, 2] / d
    out = np.vstack([x, y]).T
    if ndim == 1:
        return out.reshape((2,))
    return out

@jax.jit
def barycentric(points):
    """Inverse of :func:`cartesian`."""
    points = np.asarray(points)
    ndim = points.ndim
    if ndim == 1:
        points = points.reshape((1, points.size))
    c = (2 / np.sqrt(3.0)) * points[:, 1]
    b = (2 * points[:, 0] - c) / 2.0
    a = 1.0 - c - b
    out = np.vstack([a, b, c]).T
    if ndim == 1:
        return out.reshape((3,))
    return out


def plt_2simplex_scatter(ax, points, vertexlabels=None, **kwargs):
    """Scatter plot of barycentric 2-simplex points on a 2D triangle.

    points : (N, 3) shape array
        N points on a 2-simplex
    vertexlabels : (str, str, str)
    """
    if vertexlabels is None:
        vertexlabels = ("1", "2", "3")

    projected = cartesian(points)
    ax.scatter(projected[:, 0], projected[:, 1], **kwargs)
    _draw_axes(ax, vertexlabels)


def plt_2simplex_contour(ax, f, vertexlabels=None, **kwargs):
    """Contour line plot on a 2D triangle of a function evaluated at
            barycentric 2-simplex points.

    f : (N,3) -> (N,)
    vertexlabels : (str, str, str)
        Labels for corners of the plot, in the order
        ``(a, b, c)`` where ``a == (1,0,0)``, ``b == (0,1,0)``,
        ``c == (0,0,1)``.
    """
    _2simplex_contour(ax, f, vertexlabels, ax.tricontour, **kwargs)


def plt_2simplex_contourf(ax, f, vertexlabels=None, **kwargs):
    """Filled contour plot """
    _2simplex_contour(ax, f, vertexlabels, ax.tricontourf, **kwargs)


def _2simplex_contour(ax, f, vertexlabels=None, contourfunc=None, **kwargs):
    if contourfunc is None:
        contourfunc = ax.tricontour
    if vertexlabels is None:
        vertexlabels = ("1", "2", "3")
    if 'levels' not in kwargs:
        levels = 100
    n = 200
    x = np.linspace(0, 1, n)
    y = np.linspace(0, np.sqrt(3.0) / 2.0, n)
    points2d = np.transpose(np.array([np.tile(x, len(y)), np.repeat(y, len(x))]))
    points3d = barycentric(points2d)
    valid = (points3d.sum(axis=1) == 1.0) & ((0.0 <= points3d).all(axis=1))
    points2d = points2d[np.where(valid), :][0]
    points3d = points3d[np.where(valid), :][0]
    z = f(points3d)
    contourfunc(points2d[:, 0], points2d[:, 1], z, **kwargs)
    _draw_axes(ax, vertexlabels)


def _draw_axes(ax, vertexlabels):
    # l1 = mpl.lines.Line2D([0, 0.5, 1.0, 0], [0, np.sqrt(3) / 2, 0, 0],
    #                       color="k", linewidth=1, antialiased=True)
    # ax.add_line(l1)
    ax.text(-0.05, -0.05, vertexlabels[0], ha='right')
    ax.text(1.05, -0.05, vertexlabels[1], ha='left')
    ax.text(0.5, np.sqrt(3) / 2 + 0.05, vertexlabels[2], ha='center')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_aspect("equal")

def plt_2simplex_dirichlet_pdf(ax, α, log_scale=False, **kwargs):
    from jax.scipy.stats import dirichlet
    f = lambda X: np.log(dirichlet.pdf(X.T, α)+1e-10) if log_scale else dirichlet.pdf(X.T, α)
    plt_2simplex_contourf(ax, f, **kwargs)