import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import laplace, norm
from scipy.stats import gaussian_kde
from scipy.spatial.distance import squareform, pdist, cdist

def sq_distances(X,Y=None):
    assert(X.ndim==2)
    if Y is None:
        sq_dists = squareform(pdist(X, metric='sqeuclidean'))
    else:
        assert(Y.ndim==2)
        assert(X.shape[1]==Y.shape[1])
        sq_dists = cdist(X, Y, metric='sqeuclidean')
    return sq_dists

def gauss_kernel(X, Y=None, sigma=.5):
    """ Computes the standard Gaussian kernel matrix
            k(x,y)=exp(- ||x-y||**2 / (2 * sigma**2))
    """
    sq_dists = sq_distances(X,Y)
    K = np.exp(-sq_dists / (2 * sigma**2))
    return K

def linear_kernel(X, Y):
    return np.dot(X, Y.T)

def mmd_unbiased(X, Y, kernel):
    assert X.ndim == Y.ndim == 2
    K_XX = kernel(X,X)
    K_XY = kernel(X,Y)
    K_YY = kernel(Y,Y)
    n, m = len(K_XX), len(K_YY)
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX) / (n*(n-1))  + np.sum(K_YY) / (m*(m-1))  - 2*np.sum(K_XY)/(n*m)
    return mmd


N=40000
mu=0.0
sigma2=1
b=np.sqrt(sigma2/2)

# Data
np.random.seed(2)
X = norm.rvs(size=(N,1), loc=mu, scale=np.sqrt(sigma2))
Y = laplace.rvs(size=(N,1), loc=mu, scale=b)

# Plotting
mpl.rcParams['lines.linewidth'] = 3
ht, ncols = 5, 1
fig, axs = plt.subplots(1, ncols)
fig.set_size_inches(ht*ncols+2, ht)
font = {'fontsize': 40, 'fontname': 'Times New Roman'}
cmap = plt.cm.get_cmap('bwr')

ax = axs

grid = np.linspace(-5,5,200).reshape(-1,1)
kernel = lambda X,Y: gauss_kernel(X,Y,0.5)
phi_X = np.mean(kernel(X, grid), axis=0)
phi_Y = np.mean(kernel(Y, grid), axis=0)
witness = (phi_X-phi_Y)*4
ax.plot(grid, witness, label='$\hat{\mu}_{\mathbb{P}} - \hat{\mu}_{\mathbb{Q}}$', c=cmap(0.9))

pdf_normal = norm.pdf(grid, loc=mu, scale=np.sqrt(sigma2))
pdf_laplace = laplace.pdf(grid, loc=mu, scale=b)
ax.plot(grid, pdf_normal, label=f'Normal({mu},{sigma2})', c='k')
ax.plot(grid, pdf_laplace, label=f'Laplace({mu},{b:.1f})', c='k', ls='--')

ax.set_xlim((-5,5))
ax.set_ylim((-.5,.8))
ax.legend()
ax.set_xticks([])
ax.set_yticks([])
    

plt.tight_layout()
save_path = './summary/assets/mmd_gauss_laplace_witness.png'
fig.savefig(save_path, bbox_inches='tight', dpi=100)