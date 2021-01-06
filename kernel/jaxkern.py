import jax
import jax.numpy as np

# Taken from https://github.com/IPL-UV/jaxkern

def sqeuclidean_distance(x, y):
    return np.sum((x - y) ** 2)

def euclidean_distance(x, y):
    return np.sqrt(sqeuclidean_distance(x, y))

def distmat(func, x, y):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)

def cdist_sqeuclidean(x, y):
    """ Squared euclidean distance matrix """
    return distmat(sqeuclidean_distance, x, y)

def cdist_euclidean(x, y):
    """ Squared euclidean distance matrix """
    return distmat(euclidean_distance, x, y)

def rbf_kernel(X, Y, gamma=1.):
    """Radial Basis Function Kernel
        
            k(x,y)=exp(-\gamma*||x-y||**2)
                where \gamma   = 1/(2*sigma^2)
                      \sigma^2 = 1/(2*\gamma)
 
        X, Y    (n, d)
        Returns kernel matrix of size (n, n)
    """
    return np.exp(-gamma*cdist_sqeuclidean(X, Y))

def linear_kernel(X, Y):
    return np.dot(X, Y.T)

def estimate_sigma_median(X):
    """Estimate sigma using the median heuristic
            bandwidth = median(l2dist.([X,Y]))
                with \sigma = \sqrt(bandwidth/2)
        
        X, Y    (n, d)
    """
    D = cdist_euclidean(X, X)
    D = D[np.nonzero(D)]
    bandwidth = np.median(D)
    sigma = np.sqrt(bandwidth/2)
    return sigma

def hsic(X, Y, k, l):
    """ Computes empirical HSIC = tr(KHLH)
            where H is the centering matrix
    """
    K = k(X, Y)
    L = l(X, Y)
    m = len(K)
    H = np.eye(m) - 1/m
    statistic = np.trace(K@H@L@H) / (m**2)
    return statistic

def jax_fill_diagonal(A, v):
    return jax.ops.index_update(A, np.diag_indices(A.shape[0]), v)

def mmd(X, Y, k):
    """ Computes unbiased estimate of MMD in O(n^2)"""
    Kxx = k(X,X)
    Kxy = k(X,Y)
    Kyy = k(Y,Y)
    n, m = len(Kxx), len(Kyy)
    jax_fill_diagonal(Kxx, 0)
    jax_fill_diagonal(Kyy, 0)
    mmd = np.sum(Kxx)/(n*(n-1)) + np.sum(Kyy)/(m*(m-1)) - 2*np.sum(Kxy)/(n*m)
    return mmd

def cosine_sim(x, y):
    inner = x.T@y
    xnorm = np.sqrt(x.T@x)
    ynorm = np.sqrt(y.T@y)
    cos = inner / (xnorm*ynorm)
    return cos.squeeze()