import jax
import jax.numpy as np

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
        D = -2*np.tensordot(X, Y, axes=(-1,-1))
        D += Xsqnorm + Ysqnorm.T
        D = np.maximum(D, 0)
    return D

# Taken from https://github.com/IPL-UV/jaxkern

def l1_distance(x, y):
    return np.sum(np.abs(x-y))

def sqeuclidean_distance(x, y):
    return np.sum((x - y) ** 2)

def euclidean_distance(x, y):
    return np.sqrt(sqeuclidean_distance(x, y))

def distmat(func, x, y):
    if y == None: y = x
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)

def cdist_sqeuclidean(x, y):
    """ Squared euclidean distance matrix """
    return distmat(sqeuclidean_distance, x, y)

def cdist_euclidean(x, y):
    """ euclidean distance matrix """
    return distmat(euclidean_distance, x, y)

def cdist_l1(x, y):
    """ l1 distance matrix """
    return distmat(l1_distance, x, y)

def rbf_kernel(X, Y, gamma=1.):
    """Radial Basis Function Kernel
        
            k(x,y)=exp(-\gamma*||x-y||**2)
                where \gamma   = 1/(2*sigma^2)
                      \sigma^2 = 1/(2*\gamma)
 
        X, Y    (n, d)
        Returns kernel matrix of size (n, n)
    """
    return np.exp(-gamma*cdist_sqeuclidean(X, Y))

def normalize_K(K):
    """ Normalize Kernel Matrix `K`
            http://people.csail.mit.edu/jrennie/writing/normalizeKernel.pdf
        
           K̃ᵢⱼ = Kᵢⱼ/sqrt(Kᵢᵢ Kⱼⱼ)
    """
    k = 1/np.sqrt(np.diag(K))
    k = k.reshape(-1,1)
    K = K*(k@k.T)
    return K

def LookupKernel(X, Y, A):
    return distmat(lambda x, y: A[x, y], X, Y)

def mtgp_k(XT, XTp, logℓ, logL, logv):
    X, Xp = XT[:,0], XTp[:,0]
    T, Tp = np.asarray(XT[:,1], np.int32), np.asarray(XTp[:,1], np.int32)
    
    Kx = cov_se(X, Xp, logℓ=logℓ)
    
    L = np.exp(logL)
    v = np.exp(logv)
    B = L@L.T + np.diag(v)
    Kt = LookupKernel(T, Tp, B)

    K = Kx*Kt
    return K

def cov_se(X, Y=None, logσ=0., logℓ=0.):
    # Squared Exponential kernel 
    #     logσ - log vertical lengthscale
    #     logℓ - log lengthscale
    #
    σ2 = np.exp(2*logσ)
    ℓ2 = np.exp(2*logℓ)
    return σ2*np.exp(-sqdist(X, Y)/2/ℓ2)

def cov_rq(X, Y=None, logσ=0., logℓ=0., logα=0.):
    # Rational Quadratic kernel 
    #     logσ - log vertical lengthscale
    #     logℓ - log lengthscale
    #     logα - log scale mixture
    # 
    σ2 = np.exp(2*logσ)
    ℓ2 = np.exp(2*logℓ)
    α = np.exp(logα)
    return σ2*(cdist_euclidean(X, Y)/2/α/ℓ2 + 1)**(-α)

def cov_pe(X, Y=None, σ=1, p=1, ℓ=1):
    # Periodic kernel (https://www.cs.toronto.edu/~duvenaud/cookbook/)
    #     p - period
    #     ℓ - lengthscale 
    return (σ**2)*np.exp( - 2*np.sin(np.pi*cdist_l1(X, Y)/p)**2/(ℓ**2) )

def linear_kernel(X, Y):
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    return np.dot(X, Y.T)

def estimate_sigma_median(X):
    """Estimate sigma using the median trick
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
    K = k(X, X)
    L = l(Y, Y)
    n = len(K)
    H = np.eye(n) - 1/n
    statistic = np.trace(K@H@L@H) / (n**2)
    return statistic

def cka(X, Y, k, l):
    """ Centered Kernel Alignment 
            https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py
    """ 
    statistic = hsic(X, Y, k, l)
    var1 = np.sqrt(hsic(X, X, k, k))
    var2 = np.sqrt(hsic(Y, Y, l, l))
    statistic = statistic / (var1 * var2)
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