import numpy as np

def mmd(X, Y, kernel):
    """ Computes unbiased estimate of MMD in quadratic time """
    if X.ndim == 1:
        X = X.reshape(-1,1)
    if Y.ndim == 1:
        Y = Y.reshape(-1,1)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError('data X,Y should be 2-dim matrix')

    K_XX = kernel(X, X)
    K_XY = kernel(X, Y)
    K_YY = kernel(Y, Y)

    n, m = len(K_XX), len(K_YY)

    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    mmd = np.sum(K_XX)/(n*(n-1))  + np.sum(K_YY)/(m*(m-1))  - 2*np.sum(K_XY)/(n*m)

    return mmd
