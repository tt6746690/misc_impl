
import numpy as np

def hsic(K_XX, K_YY):
    """ Computes empirical HSIC(X,Y) = (1/n^2)*tr(K_XX,K_YY) """
    if K_XX.ndim != 2 or K_YY.ndim != 2:
        raise ValueError('data K_XX, K_YY should be a matrix')
    if len(K_XX) != len(K_YY):
        raise ValueError('data K_XX, K_YY should be of same size ')

    N = len(K_XX)
    H = np.eye(N) - 1.0/N
    statistic = np.trace(K_XX.dot(H).dot(K_YY.dot(H))) / (N**2)
    return statistic

def hsic_permute(K_XX, K_YY):
    inds_X = np.random.permutation(len(K_XX))
    inds_Y = np.random.permutation(len(K_YY))
    K_XX = K_XX[inds_X, :][:, inds_X]
    K_YY = K_YY[inds_Y, :][:, inds_Y]
    statistic = hsic(K_XX, K_YY)
    return statistic
