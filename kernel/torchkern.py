import torch


def sqdist(X, Y=None):
    """ Returns D where D_ij = ||X_i - Y_j||^2 if Y is not `None`
            https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf

        X   (n, d)
    """
    if Y is None:
        D = X@X.T
        Xsqnorm = torch.diag(D).reshape(-1, 1)
        D = Xsqnorm + Xsqnorm.T - 2*D
    else:
        D = torch.cdist(X, Y, p=2)**2
    return D


def rbf_kernel(X, Y=None, sigma=1):
    D = sqdist(X, Y)
    K = torch.exp((-1/(2*sigma**2))*D)
    return K


def linear_kernel(X, Y=None):
    if Y is None:
        Y = X
    return X@Y.T


def median_heuristic(X):
    """Estimate sigma using the median trick
            bandwidth = median(l2dist.([X,Y]))
                with lengthscale = \sqrt(bandwidth/2)

        X    (n, d)
    """
    D = torch.nn.functional.pdist(X, p=2)
    D = D[torch.nonzero(D, as_tuple=True)]
    bandwidth = torch.median(D)
    legnthscale = torch.sqrt(bandwidth/2)
    return legnthscale


def hsic(X, Y, k, l, estimate='biased'):
    """ Computes empirical HSIC
    """
    if estimate not in ['biased', 'unbiased']:
        raise ValueError('estimate not \in [biased, unbiased]')

    K = k(X)
    L = l(Y)
    n = len(K)
    if estimate == 'biased':
        # # Same as below but without creating new array on gpu
        # H = torch.eye(n) - 1/n; KH = K@H; LH = L@H
        KH = K - K.mean(1, keepdim=True)
        LH = L - L.mean(1, keepdim=True)
        hsic = torch.sum((KH)*(LH)) / (n**2)
    if estimate == 'unbiased':
        K = zero_diagonal(K)
        L = zero_diagonal(L)
        hsic = torch.sum(K*L) \
            + torch.sum(K)*torch.sum(L)/(n-1)/(n-2) \
            - torch.sum(K, 1)@torch.sum(L, 1)*2/(n-2)
        hsic = hsic/n/(n-3)
    return hsic


def cka(X, Y, k, l):
    """ Centered Kernel Alignment 
            https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py

            cka = <Kx,Ky>_F / ( ||Kx||_F*||Ky||_F )
    """
    Kx = k(X)
    Ky = l(Y)
    normKx = torch.norm(Kx, 'fro')
    normKy = torch.norm(Ky, 'fro')
    cka = torch.sum(Kx*Ky) / (normKx * normKy)
    return cka


def mmd(X, Y, k):
    """ Computes unbiased estimate of MMD^2 in O(n^2)"""
    Kxx = k(X, X)
    Kxy = k(X, Y)
    Kyy = k(Y, Y)
    n, m = len(Kxx), len(Kyy)
    Kxx = zero_diagonal(Kxx)
    Kyy = zero_diagonal(Kyy)
    mmd = torch.sum(Kxx)/(n*(n-1)) + torch.sum(Kyy) / \
        (m*(m-1)) - 2*torch.sum(Kxy)/(n*m)
    return mmd


def zero_diagonal(K):
    return K - torch.diag(torch.diag(K))
