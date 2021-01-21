from math import sqrt
import torch


def sqdist(X, Y=None):
    """ Returns D where D_ij = ||X_i - Y_j||^2 if Y is not `None`
            https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
        X   (n, d)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y is not None and Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if Y is None:
        D = X@X.T
        Xsqnorm = torch.diag(D).reshape(-1, 1)
        D = - 2*D + Xsqnorm + Xsqnorm.T
        D = torch.clamp(D, min=0)
    else:
        D = torch.cdist(X, Y, p=2)**2
    return D


def rbf_kernel(X, Y=None, sigma=1):
    D = sqdist(X, Y)
    K = torch.exp((-1/(2*(sigma**2)))*D)
    return K


def linear_kernel(X, Y=None):
    if Y is None:
        Y = X
    return X@Y.T


def target_kernel(X, Y=None):
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)
    X = X.reshape(-1, 1)
    K = (X == X.T).to(torch.float32)
    return K


def median_l2dist(X):
    """ median unsquared l2 pairwise distance of X """
    with torch.no_grad():
        D = torch.nn.functional.pdist(X, p=2)
        D = D[torch.nonzero(D, as_tuple=True)]
        if len(D) == 0:
            return 0.
        else:
            return torch.median(D).item()


def median_heuristic(X):
    """Estimate sigma using the median trick
            ν = median(l2.(X))
                with σ = sqrt(ν/2)
            https://arxiv.org/pdf/1707.07269.pdf
                - Simply choose ν = sqrt(median(l2.(X)))
        X    (n, d)
    """
    bandwidth = median_l2dist(X)
    sigma = sqrt(bandwidth/2)
    return sigma


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
        # hsic = torch.trace(KH@LH)/(n**2)
        KH = K - K.mean(1, keepdim=True)
        LH = L - L.mean(1, keepdim=True)
        # tr(K^T@L) = \sum_{ij} K_{ij} L_{ij}
        hsic = torch.sum(KH.T*LH) / (n**2)
    if estimate == 'unbiased':
        K = zero_diagonal(K)
        L = zero_diagonal(L)
        hsic = torch.sum(K.T*L) \
            + torch.sum(K)*torch.sum(L)/(n-1)/(n-2) \
            - torch.sum(K, 1)@torch.sum(L, 1)*2/(n-2)
        hsic = hsic/n/(n-3)
    return hsic


def hsic_unbiased_nograd(X, Y, k, l):
    """ `.fill_diagonal_` not differentiable """
    K = k(X)
    L = l(Y)
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)
    n = len(K)
    o = torch.ones(n).reshape(-1, 1)
    A = torch.sum(K*L)
    B = -(o.T@K@L@o)*2/(n-2)
    C = (o.T@K@o)*(o.T@L@o)/(n-1)/(n-2)
    return A, B, C, (A + B + C)/n/(n-3)


def cka(X, Y, k, l):
    """ Centered Kernel Alignment 
            https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py
            cka = <Kx,Ky>_F / ( ||Kx||_F*||Ky||_F )
                where Kx,Ky are centered
    """
    Kx = k(X)
    Ky = l(Y)
    Kx = centering2(Kx)
    Ky = centering2(Ky)
    normKx = torch.norm(Kx, 'fro')
    normKy = torch.norm(Ky, 'fro')
    cka = torch.sum(Kx*Ky) / (normKx * normKy)
    return cka


def ka(X, Y, k, l):
    """ Kernel Alignment 
            http://papers.neurips.cc/paper/1946-on-kernel-target-alignment.pdf
            cka = <Kx,Ky>_F / ( ||Kx||_F*||Ky||_F )
    """
    Kx = k(X)
    Ky = l(Y)
    normKx = torch.norm(Kx, 'fro')
    normKy = torch.norm(Ky, 'fro')
    ka = torch.sum(Kx*Ky) / (normKx * normKy)
    return ka


def mmd(X, Y, k):
    """ Computes unbiased estimate of MMD^2 in O(n^2)"""
    Kxx = k(X)
    Kxy = k(X, Y)
    Kyy = k(Y)
    n, m = len(Kxx), len(Kyy)
    Kxx = zero_diagonal(Kxx)
    Kyy = zero_diagonal(Kyy)
    mmd = torch.sum(Kxx)/(n*(n-1)) + torch.sum(Kyy) / \
        (m*(m-1)) - 2*torch.sum(Kxy)/(n*m)
    return mmd


def zero_diagonal(K):
    return K - torch.diag(torch.diag(K))


def centering1(K):
    # Slow centering
    # 6.4 s ± 60.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    n = len(K)
    H = torch.eye(n)-1/n
    K = H@K@H
    return K


def centering2(K):
    # Fast centering
    # 36.1 ms ± 750 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    K = K - K.mean(1, keepdim=True)
    K = K - K.mean(0, keepdim=True)
    return K