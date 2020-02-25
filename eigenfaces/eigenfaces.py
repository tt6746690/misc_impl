import numpy as np

def pca_with_eig(X, k=None):
    N = X.shape[0]
    X = X - np.mean(X,axis=0)
    C = 1/(N-1)*X.T@X
    w, P = np.linalg.eig(C)
    I = np.argsort(-w)
    P = P[:,I]
    if k:
        assert(k <= P.shape[1])
        P = P[:,:k]
    Y = X@P
    return Y, P

def pca_with_svd(X, k=None):
    N = X.shape[0]
    X = X - np.mean(X,axis=0)
    W = X/(N-1)
    _, _, VT = np.linalg.svd(W)
    P = VT.T
    if k:
        assert(k <= P.shape[1])
        P = P[:,:k]
    Y = X@P
    return Y, P


if __name__ == '__main__':

    import os
    os.makedirs('./assets', exist_ok=True)

    import math
    import matplotlib.pyplot as plt
    import torch
    import torchvision
    # labelled faces in the wild: http://vis-www.cs.umass.edu/lfw/
    from sklearn.datasets import fetch_lfw_people

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pc", type=int, dest='n_pc', default=150)
    parser.add_argument("--n_save", type=int, dest='n_save', default=40)
    args = parser.parse_args() 
    globals().update(vars(args))

    n_save = min([n_save, n_pc])

    lfw_people = fetch_lfw_people(
        data_home='./data',
        min_faces_per_person=70,
        resize=0.4)
    n_samples, h, w = lfw_people.images.shape

    X = lfw_people.data
    Xmean = np.mean(X, axis=0)
    Y, P = pca_with_svd(X, n_pc)

    eigenfaces = np.moveaxis(P, 0, 1).reshape((n_pc, 1, h, w))
    torchvision.utils.save_image(
        torch.tensor(eigenfaces)[:n_save],
        './assets/eigenfaces.png',
        padding=1,
        normalize=True)

    torchvision.utils.save_image(
        torch.tensor(X[:n_save,:].reshape(-1,1,h,w)),
        f'./assets/X.png',
        padding=1,
        normalize=True)


    for n_pc in [5, 50, 100, 500]:
        Y, P = pca_with_svd(X, n_pc)
        recon = Y[:n_save,:] @ P.T + Xmean
        torchvision.utils.save_image(
            torch.tensor(recon.reshape((-1, 1, h, w))),
            f'./assets/Yhat_npc={n_pc}.png',
            padding=1,
            normalize=True)