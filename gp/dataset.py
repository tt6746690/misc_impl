import numpy as np

import torch
import torchvision


def load_rectangles(n_train=50, n_test=100, image_shape=(14, 14), seed=123):
    """ Load triangles, labels=1 if vertically placed 

            X_train    (n_train, image_shape, 1)
            Y_train    (n_train, 1)
            X_test     (n_test,  image_shape, 1)
            Y_test     (n_test,  1)
    """

    def make_rectangle(arr, x0, y0, x1, y1):
        arr[y0:y1, x0] = 1
        arr[y0:y1, x1] = 1
        arr[y0, x0:x1] = 1
        arr[y1, x0: x1 + 1] = 1

    def make_random_rectangle(arr):
        x0 = np.random.randint(1, arr.shape[1] - 3)
        y0 = np.random.randint(1, arr.shape[0] - 3)
        x1 = np.random.randint(x0 + 2, arr.shape[1] - 1)
        y1 = np.random.randint(y0 + 2, arr.shape[0] - 1)
        make_rectangle(arr, x0, y0, x1, y1)
        return x0, y0, x1, y1

    def make_rectangles_dataset(num, w, h):
        d, Y = np.zeros((num, h, w)), np.zeros((num, 1))
        for i, img in enumerate(d):
            for j in range(1000):  # Finite number of tries
                x0, y0, x1, y1 = make_random_rectangle(img)
                rw, rh = y1 - y0, x1 - x0
                if rw == rh:
                    img[:, :] = 0
                    continue
                Y[i, 0] = rw > rh
                break
        return (d.reshape(num, h, w, 1).astype(np.float32),
                Y.astype(np.float32))

    np.random.seed(seed)
    X_train, Y_train = make_rectangles_dataset(n_train, *image_shape)
    X_test, Y_test = make_rectangles_dataset(n_test, *image_shape)

    return X_train, Y_train, X_test, Y_test


def load_mnist():
    """MNist without data augmentation 

            X_train    (60000, 28, 28, 1)
            Y_train    (60000, 1)
            X_test     (60000, 28, 28, 1)
            Y_test     (60000, 1)
    """
    new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
    torchvision.datasets.MNIST.resources = [
        ('/'.join([new_mirror, url.split('/')[-1]]), md5)
        for url, md5 in torchvision.datasets.MNIST.resources
    ]

    train_dataset = torchvision.datasets.MNIST(
        './data', train=True,  download=True)
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, download=True)

    def proc_X(X):
        X = np.asarray(X.to(torch.float32))
        X = X[..., np.newaxis] / 255.
        return X

    def proc_Y(Y):
        Y = np.asarray(Y.to(torch.int32))
        Y = Y[..., np.newaxis]
        return Y

    X_train = proc_X(train_dataset.data)
    Y_train = proc_Y(train_dataset.targets)
    X_test = proc_X(test_dataset.data)
    Y_test = proc_Y(test_dataset.targets)

    return X_train, Y_train, X_test, Y_test


def load_cifar10():
    """ Load CIFAR10

            X_train    (50000, 32, 32, 1)
            Y_train    (50000, 1)
            X_test     (50000, 32, 32, 1)
            Y_test     (50000, 1)
    """
    train_dataset = torchvision.datasets.CIFAR10(
        './data', train=True,  download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        './data', train=False, download=True)

    def proc_X(X):
        X = X / 255.
        return X

    def proc_Y(Y):
        Y = np.asarray(Y)
        Y = Y[..., np.newaxis]
        return Y

    X_train = proc_X(train_dataset.data)
    Y_train = proc_Y(train_dataset.targets)
    X_test = proc_X(test_dataset.data)
    Y_test = proc_Y(test_dataset.targets)

    return X_train, Y_train, X_test, Y_test



def XY_subset(X, Y, Y_subset):
    """ Take subsets of `X,Y` with `Y\in Y_subset` 
            and remap values of `Y` to [0,...,len(Y_subset)]
    """
    ind = np.any(np.stack([(Y==y) for y in Y_subset]), axis=0).squeeze()
    X, Y = X[ind], Y[ind]

    # to_new_label[y] \in { 0,...,len(Y_subset) }
    to_new_label = np.zeros((10,), dtype=np.int64)
    to_new_label[Y_subset] = np.arange(len(Y_subset), dtype=np.int64)
    Y = to_new_label[Y]
    
    return X, Y


def get_dataset(dataset, Y_subset=None):
    
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist()
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10()
        
    if Y_subset is not None:
        X_train, y_train = XY_subset(X_train, y_train, Y_subset)
        X_test, y_test = XY_subset(X_test, y_test, Y_subset)

    return X_train, y_train, X_test, y_test