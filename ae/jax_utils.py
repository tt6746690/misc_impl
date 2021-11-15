import jax.numpy as np
from jax import random


def get_data_stream(key, bsz, dataset):
    """ Create data stream
        ```
        n_batches, batches = get_data_stream(key, bsz, X)
        ```
    """
    n = len(dataset[0] if isinstance(dataset, (list, tuple)) else dataset)
    n_complete_batches, leftover = divmod(n, bsz)
    n_batches = n_complete_batches + bool(leftover)

    def data_stream(key):
        while True:
            key, permkey = random.split(key)
            perm = random.permutation(permkey, n)
            for i in range(n_batches):
                ind = perm[i*bsz:(i+1)*bsz]
                if isinstance(dataset, np.ndarray):
                    yield dataset[ind]
                elif isinstance(dataset, (list, tuple)):
                    yield tuple((X[ind] for X in dataset))
                else:
                    # isinstance(dataset, torchvision.datasets.VisionDataset)
                    data = [dataset[i] for i in ind]
                    data_batched = tuple(np.stack(x) for x in list(zip(*data)))
                    yield data_batched

    return n_batches, data_stream(key)