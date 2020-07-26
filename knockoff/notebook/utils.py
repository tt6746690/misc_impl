import os
import numpy as np

import torch


def makedirs_exists_ok(path):
    os.makedirs(path, exist_ok=True)

def seed_rng(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def set_cuda_visible_devices(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def load_weights_from_file(model, weight_path):
    if weight_path != '':
        model.load_state_dict(torch.load(weight_path))

def bin_index(x, n_bins):
    """ Finds bin index for `x` in [0, 1] 
        
        x    batch_size, 1
    """
    assert(torch.min(x) >= 0 and torch.max(x) <= 1)
    cdf = torch.arange(n_bins, dtype=torch.float32, device=x.device)
    cdf.add_(1).div_(n_bins)
    mask = (x.view(-1,1).repeat((1, n_bins)) > cdf)
    return torch.sum(mask, dim=1)