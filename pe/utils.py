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