import torch


def get_device():
    """Best available torch device: cuda, mps, else cpu."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
