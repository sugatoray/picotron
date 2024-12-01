import torch
import random
import numpy as np
import builtins
import fcntl

def print(*args, is_print_rank=True, **kwargs):
    """ solves multi-process interleaved print problem """
    if not is_print_rank: return
    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

def set_all_seed(seed):
    for module in [random, np.random]: module.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    
def to_readable_format(num, precision=2):
    if num >= 1e12:
        return f"{num / 1e12:.{precision}f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"
    
def assert_no_meta_tensors(model):
    meta_tensors = []
    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            meta_tensors.append(f"Parameter '{name}' with shape {param.shape}")
    
    for name, buffer in model.named_buffers():
        if buffer.device == torch.device("meta"):
            meta_tensors.append(f"Buffer '{name}' with shape {buffer.shape}")
    
    assert len(meta_tensors) == 0, f"Found {len(meta_tensors)} meta tensors:\n" + "\n".join(meta_tensors)