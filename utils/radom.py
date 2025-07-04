import random
import numpy as np
import torch
import os


def set_all_seeds(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set PyTorch CUDA random seed if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set deterministic algorithms for better reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)