

import numpy as np
from si.data.dataset import Dataset

def sigmoid_function(x: np.ndarray) -> np.ndarray:
    
    """
    Sigmoid function
    :param x: input value
    :return: sigmoid value
    
    """
    
    
    return 1 / (1 + np.exp(-x))
