import numpy as np


def RMSE(y_true, y_pred):
    """
    calculate the root mean squared error between y_true and y_pred
    
    Parameters:
    -----------
    
    y_true: numpy array
        The true values
        
    y_pred: numpy array
        The predicted values
        
    Returns:
    -----------
    
    rmse: float
        The root mean squared error between y_true and y_pred
        
    """
    RMSE= np.sqrt(np.mean((y_true - y_pred)**2))
    
    return RMSE