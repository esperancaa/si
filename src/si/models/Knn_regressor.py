

from typing import Callable, Union

import numpy as np
from model_selection.split import stratified_train_test_split

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
   .
    
    Parameters:
    -----------
    
    k: int
        The number of neighbors to consider for prediction
    
    distance: str, default='euclidean'
        The distance metric to use for finding the nearest neighbors
    
    Returns:
    -----------
    
    KNNRegressor object
    """
    
    def __init__(self, k, distance, **kwargs):
        
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset= None
    
    def _fit(self, dataset: Dataset):
        """
        Store the training dataset for later use in predictions.
        
        Parameters:
        - dataset: dict - a dictionary containing 'data' (features) and 'target' (labels/values).
        
        Returns:
        - self: the fitted model.
        
        """
        
        
        self.dataset= dataset
        
        return self
    
    def _predict(self, dataset) -> np.ndarray:
        """
        Predict the values for the test dataset based on the k-nearest neighbors.
        
        Parameters:
        - dataset: dict - a dictionary containing 'data' (features) for the test set.
        
        Returns:
        - predictions: np.array - an array of predicted values for the testing dataset.
        """
        
        