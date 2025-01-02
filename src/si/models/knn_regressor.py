import numpy as np

from src.si.statistics.euclidean_distance import euclidean_distance
from Exercises.Ex7_RMSE import RMSE


class KNNRegressor:
    """
    K-Nearest Neighbors Regressor class for regression problems.

    Parameters:
    -----------
    k: int
        The number of nearest examples to consider.

    distance: callable
        A function that calculates the distance between a sample and the samples in the training dataset.
    
    Attributes:
    -----------
    dataset: tuple (X, y)
        Stores the training dataset, where X is the feature matrix and y is the target vector.
    """

    def __init__(self, k, distance):
        self.k = k
        self.distance = distance if distance is not None else euclidean_distance
        self.dataset = None

    def _fit(self, dataset):
        """
        Store the training dataset.

        Parameters:
        -----------
        dataset: tuple (X, y)
            The training dataset, where X is the feature matrix and y is the target vector.

        Returns:
        --------
        self: KNNRegressor
            The fitted regressor.
        """
        self.dataset = dataset
        return self

    def _predict(self, dataset):
        """
        Predict values for the test dataset.

        Parameters:
        -----------
        dataset: numpy array
            The test dataset feature matrix (X).

        Returns:
        --------
        predictions: numpy array
            The predicted values (y_pred) for the test dataset.
        """
        X_train, y_train = self.dataset
        X_test = dataset
        predictions = []

        for x in X_test:
            
            distances = np.array([self.distance(x, x_train) for x_train in X_train]) #distância entre o X e todos os training samples
            
            nearest_indexes = distances.argsort()[:self.k] #index do k nearest neighbors
            
            nearest_values = y_train[nearest_indexes]  #valores correpondentes 
            
            predictions.append(np.mean(nearest_values)) #média dos predicted values

        return np.array(predictions)

    def _score(self, dataset):
        """
        Calculate the RMSE between predictions and actual values.

        Parameters:
        -----------
        dataset: tuple (X, y)
            The test dataset, where X is the feature matrix and y is the target vector.

        Returns:
        --------
        error: float
            The RMSE between predictions and actual values.
        """
        X_test, y_test = dataset
        y_pred = self._predict(X_test)
        error = RMSE(y_test, y_pred) 
        return error