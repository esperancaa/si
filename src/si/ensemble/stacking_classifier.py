import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier:
    def __init__(self, models, final_model):
        """
        Initializes the StackingClassifier with an initial set of models and a final model.

        Parameters:
        - models: list of Model
            The initial set of models to generate predictions.
        - final_model: Model
            The final model that will make the final predictions based on the outputs of the initial models.
        """
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset):
        """
        Trains the ensemble models and the final model.

        Parameters:
        - dataset: Dataset
            The dataset containing the features (X) and labels (y).

        Returns:
        - self: StackingClassifier
            The trained StackingClassifier.
        """
        # Train the initial set of models
        predictions = []
        for model in self.models:
            model._fit(dataset)
            predictions.append(model._predict(dataset))
        
        # Combine predictions into a single dataset for the final model
        predictions = np.column_stack(predictions)
        final_dataset = Dataset(predictions, dataset.y)
        
        # Train the final model
        self.final_model._fit(final_dataset)
        
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the labels using the ensemble models and the final model.

        Parameters:
        - dataset: Dataset
            The dataset containing the features (X).

        Returns:
        - predictions: np.ndarray
            The predicted labels.
        """
        # Get predictions from the initial models
        predictions = []
        for model in self.models:
            predictions.append(model._predict(dataset))
        
        # Combine predictions into a single dataset for the final model
        predictions = np.column_stack(predictions)
        
        # Predict using the final model
        final_dataset = Dataset(predictions, None)
        return self.final_model._predict(final_dataset)

    def _score(self, dataset: Dataset) -> float:
        """
        Computes the accuracy between predicted and real labels.

        Parameters:
        - dataset: Dataset
            The dataset containing the features (X) and true labels (y).

        Returns:
        - accuracy: float
            The accuracy score of the model.
        """
        predictions = self._predict(dataset)
        return accuracy(dataset.y, predictions)