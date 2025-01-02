from unittest import TestCase
from Exercises.Ex10_StackingClassifier import StackingClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from sklearn.datasets import make_classification
from Exercises.Ex2 import Dataset


class TestStackingClassifier(TestCase):
    def setUp(self):
        # Create a synthetic dataset
        self.X, self.y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.dataset = Dataset(self.X, self.y)  # Presumindo que você tenha uma classe Dataset
        # Define base models and final model
        self.models = [DecisionTreeClassifier(max_depth=3, random_state=42) for _ in range(3)]
        self.final_model = LogisticRegression()
        self.clf = StackingClassifier(models=self.models, final_model=self.final_model)

    def test_fit(self):
        # Test if _fit method works without errors
        self.clf._fit(self.dataset)
        for model in self.models:
            self.assertTrue(hasattr(model, "tree_"))  # Verifica se o DecisionTreeClassifier foi ajustado

    def test_predict(self):
        # Test _predict method
        self.clf._fit(self.dataset)
        predictions = self.clf._predict(self.dataset)
        self.assertEqual(len(predictions), len(self.y))

    def test_score(self):
        # Test _score method
        self.clf._fit(self.dataset)
        score = self.clf._score(self.dataset)
        self.assertTrue(0 <= score <= 1)  # A precisão deve estar entre 0 e 1
