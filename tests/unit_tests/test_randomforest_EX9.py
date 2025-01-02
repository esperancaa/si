from unittest import TestCase
from Exercises.Ex9_RandomForest_Classifier import RandomForestClassifier
from Exercises.Ex2 import Dataset
import numpy as np

class TestRandomForestClassifier(TestCase):
    def setUp(self):
        np.random.seed(42)
        
        self.X = np.random.rand(100, 10)
        self.y = np.random.choice([0, 1], size=100)
        self.dataset = Dataset(self.X, self.y)
        self.model = RandomForestClassifier(n_estimators=10, min_samples_split=3, max_features=3, max_depth=5, mode='gini' seed=42)

    def test_fit(self):
        self.model.fit(self.dataset)
        self.assertEqual(len(self.model.trees), 10, "The number of trees should match n_estimators")

    def test_predict(self):
        self.model.fit(self.dataset)
        predictions = self.model.predict(self.dataset)
        self.assertEqual(len(predictions), len(self.dataset.y), "Number of predictions should match number of samples")

    def test_score(self):
        self.model.fit(self.dataset)
        accuracy = self.model.score(self.dataset)
        self.assertGreaterEqual(accuracy, 0, "Accuracy should be non-negative")
        self.assertLessEqual(accuracy, 1, "Accuracy should not exceed 1")
