from unittest import TestCase
import numpy as np
import sys
sys.path.append("/Users/utilizador/Documents/GitHub/si/src")

from Exercises.Ex3_select_percentile import SelectPercentile
from Exercises.Ex2 import Dataset
from si.base.transformer import Transformer
from si.statistics.f_classification import f_classification

class TestSelectPercentile(TestCase):
    def setUp(self):
        """
        Configura um dataset de exemplo para testes.
        """
        self.X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.y = np.array([0, 1, 1])
        self.features = ['f1', 'f2', 'f3']
        self.label = 'target'
        self.dataset = Dataset(X=self.X, y=self.y, features=self.features, label=self.label)
        self.transformer = SelectPercentile(score_func=f_classification, percentile=50)


    def test_invalid_percentile(self):
        """
        Tests invalid values for the percentile.
        """
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=150)  # Percentil inválido (maior que 100)
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=-10)  # Percentil inválido (menor que 0)

    def test_fit(self):
        """
        Tests the `_fit` method to calculate F-scores and p-values.
        """
        self.transformer._fit(self.dataset)
        self.assertIsNotNone(self.transformer.F)
        self.assertIsNotNone(self.transformer.p)

    def test_transform(self):
        """
        Tests the dataset transformation based on the percentile.
        """
        self.transformer._fit(self.dataset)
        transformed_dataset = self.transformer._transform(self.dataset)

        self.assertLess(transformed_dataset.X.shape[1], self.dataset.X.shape[1])

   


