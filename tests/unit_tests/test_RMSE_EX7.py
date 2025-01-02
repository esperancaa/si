from unittest import TestCase
import numpy as np
from Exercises.Ex7_RMSE import RMSE

class TestRMSE(TestCase):

    def test_perfect_prediction(self):
        """Test case where predictions are exactly the same as true values."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        self.assertAlmostEqual(RMSE(y_true, y_pred), 0.0, places=5)

    def test_constant_difference(self):
        """Test case where predictions differ from true values by a constant amount."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        self.assertAlmostEqual(RMSE(y_true, y_pred), 1.0, places=5)

    def test_random_values(self):
        """Test case with random values."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 1.9, 3.2])
        expected_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        self.assertAlmostEqual(RMSE(y_true, y_pred), expected_rmse, places=5)


