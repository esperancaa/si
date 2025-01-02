from unittest import TestCase
import numpy as np

from Exercises.Ex2 import Dataset  

class TestDropna(TestCase):
    def setUp(self):
        """
        Creates instances of Dataset to test the dropna function..
        """
        
        self.dataset_with_nan = Dataset(
            X=np.array([[1, 2], [3, np.nan], [5, 6]]),
            y=np.array([0, 1, 1]),
            features=["Feature1", "Feature2"],
            label="Target"
        )
        
        
        self.dataset_without_nan = Dataset(
            X=np.array([[1, 2], [3, 4], [5, 6]]),
            y=np.array([0, 1, 1]),
            features=["Feature1", "Feature2"],
            label="Target"
        )

    def test_dropna_with_nan(self):
        """
        Tests the dropna method on a Dataset with NaN values.
        """
        clean_dataset = self.dataset_with_nan.dropna()
        
        expected_X = np.array([[1, 2], [5, 6]])
        expected_y = np.array([0, 1])
        
        np.testing.assert_array_equal(clean_dataset.X, expected_X)
        np.testing.assert_array_equal(clean_dataset.y, expected_y)

    def test_dropna_without_nan(self):
        """
        Tests the dropna method on a Dataset without NaN values.
        """
        clean_dataset = self.dataset_without_nan.dropna()
        
        
        np.testing.assert_array_equal(clean_dataset.X, self.dataset_without_nan.X)
        np.testing.assert_array_equal(clean_dataset.y, self.dataset_without_nan.y)


    def test_fillna_with_value(self):
        """
        Tests the fillna method with the 'value' strategy.
        """
        self.dataset_with_nan.strategy = 'value'
        filled_dataset = self.dataset_with_nan.fillna()

        
        expected_X = np.array([[1, 2], [3, 1], [5, 6]])
        np.testing.assert_array_equal(filled_dataset.X, expected_X)

    def test_fillna_with_mean(self):
        """
        Tests the fillna method with the 'mean' strategy.
        """
        self.dataset_with_nan.strategy = 'mean'
        filled_dataset = self.dataset_with_nan.fillna()

        
        mean_value = np.array([3, (2 + 6) / 2])
        expected_X = np.array([[1, 2], [3, mean_value[1]], [5, 6]])
        np.testing.assert_array_almost_equal(filled_dataset.X, expected_X)

    def test_fillna_with_median(self):
        """
        Tests the fillna method with the 'median' strategy.
        """
        self.dataset_with_nan.strategy = 'median'
        filled_dataset = self.dataset_with_nan.fillna()

        
        median_value = np.array([3, 4])
        expected_X = np.array([[1, 2], [3, median_value[1]], [5, 6]])
        np.testing.assert_array_equal(filled_dataset.X, expected_X)