from unittest import TestCase
import numpy as np

from Exercises.Ex2 import Dataset  

class TestRemoveByIndex(TestCase):
    def setUp(self):
        """
        Creates instances of Dataset to test the remove_by_index function.
        """
        
        self.dataset = Dataset(
            X=np.array([[1, 2], [3, 4], [5, 6]]),
            y=np.array([0, 1, 1]),
            features=["Feature1", "Feature2"],
            label="Target"
        )

    def test_remove_valid_index(self):
        """
        Tests removing a valid index.
        """
        new_dataset = self.dataset.remove_by_index(1)

        expected_X = np.array([[1, 2], [5, 6]])
        expected_y = np.array([0, 1])

        np.testing.assert_array_equal(new_dataset.X, expected_X)
        np.testing.assert_array_equal(new_dataset.y, expected_y)



    
