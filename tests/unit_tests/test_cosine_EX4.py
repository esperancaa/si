from unittest import TestCase
import numpy as np
from Exercises.Ex4_Cosien_distance import cosine_distance 

class TestCosineDistance(TestCase):
    
    def test_identical_vectors(self):
        """
        Test case where the vectors are identical.
        The cosine distance should be 0.
        """
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        result = cosine_distance(x, y)
        self.assertEqual(result[0], 0.0, {result[0]})
    
    def test_orthogonal_vectors(self):
        """
        Test case where the vectors are orthogonal.
        The cosine distance should be 1.
        """
        x = np.array([1, 0])
        y = np.array([0, 1])
        result = cosine_distance(x, y)
        self.assertEqual(result[0], 1.0, {result[0]})
    
   
    
   
