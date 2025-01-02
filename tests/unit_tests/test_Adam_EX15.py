from unittest import TestCase
from Exercises.Ex15_Adam import Adam
import numpy as np

class TestAdam(TestCase):
    def setUp(self):
        self.optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.weights = np.array([1.0, 2.0, 3.0])
        self.gradients = np.array([0.1, 0.2, 0.3])

    def test_update_shapes(self):
        updated_weights = self.optimizer.update(self.weights, self.gradients)
        self.assertEqual(updated_weights.shape, self.weights.shape)

    def test_update_values(self):
        # Call update multiple times and check weight updates
        updated_weights_1 = self.optimizer.update(self.weights, self.gradients)
        updated_weights_2 = self.optimizer.update(updated_weights_1, self.gradients)
        self.assertFalse(np.array_equal(updated_weights_1, updated_weights_2))

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            self.optimizer.update(self.weights, np.array([0.1, 0.2]))


