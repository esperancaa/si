from unittest import TestCase
import numpy as np
from si.neural_networks.losses import LossFunction
from Exercises.Ex14_CategoricalCrossEntropy import CategoricalCrossEntropy

class TestCategoricalCrossEntropy(TestCase):
    def setUp(self):
        self.loss = CategoricalCrossEntropy()
        self.y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # One-hot encoded labels
        self.y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])  # Predicted probabilities

    def test_forward(self):
        loss = self.loss.forward(self.y_true, self.y_pred)
        self.assertGreater(loss, 0)
        self.assertAlmostEqual(loss, 0.2798, places=4)  # Valor esperado calculado previamente

    def test_backward(self):
        grad = self.loss.backward(self.y_true, self.y_pred)
        self.assertEqual(grad.shape, self.y_true.shape)
        self.assertTrue(np.all(grad <= 0))  # Gradiente deve ser <= 0

    def test_invalid_shapes(self):
        y_pred_invalid = np.array([0.9, 0.1])  # Formato inválido
        with self.assertRaises(ValueError):
            self.loss.forward(self.y_true, y_pred_invalid)

   

