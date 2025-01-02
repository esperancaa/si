from unittest import TestCase
import numpy as np
from Exercises.Ex13_SoftmaxActivation import SoftmaxActivation

class TestSoftmaxActivation(TestCase):
    def setUp(self):
        self.softmax = SoftmaxActivation()
        self.input_data = np.array([[1.0, 2.0, 3.0], [1.0, -1.0, 0.0]])
        self.output_error = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    def test_forward_propagation(self):
        output = self.softmax.forward_propagation(self.input_data)
        self.assertEqual(output.shape, self.input_data.shape)
        np.testing.assert_almost_equal(np.sum(output, axis=1), 1.0)

    def test_forward_propagation_unidimensional(self):
        input_data = np.array([1.0, 2.0, 3.0])
        output = self.softmax.forward_propagation(input_data)
        self.assertEqual(output.shape, (1, 3))
        np.testing.assert_almost_equal(np.sum(output, axis=1), 1.0)

    def test_backward_propagation(self):
        self.softmax.forward_propagation(self.input_data)
        backward_output = self.softmax.backward_propagation(self.output_error)
        self.assertEqual(backward_output.shape, self.input_data.shape)

    def test_output_shape(self):
        self.softmax.forward_propagation(self.input_data)
        self.assertEqual(self.softmax.output_shape(), self.input_data.shape)

    def test_error_without_forward(self):
        with self.assertRaises(ValueError):
            self.softmax.output_shape()


