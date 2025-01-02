from unittest import TestCase
import numpy as np
from Exercises.Ex12_Dropout import Dropout

class TestDropout(TestCase):
    def setUp(self):
        self.dropout = Dropout(probability=0.5)
        self.input = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.output_error = np.array([[0.1, 0.2], [0.3, 0.4]])

    def test_forward_training(self):
        output = self.dropout.forward_propagation(self.input, training=True)
        self.assertEqual(output.shape, self.input.shape)
        self.assertTrue(np.all((output == 0) | (output >= self.input).any()))

    def test_forward_inference(self):
        output = self.dropout.forward_propagation(self.input, training=False)
        np.testing.assert_array_equal(output, self.input)

    def test_backward(self):
        self.dropout.forward_propagation(self.input, training=True)
        backward_output = self.dropout.backward_propagation(self.output_error)
        self.assertEqual(backward_output.shape, self.output_error.shape)

    def test_output_shape(self):
        self.dropout.forward_propagation(self.input)
        self.assertEqual(self.dropout.output_shape(), self.input.shape)

    def test_parameters(self):
        self.assertEqual(self.dropout.parameters, 0)