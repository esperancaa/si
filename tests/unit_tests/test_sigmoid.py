from unittest import TestCase

from si.statistics.sigmoid_function import sigmoid_function

class TestSigmoid(TestCase):

    def test_sigmoid(self):

        self.assertEqual(sigmoid_function(0), 0.5)
        