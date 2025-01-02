
from si.neural_networks.layers import Layer
import numpy as np

class Dropout(Layer):
    def __init__(self, probability: float):
        """
        Initializes the Dropout layer.
        
        Parameters:
        - probability (float): Dropout rate, between 0 and 1.
        """
        if not (0 <= probability <= 1):
            raise ValueError("Dropout probability must be between 0 and 1.")
        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform forward propagation through the dropout layer.
        
        Parameters:
        - input (np.ndarray): The input array.
        - training (bool): Whether we are in training mode or not.
        
        Returns:
        - np.ndarray: The output array after applying dropout (or unchanged input during inference).
        """
        self.input = input
        if training:
            scaling_factor = 1 / (1 - self.probability)
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)
            self.output = input * self.mask * scaling_factor
        else:
            self.output = input
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation through the dropout layer.
        
        Parameters:
        - output_error (np.ndarray): The output error of the layer.
        
        Returns:
        - np.ndarray: The error propagated back through the layer.
        """
        return output_error * self.mask

    def output_shape(self) -> tuple:
        """
        Returns the shape of the input (dropout does not change the shape).
        
        Returns:
        - tuple: The shape of the input.
        """
        return self.input.shape if self.input is not None else None

    @property
    def parameters(self) -> int:
        """
        Returns the number of parameters in the layer.
        
        Returns:
        - int: Always 0 for the dropout layer.
        """
        return 0