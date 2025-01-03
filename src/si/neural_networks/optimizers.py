from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient


class Adam:
    """
    Adam Optimizer
    Combines the benefits of RMSprop and SGD with momentum.

    Parameters:
    - learning_rate: The learning rate for updating weights.
    - beta_1: Exponential decay rate for the 1st moment estimates (default 0.9).
    - beta_2: Exponential decay rate for the 2nd moment estimates (default 0.999).
    - epsilon: A small constant for numerical stability (default 1e-8).
    """

    def __init__(self, learning_rate= 0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        
        super().__init__(learning_rate) #vai buscar o parametro learning rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0  # Time step (epoch)

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update weights using Adam optimization algorithm.

        Parameters:
        - w: Current weights.
        - grad_loss_w: Gradient of the loss function with respect to the weights.

        Returns:
        - Updated weights.
        """
        if self.m is None or self.v is None:
            # Initialize m and v as zeros matrices with the same shape as weights
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        else:
            self.t += 1

            # Update biased first moment estimate
            self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w

            # Update biased second raw moment estimate
            self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m / (1 - self.beta_1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v / (1 - self.beta_2 ** self.t)

            # Update weights
            w_updated = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return w_updated