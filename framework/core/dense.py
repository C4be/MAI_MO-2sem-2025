import numpy as np
from typing import Tuple
from .layer import Layer
from .activations import identity, identity_derivative


class Dense(Layer):
    """
    Полносвязный (Dense) слой.

    Аргументы:
        input_dim: размерность входного вектора,
        output_dim: число нейронов,
        activation: функция активации ('sigmoid', 'relu', 'tanh', 'softmax' или 'identity').
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = "identity"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.bias = np.zeros((1, output_dim))
        self.activation, self.activation_derivative = self._get_activation(activation)
        self.inputs = None
        self.z = None

    def _get_activation(self, activation: str):
        if activation == "sigmoid":
            from .activations import sigmoid, sigmoid_derivative
            return sigmoid, sigmoid_derivative
        
        elif activation == "relu":
            from .activations import relu, relu_derivative
            return relu, relu_derivative
        
        elif activation == "tanh":
            from .activations import tanh, tanh_derivative
            return tanh, tanh_derivative
        
        elif activation == "softmax":
            from .activations import softmax, softmax_derivative
            return softmax, softmax_derivative
        
        elif activation == "identity":
            return identity, identity_derivative
        
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Прямое распространение: вычисление z = XW + b и применение активации.
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        return self.activation(self.z)

    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Обратное распространение: вычисление градиентов, обновление весов и передача градиента на предыдущий слой.
        """
        activation_grad = self.activation_derivative(self.z) * grad
        grad_weights = np.dot(self.inputs.T, activation_grad)
        grad_bias = np.sum(activation_grad, axis=0, keepdims=True)
        grad_input = np.dot(activation_grad, self.weights.T)
        # Обновление параметров
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        return grad_input
