import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def softmax(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def softmax_derivative(x: np.ndarray) -> np.ndarray:
    # Примечание: производная softmax обычно используется вместе с кросс-энтропией
    s = softmax(x)
    return s * (1 - s)


def identity(x: np.ndarray) -> np.ndarray:
    return x


def identity_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)
