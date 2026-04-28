import numpy as np


class Neuron:
    """Single neuron with sigmoid activation."""

    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0
        self._last_input = None
        self._last_z = None
        self._last_a = None

    def forward(self, X):
        """Forward pass: z = w·x + b, a = σ(z)"""
        self._last_input = X
        self._last_z = X @ self.weights + self.bias
        self._last_a = self.sigmoid(self._last_z)
        return self._last_a
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid: σ'(z) = σ(z)(1 - σ(z))"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def backward(self, dL_da):
        """
        Backpropagation: Chain rule in action!
        dL/dw = dL/da * da/dz * dz/dw
        """
        da_dz = self.sigmoid_derivative(self._last_z)
        dL_dz = dL_da * da_dz
        dL_dw = self._last_input.T @ dL_dz if self._last_input.ndim > 1 else self._last_input * dL_dz
        dL_db = np.sum(dL_dz)
        return dL_dw, dL_db


if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([0.5, 0.3])
    y = 1.0

    neuron = Neuron(n_inputs=2)
    prediction = neuron.forward(X)
    print(f"Prediction: {prediction:.4f}")

    loss = (prediction - y) ** 2
    print(f"Loss: {loss:.4f}")

    dL_da = 2 * (prediction - y)
    dL_dw, dL_db = neuron.backward(dL_da)
    print(f"Gradients - dL/dw: {dL_dw}, dL/db: {dL_db:.4f}")