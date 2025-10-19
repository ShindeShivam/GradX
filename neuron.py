import numpy as np

class Neuron:
    """Single neuron with sigmoid activation."""
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0

    def forward(self, X):
        """Forward pass: z = w·x + b, a = σ(z)"""
        self.X = X
        self.z = X @ self.weights + self.bias
        self.a = self.sigmoid(self.z)
        return self.a
    
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
        da_dz = self.sigmoid_derivative(self.z)  # da/dz
        dL_dz = dL_da * da_dz
        dL_dw = self.X.T @ dL_dz if self.X.ndim > 1 else self.X * dL_dz

        # dL/db = dL/da * da/dz * dz/db = dL/dz * 1
        dL_db = np.sum(dL_dz)
        return dL_dw, dL_db
if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([0.5, 0.3]) # Single training example
    y = 1.0
    neuron = Neuron(n_inputs=2) 
    prediction = neuron.forward(X)
    print(f"Prediction: {prediction:.4f}") 
    loss = (prediction - y) ** 2  # Loss (MSE)
    print(f"Loss: {loss:.4f}")
    # Backward pass
    dL_da = 2 * (prediction - y)  # Derivative of MSE
    dL_dw, dL_db = neuron.backward(dL_da) 
    print(f"Gradients - dL/dw: {dL_dw}, dL/db: {dL_db:.4f}")
    
