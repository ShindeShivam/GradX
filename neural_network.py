import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation_name = activation

        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        # Storage for forward pass(needed for backprop)
        self.z_values = []
        self.a_values = []
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def tanh(self, z):
        return np.tanh(z)
    
    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2
    
    def activate(self, z):
        if self.activation_name == 'relu':
            return self.relu(z)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_name == 'tanh':
            return self.tanh(z)
        else:
            return z
        
    def activate_derivative(self, z):
        if self.activation_name == 'relu':
            return self.relu_derivative(z)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif self.activation_name == 'tanh':
            return self.tanh_derivative(z)
        else:
            return np.ones_like(z)
        
    def forward(self, X):
        self.z_values = []
        self.a_values = [X]

        a = X
        for i in range(self.num_layers - 1):
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)

            if i < self.num_layers - 2:
                a = self.activate(z)
            else:
                a = self.sigmoid(z)
            self.a_values.append(a)
        return a
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        dW = [None] * (self.num_layers - 1)
        db = [None] * (self.num_layers - 1)

        dz = self.a_values[-1] - y
        for i in range(self.num_layers - 2, -1, -1):
            dW[i] = (1 / m) * (self.a_values[i].T @ dz)
            db[i] = (1 / m) * np.sum(dz, axis=0, keepdims=True)

            if i > 0:
                da = dz @ self.weights[i].T
                dz = da * self.activate_derivative(self.z_values[i-1])
            
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]

    def fit(self, X, y, learning_rate=0.01, epochs=100, verbose=True):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        for epoch in range(epochs):
            y_pred = self.forward(X)
            y_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = -np.mean(y * np.log(y_clipped) + 
                           (1 - y) * np.log(1 - y_clipped))
            self.backward(X,y, learning_rate)
            if verbose and epoch % 100 == 0:
                    accuracy = np.mean((y_pred > 0.5).astype(int) == y)
                    print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | Accuracy: {accuracy:.4f}")
    def predict_proba(self,X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.forward(X)
    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
  