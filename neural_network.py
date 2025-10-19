import numpy as np

class NeuralNetwork:
    """
    Multi-Layer Perceptron with Backpropagation
    
    Architecture: Fully connected feedforward neural network
    Training: Gradient descent with backpropagation
    """
    
    def __init__(self, layer_sizes, activation='relu'):
        """
        Initialize neural network with specified architecture
        
        Args:
            layer_sizes: List of integers [input_size, hidden1, hidden2, ..., output_size]
                        Example: [784, 64, 10] = 784 inputs, 64 hidden, 10 outputs
            activation: Activation function for hidden layers ('relu', 'sigmoid', 'tanh')
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation_name = activation
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        
        # He initialization: w ~ N(0, sqrt(2/n_in))
        # Better convergence than random initialization
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Storage for forward pass values (needed for backpropagation)
        self.z_values = []  # Linear outputs: z = W·a + b
        self.a_values = []  # Activations: a = activation(z)
    
    # ==================== Activation Functions ====================
    
    def relu(self, z):
        """ReLU: max(0, z) - Most common for hidden layers"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """ReLU derivative: 1 if z>0, else 0"""
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """Sigmoid: 1/(1+e^-z) - Squashes to [0,1]"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        """Sigmoid derivative: σ(z) * (1 - σ(z))"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def tanh(self, z):
        """Tanh: Squashes to [-1,1]"""
        return np.tanh(z)
    
    def tanh_derivative(self, z):
        """Tanh derivative: 1 - tanh²(z)"""
        return 1 - np.tanh(z) ** 2
    
    def softmax(self, z):
        """Softmax: Converts logits to probabilities (sums to 1)"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def activate(self, z):
        """Apply chosen activation function"""
        if self.activation_name == 'relu':
            return self.relu(z)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_name == 'tanh':
            return self.tanh(z)
        else:
            return z
    
    def activate_derivative(self, z):
        """Derivative of chosen activation function"""
        if self.activation_name == 'relu':
            return self.relu_derivative(z)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif self.activation_name == 'tanh':
            return self.tanh_derivative(z)
        else:
            return np.ones_like(z)
    
    # ==================== Forward Propagation ====================
    
    def forward(self, X):
        """
        Forward pass: Compute predictions
        
        Flow: X → [Linear → Activation] × layers → Output
        Stores intermediate values for backpropagation
        
        Args:
            X: Input data (n_samples, n_features)
        Returns:
            Final layer output (n_samples, n_outputs)
        """
        self.z_values = []
        self.a_values = [X]  # First activation is input itself
        
        a = X
        for i in range(self.num_layers - 1):
            # Linear transformation: z = a·W + b
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation function
            if i < self.num_layers - 2:  # Hidden layers
                a = self.activate(z)
            else:                         # Output layer
                a = self.softmax(z)
            
            self.a_values.append(a)
        
        return a
    
    # ==================== Backward Propagation ====================
    
    def backward(self, X, y, learning_rate):
        """
        Backward pass: Compute gradients and update weights
        
        Uses chain rule to propagate error backward through network:
        ∂Loss/∂W = ∂Loss/∂a × ∂a/∂z × ∂z/∂W
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            learning_rate: Step size for weight updates
        """
        m = X.shape[0]  # Number of samples
        
        # Initialize gradient storage
        dW = [None] * (self.num_layers - 1)
        dB = [None] * (self.num_layers - 1)
        
        # Output layer gradient (softmax + cross-entropy simplifies to a - y)
        dz = self.a_values[-1] - y
        
        # Backpropagate through each layer
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients for weights and biases
            dW[i] = (1/m) * (self.a_values[i].T @ dz)
            dB[i] = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            # Propagate gradient to previous layer
            if i > 0:
                da = dz @ self.weights[i].T              # Gradient w.r.t activation
                dz = da * self.activate_derivative(self.z_values[i-1])  # Chain rule
        
        # Update weights and biases using gradient descent
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * dB[i]
    
    # ==================== Training & Prediction ====================
    
    def fit(self, X, y, learning_rate=0.01, epochs=100, verbose=True):
        """
        Train the neural network
        
        Args:
            X: Training data (n_samples, n_features)
            y: Training labels (n_samples, n_classes) - one-hot encoded
            learning_rate: Learning rate for gradient descent
            epochs: Number of training iterations
            verbose: Print training progress
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss (cross-entropy)
            y_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = -np.mean(y * np.log(y_clipped) + (1 - y) * np.log(1 - y_clipped))
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                accuracy = np.mean((y_pred > 0.5).astype(int) == y)
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | Accuracy: {accuracy:.4f}")
    
    def predict_proba(self, X):
        """Return probability predictions"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.forward(X)
    
    def predict(self, X, threshold=0.5):
        """Return binary class predictions"""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)