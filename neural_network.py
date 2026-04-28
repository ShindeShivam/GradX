import numpy as np
from gradx.optimizers import SGD, Momentum, AdaGrad, RMSProp, Adam, AdaMax, NAdam, AdamW


class NeuralNetwork:
    """
    Multi-Layer Perceptron with Backpropagation
    
    Architecture: Fully connected feedforward neural network
    Training: Gradient descent with backpropagation
    """
    
    def __init__(self, layer_sizes, activation='relu', optimizer='adam', dropout=0):
        """
        Initialize neural network with specified architecture
        
        Args:
            layer_sizes: List of integers [input_size, hidden1, hidden2, ..., output_size]
                        Example: [784, 64, 10] = 784 inputs, 64 hidden, 10 outputs
            activation: Activation function for hidden layers ('relu', 'sigmoid', 'tanh')
            optimizer: Optimizer name string
            dropout: Dropout rate applied after each hidden layer activation (0 = disabled)
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation_name = activation
        self.dropout_rate = dropout
        self.training_mode = True
        
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        self.z_values = []
        self.a_values = []
        self.dropout_mask = []

        self.optimizer = self._create_optimizer(optimizer)

    def _create_optimizer(self, optimizer_name):
        optimizers = {
            'sgd': SGD(learning_rate=0.01),
            'momentum': Momentum(learning_rate=0.01, momentum=0.9),
            'adagrad': AdaGrad(learning_rate=0.01),
            'rmsprop': RMSProp(learning_rate=0.001, decay_rate=0.9),
            'adam': Adam(learning_rate=0.001, beta1=0.9, beta2=0.999),
            'adamax': AdaMax(learning_rate=0.002),
            'nadam': NAdam(learning_rate=0.001),
            'adamw': AdamW(learning_rate=0.001, weight_decay=0.01)
        }
        if optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from {list(optimizers.keys())}")
        return optimizers[optimizer_name]

    # ==================== Activation Functions ====================
    
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
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def activate(self, z):
        if self.activation_name == 'relu':
            return self.relu(z)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_name == 'tanh':
            return self.tanh(z)
        return z
    
    def activate_derivative(self, z):
        if self.activation_name == 'relu':
            return self.relu_derivative(z)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif self.activation_name == 'tanh':
            return self.tanh_derivative(z)
        return np.ones_like(z)

    def _create_batches(self, X, y, batch_size, shuffle=True):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        batches = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batches.append((X[batch_indices], y[batch_indices]))
        return batches

    # ==================== Forward Propagation ====================
    
    def forward(self, X):
        """
        Forward pass: Compute predictions
        
        Flow: X → [Linear → Activation] × layers → Softmax output
        Stores intermediate values for backpropagation
        
        Args:
            X: Input data (n_samples, n_features)
        Returns:
            Final layer output (n_samples, n_outputs)
        """
        self.z_values = []
        self.a_values = [X]
        self.dropout_mask = []
        
        a = X
        for i in range(self.num_layers - 1):
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            if i < self.num_layers - 2:
                a = self.activate(z)
                if self.training_mode and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
                    a = (a * mask) / (1 - self.dropout_rate)
                    self.dropout_mask.append(mask)
                else:
                    self.dropout_mask.append(None)
            else:
                a = self.softmax(z)
            
            self.a_values.append(a)
        
        return a

    # ==================== Backward Propagation ====================
    
    def backward(self, X, y):
        """
        Backward pass: Compute gradients
        
        Uses chain rule to propagate error backward through network:
        ∂Loss/∂W = ∂Loss/∂a × ∂a/∂z × ∂z/∂W
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
        Returns:
            dW: Weight gradients per layer
            dB: Bias gradients per layer
        """
        m = X.shape[0]
        
        dW = [None] * (self.num_layers - 1)
        dB = [None] * (self.num_layers - 1)
        
        dz = self.a_values[-1] - y
        
        for i in range(self.num_layers - 2, -1, -1):
            dW[i] = (1 / m) * (self.a_values[i].T @ dz)
            dB[i] = (1 / m) * np.sum(dz, axis=0, keepdims=True)
            
            if i > 0:
                da = dz @ self.weights[i].T
                dz = da * self.activate_derivative(self.z_values[i-1])
                if self.dropout_mask[i-1] is not None:
                    dz = (dz * self.dropout_mask[i-1]) / (1 - self.dropout_rate)
        
        return dW, dB

    # ==================== Training & Prediction ====================
    
    def fit(self, X, y, learning_rate=0.01, batch_size=32, epochs=100, verbose=True):
        """
        Train the neural network
        
        Args:
            X: Training data (n_samples, n_features)
            y: Training labels (n_samples, n_classes) — one-hot encoded
            learning_rate: Learning rate for gradient descent
            batch_size: Mini-batch size (None = full batch)
            epochs: Number of training iterations
            verbose: Print training progress
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.optimizer.learning_rate = learning_rate
        n_samples = X.shape[0]

        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
            batches = [(X, y)] if batch_size is None else self._create_batches(X, y, batch_size)

            epoch_loss = 0
            epoch_correct = 0

            for idx, (X_batch, y_batch) in enumerate(batches):
                y_pred = self.forward(X_batch)

                y_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
                batch_loss = -np.mean(np.sum(y_batch * np.log(y_clipped), axis=1))
                epoch_loss += batch_loss * X_batch.shape[0]

                predictions = np.argmax(y_pred, axis=1)
                targets = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(predictions == targets)

                dW, dB = self.backward(X_batch, y_batch)

                params = []
                for i in range(len(self.weights)):
                    params.append({'name': 'weight', 'data': self.weights[i], 'grad': dW[i]})
                    params.append({'name': 'bias',   'data': self.biases[i],  'grad': dB[i]})
                self.optimizer.step(params)

                if verbose:
                    print(f"\rBatch {idx+1}/{len(batches)} | Loss: {batch_loss:.4f}", end="")
            
            avg_loss = epoch_loss / n_samples
            accuracy = epoch_correct / n_samples

            if verbose:
                print(f" | Avg Loss: {avg_loss:.6f} | Accuracy: {accuracy:.4f}")
    
    def train(self):
        self.training_mode = True
    
    def eval(self):
        self.training_mode = False
    
    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.forward(X)
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)