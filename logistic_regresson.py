import numpy as np

class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        if  X.ndim == 1:
            X = X.reshape(-1,1)
        if y.ndim == 1:
            y = y.reshape(-1,1)

        X = X.squeeze()
        y = y.squeeze()

        n_samples, n_features = X.shape
        # Initialize Parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for epoch in range(epochs):
            # Forward Pass
            linear_output = X @ self.weights + self.bias
            y_pred = self.sigmoid(linear_output)

            # Compute Gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update Parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # Binary Cross-Entropy Loss
            # Clip predictions to prevent log(0)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            if epoch == 0:
                print(f"Starting Loss:{loss:.6f}")
            elif epoch == epochs - 1:
                print(f"Final Loss:{loss:.6f}")

    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        linear_output = X @ self.weights + self.bias
        return self.sigmoid(linear_output)
    
    def predict(self, X, thershold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= thershold).astype(int)
    
    


