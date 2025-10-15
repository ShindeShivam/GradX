import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def fit_transform(self, X):
        """Generate polynomial features"""
        if X.ndim == 1:
            X = X.reshape(-1,1)
        
        n_samples, n_features = X.shape

        features = []

        combs = []
        # Generate all combinations of feature indices up to the given degree
        for d in range(0 if self.include_bias else 1, self.degree + 1):
            combs.extend(combinations_with_replacement(range(n_features), d))
        
        for comb in combs:
            if len(comb) == 0:
                # Bias term
                feature = np.ones((n_samples, 1))
            else:
                feature = np.prod(X[:,comb], axis=1, keepdims=True)
            
            features.append(feature)
        return np.hstack(features)
    
class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree)
        self.weights = None
        

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        X = X.squeeze()
        y = y.squeeze()

        # Transform to polynomial features
        X_poly = self.poly_features.fit_transform(X)
        n_samples, n_features = X_poly.shape

        # Initialize weights (no separate bias needed - it's in features)
        self.weights = np.zeros(n_features)

        # Gradient Descent
        for epoch in range(epochs):
            # Forward Pass
            y_pred = X_poly @ self.weights 
            
            # Compute Gradients
            dw = (1 / n_samples) * X_poly.T @ (y_pred - y)

            # Update weights
            self.weights -= learning_rate * dw

            # Loss
            loss = np.mean((y_pred - y)**2)
            if epoch == 0:
                print(f"Starting Loss:{loss:.6f}")
            elif epoch == epochs - 1:
                print(f"Final Loss:{loss:.6f}")

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_poly = self.poly_features.fit_transform(X)
        return X_poly @ self.weights
        
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate non-linear data: y = 3x^2 + 2x + 1 + noise
    X = np.linspace(-3, 3, 100)
    y = 3 * X**2 + 2 * X + 1 + np.random.randn(100) * 2
    test_X = np.array([-2, 0, 2])
    
    
   
    model = PolynomialRegression(degree=2)
    model.fit(X, y)
    predictions = model.predict(test_X)
    print(predictions)