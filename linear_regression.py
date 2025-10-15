import numpy as np

class LinearRegression:
    def __init__(self, method="gradient_descent"):
        """
        method: 'gradient_descent' or 'normal_equation'
        """
        self.weights = None
        self.bias = None
        self.method = method
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        n_samples, n_features = X.shape

        if self.method == "normal_equation":
            # Closed-form solution: w = (X^T X)^-1 X^T y
            X_b = np.c_[np.ones((n_samples,1)), X]
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = theta[0,0]
            self.weights = theta[1:].flatten()

            # Final Loss
            y_pred = X @ self.weights + self.bias
            loss = np.mean((y_pred - y)**2)
            print(f"Normal Equation - Final Loss: {loss:.6f}")
        else:
            X = X.squeeze()
            y = y.squeeze()
            # Initialize Parameters
            self.weights = np.zeros(n_features)
            self.bias = 0

            #Gradient Descent
            for epoch in range(epochs):
                #Forward Pass
                y_pred = X @ self.weights + self.bias

                #Compute Gradinets
                dw = (1 / n_samples) * X.T @ (y_pred - y)
                db = (1 / n_samples) * np.sum(y_pred - y)
   
                #Update Parameters
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db

                if epoch == 0:
                    print(f"Starting Loss:{loss:.6f}")
                elif epoch == epochs - 1:
                    print(f"Final Loss:{loss:.6f}")
        
    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.weights + self.bias

if __name__ == "__main__":
    np.random.seed(42)

    X = 2 * np.random.rand(100, 2)
    w_true = np.array([[3],[3]])
    y = 2 +  X @ w_true + np.random.randn(100, 1) 

    # X = X.squeeze()
    # y = y.squeeze()
    print(X.shape)
    print(y.shape)
    model = LinearRegression()
    model_ne = LinearRegression(method='normal_equation')
    model_ne.fit(X,y)
    # model.fit(X,y)
    test_X = np.array([[1, 1], [2, 2]])
    predictions = model_ne.predict(test_X)
    print(predictions)
