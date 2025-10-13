import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        n_samples, n_features = X.shape
        
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

            if epoch % 10 == 0:
                loss = np.mean((y_pred - y)**2)
                print(f"Epoch:{epoch}, Loss:{loss:.4f}")
        
    def predict(self, X):
        return X @ self.weights + self.bias

if __name__ == "__main__":
    np.random.seed(42)

    X = 2 * np.random.rand(100, 2)
    w_true = np.array([[3],[3]])
    y = 2 +  X @ w_true + np.random.randn(100, 1) 

    X = X.squeeze()
    y = y.squeeze()
    print(X.shape)
    print(y.shape)
    model = LinearRegression()
    model.fit(X,y)
