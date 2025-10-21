import numpy as np

class Optimizer:
    """ Base class for all optimizers """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, weights, biases, dW, dB):
        raise NotImplementedError
    
class SGD(Optimizer):
    """
    Basic Gradient Descent: w = w - lr * ∇w
    Simple but slow convergence
    """
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, weights, biases, dW, dB):
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * dW[i]
            biases[i] -= self.learning_rate * dB[i]
        return weights, biases

class Momentum(Optimizer):
    """
    SGD with Momentum: Accelerates in consistent directions
    
    v = β*v + ∇w
    w = w - lr * v
    
    Typical β = 0.9 (keeps 90% of previous direction)
    Helps escape local minima and speeds up convergence
    """
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biases, dW, dB):
        # Initialize velocity on first call
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.velocity_b = [np.zeros_like(b) for b in biases]

        # Update velocity and weights
        for i in range(len(weights)):
            self.velocity_w[i] = self.momentum * self.velocity_w[i] + dW[i]
            self.velocity_b[i] = self.momentum * self.velocity_b[i] + dB[i]
            weights[i] -= self.learning_rate * self.velocity_w[i]
            biases[i] -= self.learning_rate * self.velocity_b[i]

        return weights, biases


class Nesterov(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity_w = None
        self.velocity_b = None

    def update(self, weights, biases, dW, dB):
        if self.velocity_w is None:
            self.velocity_w = [np.zeros_like(w) for w in weights]
            self.velocity_b = [np.zeros_like(b) for b in biases]

        
        for i in range(len(weights)):
            # Look-ahead step
            lookahead_w = weights[i] - self.learning_rate * self.velocity_w[i]
            lookahead_b = biases[i] - self.learning_rate * self.velocity_b[i]

            # Velocity Update
            self.velocity_w[i] = self.velocity_w[i] + dW[i]
            self.velocity_b[i] = self.velocity_b[i] + dB[i]

            # Weight Update
            weights[i] -= self.learning_rate * self.velocity_w[i]
            biases[i] -= self.learning_rate * self.velocity_b[i]
        
        return weights, biases
    
class AdaGrad(Optimizer):
    """
    AdaGrad: Adapts learning rate based on historical gradients
    
    cache = cache + ∇w²
    w = w - lr * ∇w / (√cache + ε)
    
    Learning rate decreases for frequently updated parameters
    Good for sparse data, but can be too aggressive (lr → 0)
    """
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache_w = None
        self.cache_b = None

    def update(self, weights, biases, dW, dB):
        # Initialize cache on first call
        if self.cache_w is None:
            self.cache_w = [np.zeros_like(w) for w in weights]
            self.cache_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            # Accumulate squared gradients 
            self.cache_w[i] += dW[i] ** 2
            self.cache_b[i] += dB[i] ** 2

            # Adaptive learning rate
            weights[i] -= self.learning_rate * dW[i] / (np.sqrt(self.cache_w[i]) + self.epsilon)
            biases[i] -= self.learning_rate * dB[i] / (np.sqrt(self.cache_b[i]) + self.epsilon)

        return weights, biases 


        
        

