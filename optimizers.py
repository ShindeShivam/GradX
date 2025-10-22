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


class RMSProp(Optimizer):
    """
    RMSprop: Adapts learning rate per parameter
    
    s = β*s + (1-β)*∇w²
    w = w - lr * ∇w / (√s + ε)
    
    Divides gradient by running average of its magnitude
    """
    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
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
            self.cache_w[i] = self.decay_rate * self.cache_w[i] + (1 - self.decay_rate) * dW[i] ** 2
            self.cache_b[i] = self.decay_rate * self.cache_b[i] + (1 - self.decay_rate) * dB[i] ** 2

            # Adaptive learning rate update
            weights[i] -= self.learning_rate * dW[i] / (np.sqrt(self.cache_w[i]) + self.epsilon)
            biases[i] -= self.learning_rate * dB[i] / (np.sqrt(self.cache_b[i]) + self.epsilon)
        
        return weights, biases

        
class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation 
    
    Combines Momentum + RMSprop:
    - Momentum: Uses moving average of gradients
    - RMSprop: Uses moving average of squared gradients
    
    m = β1*m + (1-β1)*∇w        (first moment - mean)
    v = β2*v + (1-β2)*∇w²        (second moment - variance)
    
    m_hat = m / (1-β1^t)         (bias correction)
    v_hat = v / (1-β2^t)
    
    w = w - lr * m_hat / (√v_hat + ε)
    
    Default: β1=0.9, β2=0.999, lr=0.001
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # First moment (momentum)
        self.m_w = None
        self.m_b = None
        # Second moment(RMSprop)
        self.v_w = None
        self.v_b = None

        self.t = 0

    def update(self, weights, biases, dW, dB):
        # Initialize moments on first call
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        self.t += 1
        for i in range(len(weights)):
            # Update biased first moment (momentum)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * dB[i]
            # Update biased second moment (RMSprop)
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * dW[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * dB[i]**2

            # Bias correction (important for early iterations!)
            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

            # Update parameters
            weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            
        return weights, biases
        

class AdaMax(Optimizer):
    """
    AdaMax: Variant of Adam based on infinity norm
    Uses max norm instead of L2 norm 
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # First moment (momentum)
        self.m_w = None
        self.m_b = None
        # Infinity norm
        self.u_w = None
        self.u_b = None
        self.t = 0

    def update(self, weights, biases, dW, dB):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.u_w = [np.zeros_like(w) for w in weights]
            self.u_b = [np.zeros_like(b) for b in biases]

        self.t += 1
        for i in range(len(weights)):
            # Update biased first moment (momentum)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * dB[i]
            # Update biased second moment
            self.u_w[i] = np.maximum(self.beta2 * self.u_w[i], np.abs(dW[i]))
            self.u_b[i] = np.maximum(self.beta2 * self.u_b[i], np.abs(dB[i]))

            # Update parameters
            weights[i] -= (self.learning_rate) / (1 - self.beta1**self.t) * self.m_w[i] / (self.u_w[i] + self.epsilon)
            biases[i] -= (self.learning_rate) / (1- self.beta1**self.t) * self.m_b[i] / (self.u_b[i] + self.epsilon)

        return weights, biases

class NAdam(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # First moment (momentum)
        self.m_w = None
        self.m_b = None
        # Second moment
        self.v_w = None
        self.v_b = None
        self.t = 0
    
    def update(self, weights, biases, dW, dB):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        self.t += 1

        for i in range(len(weights)):
            # Update biased first moment (momentum)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * dB[i]
            # Update biased second moment
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * dW[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * dB[i]**2
            # Bias correction (important for early iterations!)
            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

            # Update parameters
            weights[i] -= ( self.learning_rate / (np.sqrt(v_w_hat) + self.epsilon)) * (self.beta1 * m_w_hat + (1 - self.beta1) * dW[i] / (1 - self.beta1**self.t))
            biases[i] -= (self.learning_rate / (np.sqrt(v_b_hat) + self.epsilon)) * (self.beta1 * m_b_hat + (1 - self.beta1) * dB[i] / (1 - self.beta1**self.t))

        return weights, biases
        

class AdamW(Optimizer):
    """
    Adam + weight_decay
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, weight_decay=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        # First moment (momentum)
        self.m_w = None
        self.m_b = None
        # Second moment
        self.v_w = None
        self.v_b = None
        self.t = 0
    
    def update(self, weights, biases, dW, dB):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        self.t += 1
        for i in range(len(weights)):
            # Update biased first moment (momentum)
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * dB[i]
            # Update biased second moment
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * dW[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * dB[i]**2
            # Bias correction (important for early iterations!)
            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

            # Update parameters
            weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            weights[i] -= self.learning_rate * self.weight_decay * weights[i]
            biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        return weights, biases
        

        
