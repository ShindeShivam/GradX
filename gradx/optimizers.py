import numpy as np

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.state = {}
    
    def _key(self, param):
        return id(param['data'])

    def step(self, parameters):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Basic Gradient Descent: w = w - lr * ∇w
    Simple but slow convergence
    """
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def step(self, parameters):
        for param in parameters:
            if param['grad'] is not None:
                param['data'] -= self.learning_rate * param['grad']


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
    
    def step(self, parameters):
        for param in parameters:
            if param['grad'] is not None:
                key = self._key(param)
                if key not in self.state:
                    self.state[key] = {'v': np.zeros_like(param['data'])}
                self.state[key]['v'] = self.momentum * self.state[key]['v'] + param['grad']
                param['data'] -= self.learning_rate * self.state[key]['v']


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
    
    def step(self, parameters):
        for param in parameters:
            if param['grad'] is not None:
                key = self._key(param)
                if key not in self.state:
                    self.state[key] = {'cache': np.zeros_like(param['data'])}
                self.state[key]['cache'] += param['grad'] ** 2
                param['data'] -= self.learning_rate * param['grad'] / (np.sqrt(self.state[key]['cache']) + self.epsilon)


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
    
    def step(self, parameters):
        for param in parameters:
            if param['grad'] is not None:
                key = self._key(param)
                if key not in self.state:
                    self.state[key] = {'cache': np.zeros_like(param['data'])}
                self.state[key]['cache'] = self.decay_rate * self.state[key]['cache'] + (1 - self.decay_rate) * param['grad'] ** 2
                param['data'] -= self.learning_rate * param['grad'] / (np.sqrt(self.state[key]['cache']) + self.epsilon)


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
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
    
    def step(self, parameters):
        self.t += 1
        for param in parameters:
            if param['grad'] is not None:
                key = self._key(param)
                if key not in self.state:
                    self.state[key] = {
                        'm': np.zeros_like(param['data']),
                        'v': np.zeros_like(param['data'])
                    }
                self.state[key]['m'] = self.beta1 * self.state[key]['m'] + (1 - self.beta1) * param['grad']
                self.state[key]['v'] = self.beta2 * self.state[key]['v'] + (1 - self.beta2) * param['grad'] ** 2
                m_hat = self.state[key]['m'] / (1 - self.beta1 ** self.t)
                v_hat = self.state[key]['v'] / (1 - self.beta2 ** self.t)
                param['data'] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class AdaMax(Optimizer):
    """
    AdaMax: Variant of Adam based on infinity norm
    Uses max norm instead of L2 norm 
    """
    def __init__(self, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
    
    def step(self, parameters):
        self.t += 1
        for param in parameters:
            if param['grad'] is not None:
                key = self._key(param)
                if key not in self.state:
                    self.state[key] = {
                        'm': np.zeros_like(param['data']),
                        'u': np.zeros_like(param['data'])
                    }
                self.state[key]['m'] = self.beta1 * self.state[key]['m'] + (1 - self.beta1) * param['grad']
                self.state[key]['u'] = np.maximum(self.beta2 * self.state[key]['u'], np.abs(param['grad']))
                param['data'] -= (self.learning_rate / (1 - self.beta1 ** self.t)) * self.state[key]['m'] / (self.state[key]['u'] + self.epsilon)


class NAdam(Optimizer):
    """
    NAdam: Adam with Nesterov momentum
    
    Applies Nesterov lookahead to Adam's first moment estimate
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
    
    def step(self, parameters):
        self.t += 1
        for param in parameters:
            if param['grad'] is not None:
                key = self._key(param)
                if key not in self.state:
                    self.state[key] = {
                        'm': np.zeros_like(param['data']),
                        'v': np.zeros_like(param['data'])
                    }
                self.state[key]['m'] = self.beta1 * self.state[key]['m'] + (1 - self.beta1) * param['grad']
                self.state[key]['v'] = self.beta2 * self.state[key]['v'] + (1 - self.beta2) * param['grad'] ** 2
                m_hat = self.state[key]['m'] / (1 - self.beta1 ** self.t)
                v_hat = self.state[key]['v'] / (1 - self.beta2 ** self.t)
                param['data'] -= (self.learning_rate / (np.sqrt(v_hat) + self.epsilon)) * \
                    (self.beta1 * m_hat + (1 - self.beta1) * param['grad'] / (1 - self.beta1 ** self.t))


class AdamW(Optimizer):
    """
    AdamW: Adam with decoupled weight decay
    
    Weight decay is applied directly to parameters rather than
    being folded into the gradient, which is more principled
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.t = 0
    
    def step(self, parameters):
        self.t += 1
        for param in parameters:
            if param['grad'] is not None:
                key = self._key(param)
                if key not in self.state:
                    self.state[key] = {
                        'm': np.zeros_like(param['data']),
                        'v': np.zeros_like(param['data'])
                    }
                self.state[key]['m'] = self.beta1 * self.state[key]['m'] + (1 - self.beta1) * param['grad']
                self.state[key]['v'] = self.beta2 * self.state[key]['v'] + (1 - self.beta2) * param['grad'] ** 2
                m_hat = self.state[key]['m'] / (1 - self.beta1 ** self.t)
                v_hat = self.state[key]['v'] / (1 - self.beta2 ** self.t)
                param['data'] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

                if param['name'] in ['weight', 'gamma']:
                    param['data'] -= self.learning_rate * self.weight_decay * param['data']