import numpy as np

class Module:
    def __init__(self):
        self.training = True
        self._parameters = []
        self._modules = []
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        """Return all trainable parameters"""
        params = []
        params.extend(self._parameters) # Get parameters from this module
        for module in self._modules:
            params.extend(module.parameters()) # Get parameters from child modules
        return params
    
    def train(self):
        """Set to training mode"""
        self.training = True
        for module in self._modules:
            module.train()
    
    def eval(self):
        """Set to evaluation mode"""
        self.training = False
        for module in self._modules:
            module.eval()


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Weight Initialization
        self.weights = np.random.randn(self.in_features, self.out_features) * np.sqrt(2 / self.in_features)
        self.bias = np.zeros((1, self.out_features)) if self.use_bias else None

        # Cache for backprop
        self.input = None  # Store input (like a_values[i] in old implementation)
        self.output = None  # Store output if needed

        # Gradients 
        self.grad_weights = None
        self.grad_bias = None

        # Register as parameters
        self._parameters = [
            {'name': 'weight', 'data': self.weights, 'grad': None},
            {'name': 'bias', 'data': self.bias, 'grad': None} if self.use_bias else None
        ]

        self._parameters = [p for p in self._parameters if p is not None] # The None is removed if use_bias is False

    def forward(self, x):
        self.input = x
        self.output = x @ self.weights
        if self.use_bias:
            self.output += self.bias
        return self.output
    
    def backward(self, grad_output):
         """
        Backward pass
        
        Uses stored self.input 
        Args:
            grad_output: Gradient from next layer (batch_size, out_features)
            grad_output = dZ = gradient of loss w.r.t. this layer's output
        
        Returns:
            grad_input: Gradient to pass to previous layer (batch_size, in_features)
            
        """
         batch_size = grad_output.shape[0]

         # Gradient w.r.t weights: X^T @ grad_output
         self.grad_weights = (1 / batch_size) * (self.input.T @ grad_output)
         # Gradient w.r.t bias: sum over batch
         if self.use_bias:
             self.grad_bias = (1 / batch_size) * np.sum(grad_output, axis=0, keepdims=True)
         # Store gradients in parameters dict (for optimizer)
         for param in self._parameters:
             if param['name'] == 'weight':
                 param['grad'] = self.grad_weights
             elif param['name'] == 'bias':
                 param['grad'] = self.grad_bias
         # Gradient w.r.t input: grad_output @ W^T
         # (Pass this to previous layer)
         grad_input = grad_output @ self.weights.T

         return grad_input
    
    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"

                 
class Sequential(Module):
    """
    Sequential container 
    Automatically handles forward and backward through all layers
    Each layer stores its own intermediate values
    """
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        self._modules = self.layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Each layer stores what it needs in its own attributes
        return x
    
    def backward(self, grad_output):  
        """
        Backward pass through all layers in reverse
        
        Each layer uses its stored intermediate values for gradient computation
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

        return grad_output
    
    def add(self, layer):
         """Add a layer to the sequence"""
         self.layers.append(layer)
         self._modules.append(layer)

    def __repr__(self):
        layer_strs = [f"  ({i}): {layer}" for i, layer in enumerate(self.layers)]
        return "Sequential(\n" + "\n".join(layer_strs) + "\n)"
    
    def __getitem__(self, idx):
        """Access layers by index"""
        return self.layers[idx]
    
    def __len__(self):
        return len(self.layers)
    
    
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0).astype(float)
    
    def __repr__(self):
        return "ReLU()"
    
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None
    
    def forward(self, x):
        self.output = np.tanh(x)
    
    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

class Dropout(Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        self.mask = None #Store mask for backward pass
    
    def forward(self, x):
        """Apply dropout and store mask"""
        if self.training and self.p > 0:
            self.mask = np.random.binomial(1, 1 - self.p, size=x.shape)
            return (x * self.mask) / (1 - self.p)
        else:
            self.mask = None
            return x
        
    def backward(self, grad_output):
        """Use stored mask for gradient"""
        if self.training and self.p > 0 and self.mask is not None:
            return (grad_output * self.mask) / (1 - self.p)
        else:
            return grad_output
    def __repr__(self):
        return f"Drpout(p={self.p})"
        