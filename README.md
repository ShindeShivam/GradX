# GradX

Machine learning and deep learning implemented from scratch in Python — no TensorFlow, no PyTorch, just NumPy.

---

## Algorithms

### Regression
| Algorithm | File |
|---|---|
| Linear Regression (Gradient Descent + Normal Equation) | `linear_regression.py` |
| Logistic Regression | `logistic_regression.py` |
| Polynomial Regression | `polynomial_regression.py` |

### Neural Networks
| Component | File |
|---|---|
| Single Neuron with Backprop | `neuron.py` |
| Multi-Layer Perceptron | `neural_network.py` |
| Modular Layer API (Linear, ReLU, Sigmoid, Tanh, Dropout) | `gradx/nn.py` |

### Optimizers
| Optimizer | File |
|---|---|
| SGD, Momentum, AdaGrad, RMSProp | `gradx/optimizers.py` |
| Adam, AdaMax, NAdam, AdamW | `gradx/optimizers.py` |

---

## Usage

```python
from neural_network import NeuralNetwork

model = NeuralNetwork(layer_sizes=[784, 128, 64, 10], activation='relu', optimizer='adam')
model.fit(X_train, y_train, learning_rate=0.001, batch_size=32, epochs=20)
```

```python
from gradx.nn import Sequential, Linear, ReLU, Dropout
from gradx.optimizers import Adam

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)
optimizer = Adam(learning_rate=0.001)
```

See `validation_and_examples.ipynb` for full training examples including MNIST.