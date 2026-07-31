"""
A small feed-forward neural net implemented directly in NumPy, with manual
forward/backward passes and Adam optimizer. No PyTorch/TensorFlow needed —
this keeps the client/server processes lightweight and dependency-free
(just numpy), which also makes the FedAvg weight-averaging step completely
transparent: weights are literally a list of plain ndarrays.

Architecture: input(65) -> Dense(48, ReLU) -> Dense(24, ReLU) -> Dense(1, Sigmoid)
Same shape as the browser TF.js version, so accuracy should land in the same range.
"""
import numpy as np

HIDDEN_1 = 48
HIDDEN_2 = 24


def init_weights(input_dim, seed=None):
    rng = np.random.default_rng(seed)
    def he(fan_in, fan_out):
        return rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)).astype(np.float32)
    return [
        he(input_dim, HIDDEN_1), np.zeros(HIDDEN_1, dtype=np.float32),   # W1, b1
        he(HIDDEN_1, HIDDEN_2), np.zeros(HIDDEN_2, dtype=np.float32),    # W2, b2
        he(HIDDEN_2, 1), np.zeros(1, dtype=np.float32),                  # W3, b3
    ]


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def forward(weights, X):
    W1, b1, W2, b2, W3, b3 = weights
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    a3 = sigmoid(z3)
    cache = (X, z1, a1, z2, a2, z3, a3)
    return a3.ravel(), cache


def backward(weights, cache, y):
    W1, b1, W2, b2, W3, b3 = weights
    X, z1, a1, z2, a2, z3, a3 = cache
    n = X.shape[0]
    y = y.reshape(-1, 1)

    dz3 = (a3 - y) / n                      # binary cross-entropy + sigmoid gradient
    dW3 = a2.T @ dz3
    db3 = dz3.sum(axis=0)

    da2 = dz3 @ W3.T
    dz2 = da2 * (z2 > 0)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return [dW1, db1, dW2, db2, dW3, db3]


def binary_cross_entropy(probs, y):
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class AdamState:
    """Per-weight Adam optimizer moments. Reset at the start of each local round
    (standard practice in FedAvg — the server only ever transmits/averages the
    model weights, never optimizer state)."""
    def __init__(self, weights, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = [np.zeros_like(w) for w in weights]
        self.v = [np.zeros_like(w) for w in weights]
        self.t = 0
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps

    def step(self, weights, grads):
        self.t += 1
        new_weights = []
        for i, (w, g) in enumerate(zip(weights, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            new_weights.append(w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return new_weights


def train_local(weights, X, y, epochs=1, batch_size=128, lr=0.001, seed=0):
    """One local training run on a client's own data, starting from the weights
    the server most recently broadcast. Returns updated weights."""
    rng = np.random.default_rng(seed)
    w = [wi.copy() for wi in weights]
    opt = AdamState(w, lr=lr)
    n = X.shape[0]

    for _ in range(epochs):
        idx = rng.permutation(n)
        Xs, ys = X[idx], y[idx]
        for start in range(0, n, batch_size):
            xb = Xs[start:start + batch_size]
            yb = ys[start:start + batch_size]
            probs, cache = forward(w, xb)
            grads = backward(w, cache, yb)
            w = opt.step(w, grads)
    return w


def evaluate(weights, X, y):
    probs, _ = forward(weights, X)
    loss = binary_cross_entropy(probs, y)
    preds = (probs >= 0.5).astype(np.float32)
    acc = float((preds == y).mean())
    tp = float(((preds == 1) & (y == 1)).sum())
    fp = float(((preds == 1) & (y == 0)).sum())
    tn = float(((preds == 0) & (y == 0)).sum())
    fn = float(((preds == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {'loss': loss, 'accuracy': acc, 'precision': precision, 'recall': recall,
            'f1': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def federated_average(weight_sets, sample_counts):
    """Sample-weighted average of N clients' weight lists — the actual FedAvg step,
    run server-side. Each client contributes proportional to how much data it has."""
    total = sum(sample_counts)
    num_layers = len(weight_sets[0])
    avg = []
    for l in range(num_layers):
        layer_sum = sum(weight_sets[c][l] * (sample_counts[c] / total) for c in range(len(weight_sets)))
        avg.append(layer_sum.astype(np.float32))
    return avg
