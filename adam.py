import numpy as np


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = None   # 1st moment estimate (mean of gradients)
        self.v = None   # 2nd raw moment estimate (uncentered variance)
        self.t = 0      # timestep

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        # Update biased moment estimates (exponential moving averages)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        # Bias correction (moments start at 0, so they are biased toward 0)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        # Parameter update
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdaMax:
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = None
        self.u = None   # exponentially weighted infinity norm
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.u = np.zeros_like(params)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.u = np.maximum(self.beta2 * self.u, np.abs(grads))
        # Only the first moment needs bias correction
        step = (self.lr / (1 - self.beta1 ** self.t)) * self.m / (self.u + self.eps)
        return params - step


class SGDMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.mu, self.v = lr, momentum, None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.mu * self.v - self.lr * grads
        return params + self.v


def make_data(n=2000, d=20, seed=0):
    rng = np.random.default_rng(seed)
    # Badly scaled features: Adam's per-parameter scaling should help here
    X = rng.normal(size=(n, d)) * np.logspace(-1, 1, d)
    w_true = rng.normal(size=d)
    y = (X @ w_true + rng.normal(size=n) > 0).astype(float)
    return X, y


def loss_and_grad(w, Xb, yb):
    p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
    loss = -np.mean(yb * np.log(p + 1e-12) + (1 - yb) * np.log(1 - p + 1e-12))
    grad = Xb.T @ (p - yb) / len(yb)
    return loss, grad


def train(optimizer, X, y, epochs=30, batch=128, seed=1):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    history = []
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            b = idx[i:i + batch]
            _, g = loss_and_grad(w, X[b], y[b])   # stochastic (minibatch) gradient
            w = optimizer.step(w, g)
        history.append(loss_and_grad(w, X, y)[0])
    return history


if __name__ == "__main__":
    X, y = make_data()
    runs = {
        "Adam (lr=0.05)":    Adam(lr=0.05),
        "AdaMax (lr=0.05)":  AdaMax(lr=0.05),
        "SGD+momentum":      SGDMomentum(lr=0.001, momentum=0.9),
    }
    print(f"{'optimizer':<20}{'epoch 1':>10}{'epoch 10':>10}{'epoch 30':>10}")
    for name, opt in runs.items():
        h = train(opt, X, y)
        print(f"{name:<20}{h[0]:>10.4f}{h[9]:>10.4f}{h[29]:>10.4f}")