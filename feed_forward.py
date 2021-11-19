import numpy as np
def just_feed_forward(train_set, n_x, n_h1, n_h2, n_y):

    W1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) * 0.01
    b3 = np.zeros((n_y, 1))

    return feed_forward_calculations(train_set, W1, W2, W3, b1, b2, b3)

def feed_forward_calculations(train_set, W1, W2, W3, b1, b2, b3):
    A3s_Ys = []
    for A0, label in train_set:
        Z1 = W1 @ A0 + b1
        A1 = 1 / (1 + np.exp(-Z1))
        Z2 = W2 @ A1 + b2
        A2 = 1 / (1 + np.exp(-Z2))
        Z3 = W3 @ A2 + b3
        A3 = 1 / (1 + np.exp(-Z3))
        A3s_Ys.append((A3, label))
    return A3s_Ys

