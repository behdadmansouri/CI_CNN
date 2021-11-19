import numpy as np
def just_feed_forward(train_set, weights):
    W1, b1, W2, b2, W3, b3 = weights

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

