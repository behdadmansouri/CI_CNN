import numpy as np
import random


def backpropagation(train_set, data_limit, n_x, n_h1, n_h2, n_y, epochs, batch_size, weights, alpha, vectorized):
    total_cost = []
    A3s_Ys = []

    W1, b1, W2, b2, W3, b3 = weights

    for range_epoch in range(epochs):
        random.shuffle(train_set)
        A3s_Ys = []
        # calculate cost average per epoch
        cost = 0
        batches = [train_set[x:x + batch_size] for x in
                   range(0, data_limit, batch_size)]  # we don't delete second column anymore
        for batch in batches:
            grad_W1 = np.zeros((n_h1, n_x))
            grad_b1 = np.zeros((n_h1, 1))
            grad_W2 = np.zeros((n_h2, n_h1))
            grad_b2 = np.zeros((n_h2, 1))
            grad_W3 = np.zeros((n_y, n_h2))
            grad_b3 = np.zeros((n_y, 1))
            for A0, label in batch:
                Z1 = W1 @ A0 + b1
                A1 = 1 / (1 + np.exp(-Z1))
                Z2 = W2 @ A1 + b2
                A2 = 1 / (1 + np.exp(-Z2))
                Z3 = W3 @ A2 + b3
                A3 = 1 / (1 + np.exp(-Z3))
                A3s_Ys.append((A3, label))
                # batch
                for j in range(n_y):
                    cost += np.power(A3[j] - label[j], 2)
                if vectorized:
                    vectorized_grad(A0, A1, A2, A3, W2, W3, grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3, label,
                                    n_h1, n_h2)
                else:
                    grad(A0, A1, A2, A3, W2, W3, grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3, label, n_h1,
                         n_h2)

            W1 = W1 - (alpha * (grad_W1 / batch_size))
            b1 = b1 - (alpha * (grad_b1 / batch_size))
            W2 = W2 - (alpha * (grad_W2 / batch_size))
            b2 = b2 - (alpha * (grad_b2 / batch_size))
            W3 = W3 - (alpha * (grad_W3 / batch_size))
            b3 = b3 - (alpha * (grad_b3 / batch_size))
            # use for to find cost for each W and B
        total_cost.append(sum(cost))
        # print(f"{range_epoch + 1} epochs done out of {epochs}")
    return A3s_Ys, total_cost, (W1, b1, W2, b2, W3, b3)

    #
    # for A0 in train_set[:data_limit]:
    #     Z1 = np.cross(W1, A0[0]) + b1
    #     A1 = 1 / (1 + np.exp(-Z1))
    #     Z2 = np.cross(W2, A1) + b2
    #     A2 = 1 / (1 + np.exp(-Z2))
    #     Z3 = np.cross(W3, A2) + b3
    #     A3 = 1 / (1 + np.exp(-Z3))
    #
    #     predicted_number = np.where(A3 == np.amax(A3))
    #     print(predicted_number)
    #     print(train_set[1])
    #     print("----------------")
    #     print(np.amax(train_set[1]))
    #
    #     real_number = np.where(train_set[1] == np.amax(train_set[1]))
    #
    #     if predicted_number == real_number:
    #         number_of_correct_estimations += 1
    #
    # print(f"Accuracy: {number_of_correct_estimations / data_limit}")


def grad(A0, A1, A2, A3, W2, W3, grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3, label, n_h1, n_h2):
    # weight
    for j in range(grad_W3.shape[0]):
        for k in range(grad_W3.shape[1]):
            grad_W3[j, k] += 2 * (A3[j, 0] - label[j, 0]) * A3[j, 0] * (1 - A3[j, 0]) * A2[k, 0]
    # bias
    for j in range(grad_b3.shape[0]):
        grad_b3[j, 0] += 2 * (A3[j, 0] - label[j, 0]) * A3[j, 0] * (1 - A3[j, 0])
    # ---- 3rd layer
    # activation
    delta_3 = np.zeros((n_h2, 1))
    for k in range(grad_W3.shape[1]):
        for j in range(grad_W3.shape[0]):
            delta_3[k, 0] += 2 * (A3[j, 0] - label[j, 0]) * A3[j, 0] * (1 - A3[j, 0]) * W3[j, k]
    # weight
    for k in range(grad_W2.shape[0]):
        for m in range(grad_W2.shape[1]):
            grad_W2[k, m] += delta_3[k, 0] * A2[k, 0] * (1 - A2[k, 0]) * A1[m, 0]
    # bias
    for k in range(grad_b2.shape[0]):
        grad_b2[k, 0] += delta_3[k, 0] * A2[k, 0] * (1 - A2[k, 0])
    # ---- 2nd layer
    # activation
    delta_2 = np.zeros((n_h1, 1))
    for m in range(grad_W2.shape[1]):
        for k in range(grad_W2.shape[0]):
            delta_2[m, 0] += delta_3[k, 0] * A2[k, 0] * (1 - A2[k, 0]) * W2[k, m]
    # weight
    for m in range(grad_W1.shape[0]):
        for v in range(grad_W1.shape[1]):
            grad_W1[m, v] += delta_2[m, 0] * A1[m, 0] * (1 - A1[m, 0]) * A0[v, 0]
    # bias
    for m in range(grad_b1.shape[0]):
        grad_b1[m, 0] += delta_2[m, 0] * A1[m, 0] * (1 - A1[m, 0])


def vectorized_grad(A0, A1, A2, A3, W2, W3, grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3, label, n_h1, n_h2):
    # ---- Last layer
    # weight
    grad_W3 += (2 * (A3 - label) * A3 * (1 - A3)) @ np.transpose(A2)
    # bias
    grad_b3 += 2 * (A3 - label) * A3 * (1 - A3)
    # ---- 3rd layer
    # activation
    # delta_3 = np.zeros((n_h2, 1))
    delta_3 = np.transpose(W3) @ (2 * (A3 - label) * (A3 * (1 - A3)))
    # weight
    grad_W2 += (A2 * (1 - A2) * delta_3) @ np.transpose(A1)
    # bias
    grad_b2 += delta_3 * A2 * (1 - A2)
    # ---- 2nd layer
    # activation
    # delta_2 = np.zeros((n_h1, 1))
    delta_2 = np.transpose(W2) @ (delta_3 * A2 * (1 - A2))
    # weight
    grad_W1 += (delta_2 * A1 * (1 - A1)) @ np.transpose(A0)
    # bias
    grad_b1 += delta_2 * A1 * (1 - A1)
