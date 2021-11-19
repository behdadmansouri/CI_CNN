import pickle
import numpy as np


def loading_datasets():
    import random

    # loading training set features
    f = open("Datasets/train_set_features.pkl", "rb")
    train_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=train_set_features2, axis=0)
    train_set_features = train_set_features2[:, features_STDs > 52.3]

    # changing the range of data between 0 and 1
    train_set_features = np.divide(train_set_features, train_set_features.max())

    # loading training set labels
    f = open("Datasets/train_set_labels.pkl", "rb")
    train_set_labels = pickle.load(f)
    f.close()

    # ------------
    # loading test set features
    f = open("Datasets/test_set_features.pkl", "rb")
    test_set_features2 = pickle.load(f)
    f.close()

    # reducing feature vector length
    features_STDs = np.std(a=test_set_features2, axis=0)
    test_set_features = test_set_features2[:, features_STDs > 48]

    # changing the range of data between 0 and 1
    test_set_features = np.divide(test_set_features, test_set_features.max())

    # loading test set labels
    f = open("Datasets/test_set_labels.pkl", "rb")
    test_set_labels = pickle.load(f)
    f.close()

    # ------------
    # preparing our training and test sets - joining datasets and lables
    train_set = []
    test_set = []

    for i in range(len(train_set_features)):
        label = np.array([0, 0, 0, 0])
        label[int(train_set_labels[i])] = 1
        label = label.reshape(4, 1)
        train_set.append((train_set_features[i].reshape(102, 1), label))

    for i in range(len(test_set_features)):
        label = np.array([0, 0, 0, 0])
        label[int(test_set_labels[i])] = 1
        label = label.reshape(4, 1)
        test_set.append((test_set_features[i].reshape(102, 1), label))

    # shuffle
    random.shuffle(train_set)
    random.shuffle(test_set)

    # print size
    # print(len(train_set))  # 1962
    # print(len(test_set))  # 662
    return train_set, test_set


#
# def print_correct_percentage(A3s_Ys):
#     correct = 0
#
#     for A3, Y in A3s_Ys:
#         predicted_number = A3.tolist().index(max(A3))
#         real_number = Y.tolist().index([1])
#         print(predicted_number, real_number)
#         if predicted_number == real_number:
#             correct += 1
#
#     print(f"Accuracy: {correct / len(A3s_Ys)}")
#
# def plot_costs_per_epoch(epochs, total_cost):
#     import matplotlib.pyplot as plt
#     epoch_size = [x + 1 for x in range(epochs)]
#     plt.plot(epoch_size, [cost / 4 for cost in total_cost])
#     plt.show()


def plot_cost(epochs, total_cost):
    import matplotlib.pyplot as plt

    epoch_size = [x + 1 for x in range(epochs)]
    for each in total_cost:
        plt.plot(epoch_size, [cost / 4 for cost in each])
    plt.show()


def print_correct(A3s_Ys):
    accu = []
    for each in A3s_Ys:
        correct = 0
        for A3, Y in each:
            predicted_number = A3.tolist().index(max(A3))
            real_number = Y.tolist().index([1])
            if predicted_number == real_number:
                correct += 1
        accu.append(correct / len(each))
    print(f"average accuracy: {100 * sum(accu) / len(accu)}")


def default_weights(n_x, n_h1, n_h2, n_y):
    W1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros((n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros((n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) * 0.01
    b3 = np.zeros((n_y, 1))
    return W1, b1, W2, b2, W3, b3
