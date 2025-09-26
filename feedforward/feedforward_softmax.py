import numpy as np
import matplotlib.pyplot as plt


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def tanh(X):
    return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))


def softmax(X):
    return np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)


def feedforward(X, W1, b, W2, c):
    Z = tanh(X.dot(W1) + b)
    A = Z.dot(W2) + c  # activation function
    return softmax(A)


def classification_rate(Y, P):
    assert Y.shape == P.shape

    n_correct = 0
    n_total = 0

    for i in range(Y.shape[0]):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1

    return n_correct / n_total


if __name__ == '__main__':
    # initialize the dimensions
    N = 500  # number of classes
    D = 2  # number of input size
    M = 3  # number of hidden layers
    K = 3  # number of output classes

    # we have three classes
    X1 = np.array([0, -2]) + np.random.randn(N, D)
    X2 = np.array([0, 2]) + np.random.randn(N, D)
    X3 = np.array([-2, 2]) + np.random.randn(N, D)
    X = np.vstack([X1, X2, X3])

    Y = np.array([0] * N + [1] * N + [2] * N)

    plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=.5)
    plt.show()

    # initializing the weights
    W1 = np.random.randn(D, M)
    b = np.random.randn(M)
    W2 = np.random.randn(M, K)
    c = np.random.randn(K)

    p_Y_given_X = np.argmax(feedforward(X, W1, b, W2, c), axis=1)

    print(p_Y_given_X)
    print(Y.shape, p_Y_given_X.shape)
    print(f'Accuracy of the model is: {classification_rate(Y, p_Y_given_X)}')
