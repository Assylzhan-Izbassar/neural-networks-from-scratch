import numpy as np
from neural_network import NeuralNetwork

if __name__ == '__main__':
    N = 500
    D = 2
    M = 4
    K = 3

    X1 = np.array([0, -2]) + np.random.randn(N, D)
    X2 = np.array([2, 2]) + np.random.randn(N, D)
    X3 = np.array([-2, 2]) + np.random.randn(N, D)

    X = np.vstack([X1, X2, X3])
    Y = np.array([0] * N + [1] * N + [2] * N)
    N = len(Y)

    nn = NeuralNetwork(N, D, M, K)
    pY, Z = nn.feedforward(X)
    # pY_given_X = np.argmax(pY, axis=1)
    # print(pY_given_X)
    # print(nn.classification_rate(Y, pY_given_X))
    nn.train(X, Y, 1000)
