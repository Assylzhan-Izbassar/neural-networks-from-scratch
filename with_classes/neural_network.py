import numpy as np
# import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, N, D, M, K):
        self.N = N
        self.D = D
        self.M = M
        self.K = K

        self.W1 = np.random.randn(D, M)
        self.b1 = np.random.randn(M)
        self.W2 = np.random.randn(M, K)
        self.b2 = np.random.randn(K)

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def softmax(X):
        return np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)

    @staticmethod
    def classification_rate(T, pY):
        # print(T.shape, pY.shape)
        assert T.shape == pY.shape
        return (T == pY).sum() / T.shape[0]

    @staticmethod
    def cost(T, pY):
        return (T * np.log(pY)).sum()

    def feedforward(self, X):
        Z = self.sigmoid(X.dot(self.W1) + self.b1)
        A = Z.dot(self.W2) + self.b2
        return self.softmax(A), Z

    def _derivative_W1(self, T, pY, Z, W2, X):
        dZ = Z * (1 - Z)
        return X.T.dot((T - pY).dot(W2.T) * dZ)

    def _derivative_b1(self, T, pY, Z, W2):
        dZ = Z * (1 - Z)
        return ((T - pY).dot(W2.T) * dZ).sum(axis=0)

    def _derivative_W2(self, T, pY, Z):
        return Z.T.dot(T - pY)

    def _derivative_b2(self, T, pY):
        return (T - pY).sum(axis=0)

    def _one_hot(self, Y):
        T = np.zeros((self.N, self.K))
        for i in range(self.N):
            T[i, Y[i]] = 1
        return T

    def train(self, X, Y, epochs, learning_rate=1e-3):
        Y_one_hot = self._one_hot(Y)
        costs = []
        for epoch in range(epochs):
            pY, Z = self.feedforward(X)

            if not epoch % 100:
                iter_cost = self.cost(Y_one_hot, pY)
                pY_given_X = np.argmax(pY, axis=1)
                rate = self.classification_rate(Y, pY_given_X)
                costs.append(iter_cost)
                # print('Cost: ', iter_cost, ' with classification rate: ', rate)

            gW2 = self._derivative_W2(Y_one_hot, pY, Z)
            gb2 = self._derivative_b2(Y_one_hot, pY)
            gW1 = self._derivative_W1(Y_one_hot, pY, Z, self.W2, X)
            gb1 = self._derivative_b1(Y_one_hot, pY, Z, self.W2)

            self.W2 += learning_rate * gW2
            self.b2 += learning_rate * gb2
            self.W1 += learning_rate * gW1
            self.b1 += learning_rate * gb1

        # plt.plot(costs)
        # plt.show()
        return self.W1, self.b1, self.W2, self.b2
