import numpy as np
import matplotlib.pyplot as plt


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def softmax(X):
    # X = X - X.max(axis=1, keepdims=True)
    expA = np.exp(X)
    return expA / expA.sum(axis=1, keepdims=True)


def feedforward(X, W, b, V, c):
    Z = sigmoid(X.dot(W) + b)
    return softmax(Z.dot(V) + c), Z


def classification_rate(Y, pY):
    assert Y.shape == pY.shape
    return sum(Y == pY) / Y.shape[0]


def derivative_W(T, Y, Z, V, X):
    N, K = T.shape
    M = Z.shape[1]
    D = X.shape[1]

    total = np.zeros((D, M))

    # for n in range(N):
    #     for k in range(K):
    #         for m in range(M):
    #             for d in range(D):
    #                 total[d, m] += ((T[n, k] - Y[n, k]) * Z[n, m]
    #                                 * (1 - Z[n, m]) * V[m, k]) * X[n, d]

    # return total
    
    # for n in range(N):
    #     for k in range(K):
    #         # for m in range(M):
    #             for d in range(D):
    #                 total[d, :] += ((T[n, k] - Y[n, k]) * Z[n, :]
    #                                 * (1 - Z[n, :]) * V[:, k]) * X[n, d]

    # return total

    return X.T.dot((T - Y).dot(V.T) * Z * (1 - Z))


def derivative_b(T, Y, Z, V):
    # N, K = T.shape
    # M = Z.shape[1]

    # total = np.zeros(M)

    # for m in range(M):
    #     for k in range(K):
    #         for n in range(N):
    #             total[m] += (T[n, k] - Y[n, k]) * Z[n, m] * \
    #                 (1 - Z[n, m]) * V[m, k]
    # return total
    return ((T - Y).dot(V.T) * Z * (1 - Z)).sum(axis=0)


def derivative_V(Z, T, Y):
    # N, K = T.shape
    # M = Z.shape[1]

    # total = np.zeros((M, K))

    # for n in range(N):
    #     for m in range(M):
    #         for k in range(K):
    #             total[m, k] += (T[n, k] - Y[n, k]) * Z[n, m]
    # return total
    return Z.T.dot(T - Y)


def derivative_c(T, Y):
    # return np.sum(T - Y) # maybe error
    return (T - Y).sum(axis=0)


def cost(T, Y):
    # return np.sum(T*np.log(Y))
    tot = T * np.log(Y)
    return tot.sum()


if __name__ == '__main__':
    N = 500
    D = 2
    M = 4
    K = 3

    X1 = np.array([-2, 0]) + np.random.randn(N, D)
    X2 = np.array([0, 2]) + np.random.randn(N, D)
    X3 = np.array([2, 0]) + np.random.randn(N, D)
    X = np.vstack([X1, X2, X3])
    Y = np.array([0] * N + [1] * N + [2] * N)

    N = len(Y)
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1

    # print(T)
    # print(Y)

    # plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5)
    # plt.show()

    W = np.random.randn(D, M)
    b = np.random.randn(M)
    V = np.random.randn(M, K)
    c = np.random.randn(K)

    # p_Y = predict(X, W, b, V, c)

    # print(p_Y)
    # print(classification_rate(Y, p_Y))
    epochs = 5000
    learning_rate = 1e-3
    costs = []

    for epoch in range(epochs):
        pY, Z = feedforward(X, W, b, V, c)

        if not epoch % 100:
            pY_given_X = np.argmax(pY, axis=1)
            iter_cost = cost(T, pY)
            print('Cost: ', iter_cost, ' with classification rate: ',
                  classification_rate(Y, pY_given_X))
            costs.append(iter_cost)
        
        gV = derivative_V(Z, T, pY)
        gc = derivative_c(T, pY)
        gW = derivative_W(T, pY, Z, V, X)
        gb = derivative_b(T, pY, Z, V)

        V += learning_rate * gV
        c += learning_rate * gc
        W += learning_rate * gW
        b += learning_rate * gb

    plt.plot(costs)
    plt.show()
