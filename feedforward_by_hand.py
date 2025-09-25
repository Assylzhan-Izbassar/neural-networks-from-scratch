import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


# dimensions
N = 5  # number of samples
D = 2
M = 3  # number of hidden units of the first layer

X = np.random.randn(N, D)
# print(X)
W = np.random.randn(D, M)  # inner dimensions must match
# print(W)
b = np.random.randn(M)

# test data
# N = 3
# D = 2
# M = 3

# X = [[1, 2],
#      [3, 4],
#      [5, 6]]

# W = [[1, 3, 5],
#      [2, 4, 6]]

# b = [1, 2, 3]

N = 1
D = 2
M = 3

X = [[0, 3.5]]
W = [[0.5, 0.1, -0.3],
     [0.7, -0.3, 0.2]]
b = [0.4, 0.1, 0]

# finding the values of the first hidden layer
# the correct dimensions as N by M for the first hidden layer
Z = np.zeros((N, M))
A = np.zeros((N, M))  # activation function

for i in range(N):
    for k in range(M):
        for j in range(D):
            A[i][k] += X[i][j] * W[j][k]

for k in range(M):
    for i in range(N):
        A[i][k] += b[k]

# calculating the Z
for i in range(N):
    for k in range(M):
        Z[i][k] = tanh(A[i][k])

print(Z)

V = [[0.8, 0.1, -0.1]]
c = 0.2

A_2 = 0

for i in range(N):
    for j in range(M):
        A_2 += Z[i][j] * V[i][j]

A_2 += c

print(A_2)

p_y_given_x = sigmoid(A_2)

print(f'Answer for P(y|x)={p_y_given_x}')
