import numpy as np

X = [[1, 2],
     [3, 4],
     [5, 6]]

W = [[1, 3, 5],
     [2, 4, 6]]

A = [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]]

N = 3
D = 2
M = 3

for i in range(N):
    for k in range(M):
        for j in range(D): # inner dimension
            A[i][k] += X[i][j] * W[j][k]

for i in range(N):
    for j in range(M):
        print(A[i][j], end=' ')
    print()

X = np.array(X)
W = np.array(W)

A = X.dot(W)

print(A)
