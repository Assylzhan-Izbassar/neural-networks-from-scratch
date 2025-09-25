import numpy as np


def softmax(a):  # softmax as getting the 1D array
    # .item() gets float from np.float object
    return [(np.exp(a_i) / sum(np.exp(a))).item() for a_i in a]


def softmax_2D(A):
    return np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)


if __name__ == '__main__':
    # a = np.random.randn(5)
    # print(softmax(a))
    A = np.random.randn(100, 5)
    print(softmax_2D(A))
