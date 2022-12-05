from __future__ import division
import numpy as np

# Softmax
class SoftmaxFunc():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

# ReLU activation function
class ReLU():
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)

#Loss 
class LossFunc(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

#Cross Entropy
class CrossEntropyFunc(LossFunc):
    def __init__(self): pass

    def loss(self, x, res):
        res = np.clip(res, 1e-15, 1 - 1e-15)
        return - x * np.log(res) - (1 - x) * np.log(1 - res)

    def acc(self, x, res):
        return accuracy(np.argmax(x, axis=1), np.argmax(res, axis=1))

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

# Calculate Accuracy
def accuracy(y_true, y_pred):
    accuracy_val = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy_val
