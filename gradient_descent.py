import numpy as np
import sys

# File to house function for gradient descent.

# Given data matrix
# step given is function with argument t.
def batch_gradient_descent(dataM, labels, neural, numIter, stepInit, step = lambda t: 1):
    w = np.zeros(dataM.shape[1] + 1)

    # Need to set first weight to 1.
    w[0] = 1
    neural.w = w
    
    n = stepInit
    # Loop through data
    for t in range(0, numIter):
        n = step(t, n)
        w[1:] = np.subtract(w[1:],gradient(neural.run(dataM), labels, dataM))
        neural.w = w

    return w

# Calculate gradient vector
def gradient(Y, T, X):
    rowNum = X.shape[0]
    colNum = X.shape[1]

    diff = np.subtract(Y, T)
    res = np.zeros(shape=(colNum, ))
    for i in range(0, rowNum):
        # Clip the results lest we overflow
        np.add(np.multiply(X[i, :], diff[i]), res, res)

    return res

def updateStep(self, t, T):
    self.n = self.n / (1 + t/T)
