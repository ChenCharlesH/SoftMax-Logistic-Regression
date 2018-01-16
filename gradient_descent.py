import numpy as np

# File to house function for gradient descent.

# Given data matrix
# step given is function with argument t.
def batch_gradient_descent(dataM, labels, neural, numIter, step = lambda t: 1):
    w = np.zeros(dataM.shape[1] + 1)

    # Need to set first weight to 1.
    w[0] = 1
    neural.w = w
    
    # Loop through data
    for t in range(0, numIter):
        w[1:] = np.subtract(w[1:], gradient(neural.run(dataM), labels, dataM))
        neural.w = w
        print t

    return w

# Calculate gradient vector
def gradient(Y, T, X):
    colNum = X.shape[1]
    rowNum = X.shape[0]

    diff = np.subtract(T, Y)
    res = np.zeros(shape=(colNum, ))
    for i in range(0, rowNum):
        res = np.add(np.multiply(X[i, :], diff[i]), res)

    return res

def updateWeights(self):
    pass

def updateStep(self, t, T):
    self.n = self.n / (1 + t/T)
