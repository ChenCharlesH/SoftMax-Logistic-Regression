import numpy as np
import neural_util as ut
import matplotlib.pyplot as plt
import sys

# File to house function for gradient descent.

# Given data matrix
# step given is function with argument t.
#def batch_gradient_descent(dataM, labels, neural, numIter, stepInit, step = lambda t: 1):
def batch_gradient_descent(dataM, labels, neural, numIter, n0, T, test_images, test_labels):
    w = np.zeros(dataM.shape[1])
    neural.w = w
    earlyStop = 0
    minError = 1
    minErrorWeight = 0
    # n = stepInit
    train_images, train_labels, holdout_images, holdout_labels = ut.getHoldout(dataM, labels, 0.1)
    errorTrain = []
    errorHoldout = []
    errorTest = []
    # Loop through data
    for t in range(0, numIter):
        errorOld = ut.error_rate2(neural.run(holdout_images), holdout_labels)
        n = n0/(1+t/float(T))
        w = np.subtract(w,n * gradient(neural.run(train_images), train_labels, train_images))
        neural.w = w
        errorNew = ut.error_rate2(neural.run(holdout_images), holdout_labels)
        errorTrain.append(ut.avg_cross_entropy(neural.run(train_images),train_labels))
        errorHoldout.append(ut.avg_cross_entropy(neural.run(holdout_images),holdout_labels))
        errorTest.append(ut.avg_cross_entropy(neural.run(test_images),test_labels))
        # s = "t="+repr(t)+",n="+repr(n)+",errorOld="+repr(errorOld)+",errorNew="+repr(errorNew)
        # print s
        if(errorNew<minError):
            minError = errorNew
            minErrorWeight = w
        if errorNew > errorOld:
            earlyStop = earlyStop+1
            if earlyStop==3 and t>10:
                break
        else:
            earlyStop=0
    neural.w = minErrorWeight
    w = minErrorWeight
    plt.plot(errorTrain)
    plt.plot(errorHoldout)
    plt.plot(errorTest)
    plt.show()
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

#def updateStep(self, t, T):
#    self.n = self.n / (1 + t/T)