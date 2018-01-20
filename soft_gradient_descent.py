import numpy as np
import neural_util as ut
import matplotlib.pyplot as plt
import sys

# File to house function for gradient descent.

# Given data matrix
# step given is function with argument t.
# Uses mini-batch gradient descent.
#def batch_gradient_descent(dataM, labels, neural, numIter, stepInit, step = lambda t: 1):
def gradient_descent(dataM, labels, classes, neural, numIter, n0, T, test_images, test_labels, regConst, normNum, isLog):
    # Set up weights.
    W = np.zeros((dataM.shape[1],10))
    neural.W = W

    # Variable to keep track of how many epochs we have stopped
    earlyStop = 0

    # Minimum error
    minError = 1
    minErrorWeight = []

    # n = stepInit
    train_images, train_labels, holdout_images, holdout_labels = ut.getHoldout(dataM, labels, 0.1)

    # Collection data attributes
    errorTrain = []
    errorHoldout = []
    errorTest = []
    entropyTrain = []
    entropyHoldout = []
    entropyTest = []

    # Loop through data
    for t in range(0, numIter):
        errorOld = ut.error_rate3(neural.run(holdout_images), holdout_labels)
        n = n0/(1+t/float(T))
        W = W - n * gradient2(neural.run(train_images),train_labels, train_images) - n * regConst * lx_gradient(W, normNum)
        neural.W = W
        errorNew = ut.error_rate3(neural.run(holdout_images), holdout_labels)
        print errorNew

        # Logic to detect if we should stop due to hold-out.
        if errorNew < minError:
            minError = errorNew
            minErrorWeight = W
        if errorNew > errorOld:
            earlyStop = earlyStop+1
            if earlyStop==15 and t>10:
                break
        else:
            earlyStop = 0

        if isLog:
                errorTrain.append(ut.error_rate3(neural.run(train_images),train_labels))
                errorHoldout.append(ut.error_rate3(neural.run(holdout_images),holdout_labels))
                errorTest.append(ut.error_rate3(neural.run(test_images),test_labels))
                entropyTrain.append(ut.k_avg_entropy(neural.run(train_images),train_labels))
                entropyHoldout.append(ut.k_avg_entropy(neural.run(holdout_images),holdout_labels))
                entropyTest.append(ut.k_avg_entropy(neural.run(test_images),test_labels))
       


    # Grab the minimum error weight.
    neural.W = minErrorWeight
    W = minErrorWeight

    # Plot our data.
    if isLog:
        plt.plot(errorTrain,label = 'Training')
        plt.plot(errorHoldout, label = 'Holdout')
        plt.plot(errorTest, label = 'Test')
        plt.title("Error Rate during Training")
        plt.legend()
        plt.show()
        plt.plot(entropyTrain,label = 'Training')
        plt.plot(entropyHoldout, label = 'Holdout')
        plt.plot(entropyTest, label = 'Test')
        plt.title("Cross Entropy during Training")
        plt.legend()
        plt.show()
    weights_as_images(np.transpose(W))
    return W

# l-norms helper function
def lx_gradient(w, t):
    if t == 1:
        return l1_gradient(w)
    return l2_gradient(w)

def weights_as_images(weights):
    # Reshape the weights back to 28 by 28.
    reshaped = [w[1:].reshape(28,28) for w in weights]
    rows = 2
    cols = len(weights) / rows
    f, axarr = plt.subplots(rows, cols)
    for r in range(0, rows):
        for c in range(0, cols):
            axarr[r, c].imshow(reshaped[r * cols +c], cmap="gray")
            # plt.axis('off')
    
    plt.show()

# Calculate the l2 gradient vector
def l1_gradient(w):
    res = np.sign(w)
    return res


# Calculate the l1 gradient vector.
def l2_gradient(w):
    res = 2 * w
    return res


# Calculate gradient vector
def gradient(Y, X, T, c):
    rowNum = X.shape[0]
    colNum = X.shape[1]

    diff = np.subtract(Y, T[:, c])
    res = np.zeros(shape=(colNum, ))
    for i in range(0, rowNum):
        # Clip the results lest we overflow
        np.add(np.multiply(X[i, :], diff[i]), res, res)
    
    return res

def gradient2(Y,T,X):
    res = np.matmul(np.transpose(X),np.subtract(Y,T))
    return res

#def updateStep(self, t, T):
#    self.n = self.n / (1 + t/T)