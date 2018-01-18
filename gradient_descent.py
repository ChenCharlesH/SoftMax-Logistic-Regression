import numpy as np
import neural_util as ut
import matplotlib.pyplot as plt
import sys

# File to house function for gradient descent.

# Given data matrix
# step given is function with argument t.
#def batch_gradient_descent(dataM, labels, neural, numIter, stepInit, step = lambda t: 1):
def batch_gradient_descent(dataM, labels, neural, numIter, n0, T, test_images, test_labels):
    # Set up weights.
    w = np.zeros(dataM.shape[1])
    neural.w = w

    # Variable to keep track of how many epochs we have stopped
    earlyStop = 0

    # Minimum error
    minError = 1
    minErrorWeight = 0

    # n = stepInit
    train_images, train_labels, holdout_images, holdout_labels = ut.getHoldout(dataM, labels, 0.1)

    # Collection data attributes
    errorTrain = []
    errorHoldout = []
    errorTest = []

    # Loop through data
    for t in range(0, numIter):
        numMini=10
        # Caluclate the size of the minibatch.
        size = train_images.shape[0]/numMini; 
        # Grab the error rate before training.
        errorOld = ut.error_rate2(neural.run(holdout_images), holdout_labels)

        # Step size.
        n = n0/(1+t/float(T))

        # minibatch value.
        for k in range(0,numMini):
            # Our gradient value.
            w = np.subtract(w,0.2 * n * gradient(neural.run(train_images[k*size:(k+1)*size,:]), train_labels[k*size:(k+1)*size], train_images[k*size:(k+1)*size,:]))
            # Update the weights
            neural.w = w

            # Log data.
            errorTrain.append(ut.avg_cross_entropy(neural.run(train_images),train_labels))
            errorHoldout.append(ut.avg_cross_entropy(neural.run(holdout_images),holdout_labels))
            errorTest.append(ut.avg_cross_entropy(neural.run(test_images),test_labels))
       
        # Calculate the error rate after training.
        errorNew = ut.error_rate2(neural.run(holdout_images), holdout_labels)

        # Logic to detect if we should stop due to hold-out.
        if errorNew < minError:
            minError = errorNew
            minErrorWeight = w
        if errorNew > errorOld:
            earlyStop = earlyStop+1
            if earlyStop==3 and t>10:
                break
        else:
            earlyStop = 0

    # Grab the minimum error weight.
    neural.w = minErrorWeight
    w = minErrorWeight

    # Plot our data.
    plt.plot(errorTrain,label = 'Training')
    plt.plot(errorHoldout, label = 'Holdout')
    plt.plot(errorTest, label = 'Test')
    plt.legend()
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