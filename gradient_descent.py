import numpy as np
import neural_util as ut
import matplotlib.pyplot as plt
import sys

# File to house function for gradient descent.

# Given data matrix
# step given is function with argument t.
# Uses mini-batch gradient descent.
#def batch_gradient_descent(dataM, labels, neural, numIter, stepInit, step = lambda t: 1):
def gradient_descent(dataM, labels, neural, numIter, n0, T, test_images, test_labels, regConst, normNum, isLog):
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
        numMini=1000
        # Caluclate the size of the minibatch.
        size = train_images.shape[0]/numMini; 
        # Grab the error rate before training.
        errorOld = ut.error_rate2(neural.run(holdout_images), holdout_labels)

        # Learning Rate
        n = n0/(1+t/float(T))
        # minibatch value.
        for k in range(0,numMini):
            # Generate batches for mini-batch.
            batch_train_images = train_images[k*size : (k+1) * size, :]
            batch_train_size = batch_train_images.shape[0]
            batch_train_labels = train_labels[k*size : (k+1) * size]

            # Get our output results y
            neural_result = neural.run(batch_train_images)

            # Regularization
            # TODO: Fix the last untrained values if size % batchSize != 0.
            J = gradient(neural_result, batch_train_labels, batch_train_images) + regConst * lx_gradient(w, batch_train_size, normNum)

            # Our gradient value.
            w = np.subtract(w, n * J / numMini)
            # Update the weights
            neural.w = w

            # Log data.
            if isLog:
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
    if isLog:
        plt.plot(errorTrain,label = 'Training',linewidth=0.8)
        plt.plot(errorHoldout, label = 'Holdout',linewidth=0.8)
        plt.plot(errorTest, label = 'Test',linewidth=0.8)
        plt.legend()
        plt.show()
    return w

# l-norms helper function
def lx_gradient(w, batch_size, t):
    if t == 1:
        return l1_gradient(w, batch_size)
    return l2_gradient(w, batch_size)

# Calculate the l2 gradient vector
def l1_gradient(w, batch_size):
    res = np.zeros(w.shape)
    normalized = ut.vect_d_abs(w)
    for i in range(0, batch_size):
        np.add(normalized, res, res)
    
    return res


# Calculate the l1 gradient vector.
def l2_gradient(w, batch_size):
    res = np.zeros(w.shape)
    # Compound out weights for the number of iterations required.
    for i in range(0, batch_size):
        np.add(2 * w, res, res)
    
    return res


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