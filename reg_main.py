import neural_util as ut
import numpy as np
import gradient_descent as gd
import logistical_regression as lr
import data as dat


# Main to house regularization testing.
def main():
    # Load the training img and labels
    train_images, train_labels = dat.getTrainingData([2,3], [1, 0], 0, 20000)

    # Load the testing img and labels
    test_images, test_labels = dat.getTestingData([2,3], [1, 0], -2000, None)

    #1-pad the images
    train_images = ut.padOnes(train_images)
    test_images = ut.padOnes(test_images)
    
    # initiate logistical regression with cross entropy
    logreg = lr.LogReg(train_images.shape[1])

    # Number of iterations (Upper bound before holdout)
    itera = 1000
    n0 = .001
    T = 100

    # Should we plot the errors
    isLog = False

    # Regularization constant. Set to zero for normal batch gradient descent.
    regConst = 0.01

    # Norm used for regularization 1 or 2 only.
    normReg = 2

    # Gradient Descent
    finalWeights = gd.gradient_descent(
        train_images, # Images trained on
        train_labels, # Correct labels
        logreg, # Neural Network
        itera, # iterations
        n0, # initial learning rate
        T, # annealing factor
        test_images,
        test_labels,
        regConst,
        normReg,
        isLog
    )

    print "Error Rate: " + str(100 * ut.error_rate2(logreg.run(test_images), test_labels)) + str("%")

if __name__ == '__main__':
    main()