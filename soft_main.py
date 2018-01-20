import neural_util as ut
import numpy as np
import soft_gradient_descent as sd
import softmax_regression as sr
import data as dat


def main():
    classes = [x for x in range(0, 10)]

    # Load the training img and labels
    train_images, train_labels = dat.getTrainingData(classes, classes, 0, None)

    # Load the testing img and labels
    test_images, test_labels = dat.getTestingData(classes, classes, 0, None)

    #1-pad the images
    train_images = ut.padOnes(train_images)
    test_images = ut.padOnes(test_images)
    train_labels = ut.oneHotEncoding(train_labels)
    test_labels = ut.oneHotEncoding(test_labels)
    
    # initiate logistical regression with cross entropy
    soft = sr.SoftMax(train_images.shape[1], classes)

    # Number of iterations (Upper bound before holdout)
    itera = 1000
    n0 = .001
    T = 100

    # Should we plot the errors
    isLog = True

    # Regularization constant. Set to zero for normal batch gradient descent.
    regConst = 0.0

    # Norm used for regularization 1 or 2 only.
    normReg = 2

    # Gradient Descent
    finalWeights = sd.gradient_descent(
        train_images, # Images trained on
        train_labels, # Correct labels
        classes,
        soft, # Neural Network
        itera, # iterations
        n0, # initial learning rate
        T, # annealing factor
        test_images,
        test_labels,
        regConst,
        normReg,
        isLog
    )

    print "Error Rate: " + str(100 * ut.error_rate2(soft.run(test_images), test_labels)) + str("%")

if __name__ == '__main__':
    main()