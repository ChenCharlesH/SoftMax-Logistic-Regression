import neural_util as ut
import numpy as np
import gradient_descent as gd
import logistical_regression as lr

from mnist import MNIST

def main():
    # Line might be OS dependent, change based on OS.
    mndata = MNIST('MNIST/')
    mndata.gz = True
    
    # Only get subset of images and select a subset
    train_images, train_labels = mndata.load_training()
    train_images = np.array(train_images[0:20000])   
    train_labels = np.array(train_labels[0:20000]) 
    train_images, train_labels = ut.getTT(train_images, train_labels)

    # Do the same for test set
    test_images, test_labels = mndata.load_testing()
    test_images = np.array(test_images[-2000:])
    test_labels = np.array(test_labels[-2000:])
    test_images, test_labels = ut.getTT(test_images, test_labels)

    #1-pad the images
    train_images = ut.padOnes(train_images)
    test_images = ut.padOnes(test_images)
    
    # initiate logistical regression with cross entropy
    logreg = lr.LogReg(train_images.shape[1])

    # Number of iterations
    itera = 100
    n0 = 0.01
    T = 100

    # Gradient Descent
    # finalWeights = gd.batch_gradient_descent(
    #     train_images, # Images trained on
    #     train_labels, # Correct labels
    #     logreg, # Neural Network
    #     itera, # iterations
    #     1, # Step init
    #     lambda t, n: n / (1 + t/itera) # Step Function
    # )
    finalWeights = gd.batch_gradient_descent(
        train_images, # Images trained on
        train_labels, # Correct labels
        logreg, # Neural Network
        itera, # iterations
        n0, # initial learning rate
        T # annealing factor
    )
    ut.showImg(finalWeights)
    print finalWeights[1]

    finalRes = logreg.run(test_images)
    
    # Round the results
    finalRes = np.clip(np.around(finalRes, decimals=0), 0, 1)
    print "Error Rate: " + str(100 * ut.error_rate(finalRes, test_labels)) + str("%")

if __name__ == '__main__':
    main()