import neural_util as ut
import numpy as np
import gradient_descent as gd
import logistical_regression as lr
import matplotlib.pyplot as plt
import data as dat
import math


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

    # Norms to test
    norms = [1000, 100, 10, 1, 0, 0.1, 0.01, 0.001, 0.0001]
    
    for normReg in [1, 2]:
        errResults = []
        corrResults = []
        weightMag = []
        weightHist = []
        for regConst in norms:

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
            
            # Get the weight of the neural network
            weightMag.append(np.linalg.norm(logreg.w))

            # Percentage of error printout.
            print "Lambda: " + str(regConst) + " Norm: " + str(normReg)

            # choose either training or testing
            # err = 100 * ut.error_rate2(logreg.run(train_images), train_labels)
            err = 100 * ut.error_rate2(logreg.run(test_images), test_labels)

            errResults.append(err)
            corrResults.append(100-err)
            weightHist.append(np.copy(logreg.w))
            print "Error Rate: " + str(err) + str("%")
        
        plt.plot([math.log(n,10) if n!= 0 else 0 for n in norms], corrResults, label = "lambda")
        plt.legend()
        plt.show()

        # Save the weights as images.
        weights_as_images(weightHist, "rl"+str(normReg)+"n"+str(regConst)+".jpg")

# Save weight vectors as images
def weights_as_images(weights, filename):
    # Reshape the weights back to 28 by 28.
    reshaped = [w[1:].reshape(28,28) for w in weights]
    rows = 2
    cols = len(weights) / rows
    f, axarr = plt.subplots(rows, cols)
    for r in range(0, rows):
        for c in range(0, cols):
            axarr[r, c].imshow(reshaped[r * cols +c], cmap="gray")
    
    plt.show()


if __name__ == '__main__':
    main()