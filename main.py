import neural_util as ut
import numpy as np
import gradient_descent as gd
import logistical_regression as lr

from mnist import MNIST

def main():
    # Line might be OS dependent, change based on OS.
    mndata = MNIST('MNIST/')
    mndata.gz = True
    images, labels = mndata.load_training()

    # Only get subset of images
    s_images = images[0:20000]   
    s_labels = labels[0:20000] 

    s_images, s_labels = getTT(s_images, s_labels)

    # Convert images into numpy arrays.
    s_images = np.array(s_images)
    s_labels = np.array(s_labels)

    # initiate logistical regression with cross entropy
    logreg = lr.LogReg(s_images.shape[1])

    # Number of iterations
    itera = 100

    # Gradient Descent
    finalWeights = gd.batch_gradient_descent(
        s_images, # Images trained on
        s_labels, # Correct labels
        logreg, # Neural Network
        itera, # iterations
        1, # Step init
        lambda t, n: n / (1 + t/itera) # Step Function
    )

    finalRes = logreg.run(s_images)
    
    # Round the results
    finalRes = np.clip(np.around(finalRes, decimals=0), 0, 1)
    print finalRes
    print "Error Rate: " + str(100 * error_rate(finalRes, s_labels)) + str("%")
    
# gets error rate of result
def error_rate(res, givenLabel):
    err = 0
    for x in range(0, len(res)):
        if res[x] != givenLabel[x]:
            err += 1
    
    return ((float)(err)) / givenLabel.size

# Get only twos and threes.
def getTT(images, labels):
    resX = []
    resY = []
    for x in range(0, len(labels)):
        if labels[x] == 2:
            resX.append(images[x])
            resY.append(1)
        elif labels[x] == 3:
            resX.append(images[x])
            resY.append(0)

    return resX, resY

if __name__ == '__main__':
    main()