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
    s_images = images[0:20000] + images[-2000:]   
    s_labels = labels[0:20000] + labels[-2000:]   

    s_images, s_labels = getTT(s_images, s_labels)

    # Convert images into numpy arrays.
    s_images = np.array(s_images)
    s_labels = np.array(s_labels)

    # initiate logistical regression with cross entropy
    logreg = lr.LogReg(s_images.shape[1])

    # Gradient vector
    gd.batch_gradient_descent(s_images, s_labels, logreg, 4)

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