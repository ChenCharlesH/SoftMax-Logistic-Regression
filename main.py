import logistical_regression as lg
import util as ut
import numpy as np

from mnist import MNIST

def main():
    # Line might be OS dependent, change based on OS.
    mndata = MNIST('MNIST/')
    mndata.gz = True
    images, labels = mndata.load_training()

    # Convert images into numpy arrays.
    images = np.array(images)
    lables = np.array(labels)

    # initiate logistical regression with cross entropy
    logreg = lg.LogReg(ut.cross_entropy)



if __name__ == '__main__':
    main()