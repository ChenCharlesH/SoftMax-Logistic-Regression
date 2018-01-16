import numpy as np
import matplotlib.pyplot as plt
import math

# File to hold all helper functions.

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
    return np.array(resX), np.array(resY)

# Get subset of images belonging to the array passed
def getSubset(images, labels, values):
    resX = []
    resY = []
    for i in range(0, len(values)):
        resX.append(images[labels==values[i]])
        resY.append(i)
    return np.array(resX), np.array(resY)

def getHoldout(images, labels, fraction):
	s = images.shape[0]
	randomVal = np.random.rand(s)
	idx = randomVal<=fraction
	holdout_images = images[idx]
	holdout_labels = labels[idx]
	idx = randomVal>fraction
	train_images = images[idx]
	train_labels = labels[idx]
	return train_images, train_labels, holdout_images, holdout_labels

# 1-pad input data 
def padOnes(images):
	s = images.shape
	res = np.ones(shape=(s[0], s[1] + 1))
	res[:, 1:] = images
	return res

# Plot grayscale image
def showImg(image):
	image = image[1:]
	image = np.array(image, dtype='uint8')
	image = image.reshape((28, 28))
	plt.imshow(image, cmap="gray")
	plt.show()

def cross_entropy(T, Y):
    res = 0
    for x in range(0, T.size):
        res += T[x] * math.log(Y[x]) + ((1 - T[x]) * math.log(1-Y[x]))
    return -res

def avg_cross_entropy(T, Y):
    return cross_entropy(T, Y) / T.shape[0]

# Clipped values due to overflow
def sig(x):
    return 1 / (1 + math.exp(-np.clip(x, -500, 500)))

# gets error rate of result
def error_rate(res, givenLabel):
    err = 0
    for x in range(0, len(res)):
        if res[x] != givenLabel[x]:
            err += 1
    
    return ((float)(err)) / givenLabel.size

vect_sig = np.vectorize(sig)