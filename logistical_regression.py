import numpy as np
import neural_util as ut

# Class to house logistical regression.
# Evaluation functions should be with parameters Y, T
# as defined in util.py
class LogReg:
    # Output Vector of Previous Run
    out = np.array([])

    # Weight vector
    w = np.array([])

    # Requires a set evaluation function.
    def __init__(self, dim):
        self.DIM = dim + 1
        self.w = np.zeros(self.DIM)
        self.w[0] = 1
    
    def run(self, dataM):
        s = dataM.shape
        dataC = np.zeros(shape=(s[0], s[1] + 1))
        dataC.fill(1)
        dataC[:, 1:] = dataM

        # Multiply each data with each weight.
        r = np.dot(dataC, self.w).flatten()

        # apply sigmoid
        r = ut.vect_sig(r)
        self.out = r
        return r

