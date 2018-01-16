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

    # Constructor
    def __init__(self, dim):
        self.DIM = dim;
        self.w = np.zeros(self.DIM)
    
    # Runs the neural net and output result
    # saves last result in self.out
    def run(self, dataM):
        # Multiply each data with each weight.
        r = np.dot(dataM, self.w).flatten()

        # apply sigmoid
        r = ut.vect_sig(r)
        self.out = r
        return r

