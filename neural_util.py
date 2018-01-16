import numpy as np

import math

# File to hold all helper functions.

def cross_entropy(T, Y):
    res = 0
    for x in range(0, t.size):
        res += T[x] * math.log(Y[x]) + ((1 - T[x]) * math.log(1-Y[x]))
    return -res

def avg_cross_entropy(T, Y):
    return cross_entropy(T, Y) / T.shape[0]

def sig_arr(x):
    return 1.0 / (1 + math.exp(-x[0]))