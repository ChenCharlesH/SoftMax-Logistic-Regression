import numpy as np
import neural_util as ut

def main():
	a = ut.oneHotEncoding(np.ones(10))
	b = ut.oneHotEncoding(np.ones(10))
	print a
	print b
	print ut.error_rate3(a,b)

if __name__ == '__main__':
    main()