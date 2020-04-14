import numpy as np


def R_2 (y, yi):
	yi_mean = np.mean(yi)

	num = np.sum(np.power(y - yi, 2))
	denum = np.sum(np.power(y - yi_mean, 2))

	R2 = 1 - num/denum

	return R2