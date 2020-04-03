import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def crop_data(data, start, end):
	return data[start, end]

if name == '__name__' :

	filename_fit = '/Volumes/HD/Apeira/Ergocycle/Physio/Data/TestPma/2020-04-02-15-10-05.csv'
	start = 0
	end = 1440
	df = pd.read_csv(filename_fit, sep=';')
	P = df['Power'].tolist()
	Fc = df['Fc'].tolist()

	P_c = crop_data(P, start, end)


	x = np.linspace(0, len(P), num=len(P))

	plt.figure()
	plt.plot(x, P)
	plt.plot(x, Fc)
	plt.show()