import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.optimize import curve_fit

class data_P_Fc:
	def __init__(self, data_P, data_Fc, data_P_full, data_Fc_full, start, end):
		self.data_P = deepcopy(data_P)
		self.data_Fc = deepcopy(data_Fc)
		self.data_P_full = deepcopy(data_P_full)
		self.data_Fc_full = deepcopy(data_Fc_full)
		self.start = int(start)
		self.end = int(end)
		self.mean_P = 0
		self.std_P = 0
		self.mean_Fc = 0
		self.std_Fc = 0
		self.nb_data = 0
		self.derive_seg = 0
		self.process_data()

	def process_data(self):
		self.mean_P = np.average(self.data_P)
		self.std_P = np.std(self.data_P)
		self.mean_Fc = np.average(self.data_Fc)
		self.std_Fc = np.std(self.data_Fc)
		self.nb_data = len(self.data_P)

def crop_data(data, start, end):
	return data[start:end]

def extract_signal(data_p, data_fc, time_step=180, time_interest=90):
	nb_step = len(data_p) // time_step
	l_roi_step = []
	# print("size data : ", len(data_p))
	# print("nb_step : ", nb_step)
	for i in range(nb_step):
		start = i * time_step + time_interest
		end = i * time_step + time_step
		# print("start : " + str(start) + " end : " + str(end))
		l_roi_step.append(data_P_Fc(data_p[start:end], data_fc[start:end], data_p, data_fc, start, end))

	return l_roi_step

def display(l_data):
	print('step.mean_P' + '|\t' + 'step.mean_Fc' + '|\t' + 'step.std_P' + '|\t' + 'step.std_Fc')
	for step in l_data:
		print("%.2f" % step.mean_P + '|\t' + "%.2f" % step.mean_Fc + '|\t' + "%.2f" % step.std_P + '|\t' + "%.2f" % step.std_Fc)

def create_array_debug(l_step, siz):
	displ_data = np.zeros(siz)
	for step in l_step:
		# print(step.start)
		# print(step.end)
		# print(step.end-step.start)
		idx = np.linspace(step.start, step.end, num=(step.end-step.start))
		for i in idx:
			# print(int(i))
			if i < siz:
				displ_data[int(i)] = 400

	return displ_data

def func_aff(x, a, b):
	return a * x + b

def computer_derive(step):
	x = np.linspace(0, step.nb_data, num=step.nb_data)
	y = step.data_Fc
	popt, pcov = curve_fit(func_aff, x, y)
	plt.plot(x, y, 'b-', label='data')
	plt.plot(x, func_aff(x, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend()
	plt.show()



if __name__ == '__main__':

	filename_fit = '/Volumes/HD/Apeira/Ergocycle/Physio/Data/TestPma/2020-04-02-15-10-05.csv'
	start = 0
	end = 1440
	l_data = []

	df = pd.read_csv(filename_fit, sep=';')
	P = df['Power'].tolist()
	Fc = df['Fc'].tolist()

	P_c = crop_data(P, start, end)
	Fc_c = crop_data(Fc, start, end)

	l_data = extract_signal(P_c, Fc_c, 180, 90)
	debug_data = create_array_debug(l_data, len(P_c))
	display(l_data)

	computer_derive(l_data[4])


	x = np.linspace(0, len(P_c), num=len(P_c))

	plt.figure()
	plt.plot(x, P_c)
	plt.plot(x, Fc_c)
	plt.plot(x, debug_data)
	plt.show()