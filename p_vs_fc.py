import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class data_P_Fc:
	def __init__(self, data_P, data_Fc, start, end):
		self.data_P = deepcopy(data_P)
		self.data_Fc = deepcopy(data_Fc)
		self.start = start
		self.end = end
		self.mean_P = 0
		self.std_P = 0
		self.mean_Fc = 0
		self.std_Fc = 0
		self.nb_data = 0
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
	print("size data : ", len(data_p))
	print("nb_step : ", nb_step)
	for i in range(nb_step):
		start = i * time_step + time_interest
		end = start + time_step
		l_roi_step.append(data_P_Fc(data_p[start:end], data_fc[start:end], start, end))

	return l_roi_step

def display(l_data):
	print('step.mean_P' + '|\t' + 'step.mean_Fc' + '|\t' + 'step.mean_P' + '|\t' + 'step.mean_Fc')
	for step in l_data:
		print("%.2f" % step.mean_P + '|\t' + "%.2f" % step.mean_Fc + '|\t' + "%.2f" % step.std_P + '|\t' + "%.2f" % step.std_Fc)

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
	display(l_data)

	# TODO : display all start and stop sur le graphique

	x = np.linspace(0, len(P_c), num=len(P_c))

	# plt.figure()
	# plt.plot(x, P_c)
	# plt.plot(x, Fc_c)
	# plt.show()