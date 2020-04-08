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
		self.param_fit = []
		self.param_fit_exp = []
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
		l_roi_step.append(data_P_Fc(data_p[start:end], data_fc[start:end], data_p[i * time_step:i * time_step + time_step], data_fc[i * time_step:i * time_step + time_step], start, end))

	return l_roi_step

def print_stat_step(l_data):
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

def mod_charge_exp(x, p, tho, b):
	a = p - b
	y = (a * (1 - np.exp(- x / tho))) + b

	return y

def compute_mod_exp(step):
	x = np.linspace(0, len(step.data_Fc_full)-1, num=len(step.data_Fc_full))
	y = step.data_Fc_full
	x = [int(elt) for elt in x]
	y_np = np.array(y)
	# mini = np.min(y_np)
	# y_np = y_np - mini
	# y_np = y_np / np.max(y_np)  
	print("Valeurs dans le tableau x : ", x)
	print("Valeurs dans le tableau y : ", y)
	print("Nombre de valeurs dans le tableau y : ", len(y))
	print("Nombre de valeurs dans le tableau x : ", len(x))
	popt, pcov = curve_fit(mod_charge_exp, x, y_np, bounds=([161, 45., 150], [166., 75, 156]))

	step.param_fit_exp = popt[:]

def display_mod_exp(step):
	x = np.linspace(0, len(step.data_Fc_full), num=len(step.data_Fc_full))
	y = step.data_Fc_full
	plt.figure()
	plt.plot(x, y, 'o')
	plt.plot(x, mod_charge_exp(x, *step.param_fit_exp), 'r-')
	plt.show()


def compute_derive(step):
	x = np.linspace(0, step.nb_data, num=step.nb_data)
	y = step.data_Fc
	popt, pcov = curve_fit(func_aff, x, y)

	step.param_fit = popt[:]

def computer_derive_all(l_step):
	for step in l_step:
		compute_derive(step)

def plot_all_derive(l_step):
	cpt = 0
	fig, axs = plt.subplots(len(l_step), 1, constrained_layout=True)
	for step in l_step:
		x = np.linspace(0, step.nb_data, num=step.nb_data)
		y = step.data_Fc

		axs[cpt].plot(x, y, 'o')
		axs[cpt].plot(x, func_aff(x, *step.param_fit), 'r-')
		# axs[cpt].set_title('Palier : ' + str(cpt))
		# axs[cpt].set_xlabel('Temps de le palier (s)')
		# axs[cpt].set_ylabel('Fc')
		cpt += 1
	plt.legend()
	plt.show()

def print_derive(l_step):
	print('Valeur derive')
	for step in l_step:
		print("%.4f" % step.param_fit[0])

def print_derive_moyenne(l_step):
	print('Valeur derive moyenne (augmenation par minute)')
	l_derive = []
	for step in l_step:
		l_derive.append(step.param_fit[0])

	derive_moy = np.mean(l_derive) * 60
	print("%.4f" % derive_moy)


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
	print_stat_step(l_data)

	computer_derive_all(l_data)
	print_derive(l_data)
	print_derive_moyenne(l_data)
	plot_all_derive(l_data)

	compute_mod_exp(l_data[4])
	display_mod_exp(l_data[4])


	x = np.linspace(0, len(P_c), num=len(P_c))

	plt.figure()
	plt.plot(x, P_c)
	plt.plot(x, Fc_c)
	plt.plot(x, debug_data)
	plt.show()