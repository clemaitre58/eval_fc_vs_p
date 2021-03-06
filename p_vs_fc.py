import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.optimize import curve_fit

from utls import eval_model

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
		self.R2_mod_exp = 0
		self.R2_mod_sigmo = 0
		self.param_fit = []
		self.param_fit_exp = []
		self.param_fit_sigmo = []
		self.param_fit_exp_controle_vitesse = []
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

def mod_sigmo(x, a, offset, speed ,delay):
	y = a * (1 / 2 + 1 / 2 * np.tanh((x - delay) / speed)) + offset

	return y

def exp_controle_vitesse(x, a, speed, offset, delay ):
	y = (a / (1 + np.exp(- speed * (x - delay)))) + offset

	return y

def compute_exp_controle_vitesse(step):
	first_val = step.data_Fc_full[0]
	last_val = step.data_Fc_full[len(step.data_Fc_full) - 1]
	low_a = (last_val - first_val) * 0.9
	high_a = (last_val - first_val) * 1.1
	low_offset = first_val * 0.9
	high_offset = first_val * 1.1
	x = np.linspace(0, len(step.data_Fc_full)-1, num=len(step.data_Fc_full))
	y = step.data_Fc_full
	x = [int(elt) for elt in x]
	y_np = np.array(y)
	popt, pcov = curve_fit(exp_controle_vitesse, x, y_np, bounds=([low_a, 0.1, low_offset, 35], [high_a, 0.2, high_offset, 80]))

	step.param_fit_exp_controle_vitesse = popt[:]

def display_exp_controle_vitesse(step):
	x = np.linspace(0, len(step.data_Fc_full), num=len(step.data_Fc_full))
	y = step.data_Fc_full
	plt.figure()
	plt.plot(x, y, 'o')
	plt.plot(x, exp_controle_vitesse(x, *step.param_fit_exp_controle_vitesse), 'r-')
	plt.show()

def compute_R2_mod_exp(step):
	x = np.linspace(0, len(step.data_Fc_full), num=len(step.data_Fc_full))
	y = step.data_Fc_full
	yi = exp_controle_vitesse(x, step.param_fit_exp_controle_vitesse[0], 
	step.param_fit_exp_controle_vitesse[1], step.param_fit_exp_controle_vitesse[2], step.param_fit_exp_controle_vitesse[3])
	step.R2_mod_exp = eval_model.R_2(y, yi)

def compute_mod_sigmo(step):
	first_val = step.data_Fc_full[0]
	last_val = step.data_Fc_full[len(step.data_Fc_full) - 1]
	low_a = (last_val - first_val) * 0.9
	high_a = (last_val - first_val) * 1.1
	low_offset = first_val * 0.9
	high_offset = first_val * 1.1
	x = np.linspace(0, len(step.data_Fc_full)-1, num=len(step.data_Fc_full))
	y = step.data_Fc_full
	x = [int(elt) for elt in x]
	y_np = np.array(y)
	popt, pcov = curve_fit(mod_sigmo, x, y_np, bounds=([low_a, low_offset, -np.inf, 35], [high_a, high_offset, np.inf, 80]))

	step.param_fit_sigmo = popt[:]

def display_mod_sigmo(step):
	x = np.linspace(0, len(step.data_Fc_full), num=len(step.data_Fc_full))
	y = step.data_Fc_full
	plt.figure()
	plt.plot(x, y, 'o')
	plt.plot(x, mod_sigmo(x, *step.param_fit_sigmo), 'r-')
	plt.show()

def compute_R2_mod_sigmo(step):
	x = np.linspace(0, len(step.data_Fc_full), num=len(step.data_Fc_full))
	y = step.data_Fc_full
	yi = mod_sigmo(x, step.param_fit_sigmo[0], step.param_fit_sigmo[1], step.param_fit_sigmo[2], step.param_fit_sigmo[3])
	step.R2_mod_sigmo = eval_model.R_2(y, yi)

def mod_charge_exp(x, p, tho, b):
	a = p - b
	y = (a * (1 - np.exp(- x / tho))) + b

	return y

def compute_mod_exp(step):
	x = np.linspace(0, len(step.data_Fc_full)-1, num=len(step.data_Fc_full))
	y = step.data_Fc_full
	x = [int(elt) for elt in x]
	y_np = np.array(y)
	popt, pcov = curve_fit(mod_charge_exp, x, y_np, bounds=([161, 45., 152], [166., 55., 156]))

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

	plt.close('all')
	filename_fit = '/Volumes/HD/Apeira/Ergocycle/Physio/Data/TestPma/2020-04-02-15-10-05.csv'
	# filename_fit = '/Volumes/HD/Apeira/Ergocycle/Physio/Data/TestPma/ACurtil/4746600763.csv'
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

	# compute_mod_exp(l_data[4])
	# display_mod_exp(l_data[4])

	ind_2_test = 7

	compute_mod_sigmo(l_data[ind_2_test])
	display_mod_sigmo(l_data[ind_2_test])
	compute_R2_mod_sigmo(l_data[ind_2_test])
	print('R^2 pour modele sigmo : ', l_data[ind_2_test].R2_mod_sigmo)

	compute_exp_controle_vitesse(l_data[ind_2_test])
	display_exp_controle_vitesse(l_data[ind_2_test])
	compute_R2_mod_exp(l_data[ind_2_test])
	print('R^2 pour modele logistique : ', l_data[ind_2_test].R2_mod_exp)

	x = np.linspace(0, len(P_c), num=len(P_c))

	plt.figure()
	plt.plot(x, P_c)
	plt.plot(x, Fc_c)
	plt.plot(x, debug_data)
	plt.show()