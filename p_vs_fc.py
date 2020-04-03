import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename_fit = '/Volumes/HD/Apeira/Ergocycle/Physio/Data/TestPma/2020-04-02-15-10-05.csv'

df = pd.read_csv(filename_fit, sep=';')
P = df['Power'].tolist()
Fc = df['Fc'].tolist()

x = np.linspace(0, len(P), num=len(P))

plt.figure()
plt.plot(x, P)
plt.plot(x, Fc)
plt.show()