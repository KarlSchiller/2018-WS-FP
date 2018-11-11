import pint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Zur Berechnung der linearen Ausgleichsrechnung der Energieeichung
def linear(x, a, b):
    return a*x + b

#Berechnung der Aktivität am Datum 12.11.2018
def exp(n0, t12):
    t = 8*365+21+12
    lam = np.ln(2)/ t12
    return n0*exp(-lam*t)

E, Emw = np.genfromtxt(data/Eu-spectrum.txt)
plt.plot(E, Emw, label='152^Eu')
