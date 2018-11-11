import pint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties as ufloat
ureg = pint.UnitRegistry()

#Zur Berechnung der linearen Ausgleichsrechnung der Energieeichung
def linear(x, m, b):
    return m*x + b

#Berechnung der Aktivität am Datum 12.11.2018
def activity(n0, t12):
    t = 8*365+21+12 * ureg.seconds
    lam = np.ln(2)/ t12
    return n0*exp(-lam*t)


#Ausgleichsrechnung der Energieeichung
E, Emw = np.genfromtxt(data/Eu-spectrum.txt)
x = np.linspace(120, 1500)
plt.plot(E, Emw, 'rx', label='152^Eu')
print('Ausgleichsrechnung zur Energieeichung des Spektrometer des Eu-Spektrums')
params, cov = curve_fit(f=linear, xdata=E, ydata=Emw)
errors = np.sqrt(np.diag(cov))
plt.plot(x, linear(x), 'k-', label='Ausgleichsrechnung')
plt.xlabel(r'E_{\gamma} / keV')
plt.ylabel(r'Emissionswahr. / \%')
plt.savefig(build/Eu-gaugespektrum.pdf)
print('m = ', params[0], '+/-', errors[0])
print('b = ', params[1], '+/-', errors[1])

#Aktivität am Messtag
becquerel = 1 / second
tH = ufloat(4943, 5) * days
n0 = ufloat(4130, 60) * ureg.becquerel
print('Berechnete Aktivität der Eu-Probe am Messtag 12.11.2018')
print('A(t) = ', activity(n0, tH))


#Berechnung des Raumwinkels
