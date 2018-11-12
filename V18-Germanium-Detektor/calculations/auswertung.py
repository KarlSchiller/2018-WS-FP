import os
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
import matplotlib.pyplot as plt
import datetime
from astropy.io import ascii
from scipy.optimize import curve_fit
import scipy.constants as codata
from tab2tex import make_table
import pint
from scipy.signal import find_peaks
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

#-------------Aufgabenteil a)
#
#Zur Berechnung der linearen Ausgleichsrechnung der Energieeichung
def linear(x, m, b):
    return m*x + b

def eichung():
    print('Eichmessung mit der Eu-Probe')
    ursprung_aktivitaet = Q_(ufloat(4130, 60), 'becquerel')
    ursprung_aktivitaet_messtag = datetime.date(year=2000, month=10, day=1)
    praktikumstag = datetime.date(year=2018, month=11, day=12)
    zeitdelta = Q_((praktikumstag - ursprung_aktivitaet_messtag).days, 'day')
    eu_halbwertszeit = Q_(ufloat(4943, 5), 'day')
    eu_zerfallskonstante = np.log(2) / eu_halbwertszeit
    eu_aktivitaet = ursprung_aktivitaet * np.exp(1)**(-eu_zerfallskonstante*zeitdelta)
    print(f'Die ursprüngliche Aktivität am {ursprung_aktivitaet_messtag} beträgt {ursprung_aktivitaet}')
    print(f'Zwischen dem Praktikumstag und der Aktivitätenmessung sind {zeitdelta} vergangen.')
    print(f'Die Aktivität der Eu-Probe mit einer Zerfallskonstanten von {eu_zerfallskonstante :e} beträgt {eu_aktivitaet}')

    #finde die peaks
    counts = np.genfromtxt("data/Eu.txt", unpack=True)
    peaks = find_peaks(counts, height=5, distance=10)
    #print(peaks)
    indexes = peaks[0]
    #peak_heights = peaks[1]
    print(peak_heights)
    #Ausgleichsrechnung der Energieeichung
    #plt.plot(indexes, peak_heights, 'b.')
    #print('Länge von x_plot', x_plot)
    print('Anzahl an Kanälen', len(counts))
    plt.hist(counts,bins=4050, range=(0,8191))

    plt.yscale('log')
    plt.savefig('build/Eu-gaugespektrum.pdf', bbox_inches='tight')
    #print('m = ', params[0], '+/-', errors[0])
    #print('b = ', params[1], '+/-', errors[1])

    #Berechnung des Raumwinkels
    r = Q_(22.5, 'mm')
    a = Q_(ufloat(7.30, 1), 'mm')
    a = a + Q_(1.50, 'cm')
    a.to('mm')
    w = 2*np.pi*ureg.rad*(1-a*(a**2+r**2)**(-0.5))
    print(f'Der vorliegende Raumwinkel mit a = {a} und r = {r} beträgt {w}.')

if __name__ == '__main__':

    if not os.path.isdir('build'):
        os.mkdir('build')

    eichung()
