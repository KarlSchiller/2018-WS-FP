import os
import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import ufloat
from scipy.optimize import curve_fit
import scipy.constants as const
import imageio
from scipy.signal import find_peaks
from scipy.signal import argrelmin
from scipy.signal import argrelmax
import pint
import string as str
from tab2tex import make_table
ureg = pint.UnitRegistry(auto_reduce_dimensions = True)
Q_ = ureg.Quantity

#------------------------Verwendete Konstanten--------------------
c = Q_(const.value('speed of light in vacuum'), const.unit('speed of light in vacuum'))
h = Q_(const.value('Planck constant'), const.unit('Planck constant'))
#  c = const.c
#  h = const.h
muB = const.value('Bohr magneton')
gyro_faktor = 2.6752219 #rad/(sT)
max_gradient = -9

#-------------------------Verwendete Funktionen--------------------
def formel_T1(tau, m0, T1):
    return m0 * (1 - 2*np.exp(-tau / T1)) # T1 = 2

def formel_T2(t, m0, T2):
    return m0 * np.exp(-t/T2)

def diffusionskonstante(tau, D):
    return np.exp(-2/3 * D * gyro_faktor**2 * max_gradient**2 * tau**3)

#----------------------------Auswertungen---------------------------
def messung_T1():
    tau, M = np.genfromtxt('rohdaten/t1.txt', unpack=True)

    params, cov = curve_fit(formel_T1, tau, M, p0 = [-640, 2])
    errors = np.sqrt(np.diag(cov))
    print('Anfangsbedingung der Magnetisierung M0 = ', params[0] , ' +/- ', errors[0])
    print('Relaxationszeit T1 = ', params[1], ' +/- ', errors[1])

    x_range = np.linspace(min(tau), max(tau), 100000)
    plt.plot(tau, -M, 'bx', label='Messwerte')
    plt.plot(x_range, -formel_T1(x_range, *params), 'r-', label='Fit')
    plt.ylabel(r'$M\:/\:\si{\milli\volt}$')
    plt.xlabel(r'$\tau\:/\:\si{\second}$')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('build/t1')
    plt.clf()

def T2_Meiboom_Gill():
    tau, M = np.genfromtxt('rohdaten/mg_2.csv', unpack=True)
    #find Minima for regression by using the negative values of M
    _, peaks = find_peaks(-M, height=0.13)
    peak_tau = []
    for i in _:
        peak_tau.append(tau[i])

    #do regression with formula 11 in the description
    params, cov = curve_fit(formel_T2, peak_tau, -peaks["peak_heights"])
    errors = np.sqrt(np.diag(cov))
    print('Zeitkonstante T2 ist gegeben durch: D = ', params[1], '+/-', errors[1])

    #print(len(peaks["peak_heights"]), len(peak_tau))
    plt.plot(peak_tau, -peaks["peak_heights"], 'bx', label='Peaks')
    plt.plot(tau, formel_T2(tau, *params), 'r-', label='Messwerte')
    plt.ylabel(r'$M\:/\:\si{\milli\volt}$')
    plt.xlabel(r'$\tau\:/\:\si{\second}$')
    plt.savefig('build/MG.pdf')
    plt.clf()

def T2_Carr_purcell():
    tau, M = np.genfromtxt('rohdaten/cp_3.csv', unpack=True)
    peak_index, peaks = find_peaks(M, height=0.02)
    print(peaks["peak_heights"])
    peaks2_index, peaks2 = find_peaks(peaks["peak_heights"])
    print(peaks2)
    peak_tau = []
    for i in peak_index:
        peak_tau.append(tau[i])

    params, cov = curve_fit(formel_T2, peak_tau, peaks2["peak_heights"])
    errors = np.sqrt(np.diag(cov))
    print('Zeitkonstante T2 ist gegeben durch: D = ', params[1], '+/-', errors[1])

    plt.plot(tau, M, 'bx' ,label='Messwerte')
    plt.plot(peak_tau, peaks2["peak_heights"], 'kx', label='Peaks')
    plt.plot(tau, formel_T2(tau, *params), 'r-', label='Messwerte')
    #plt.plot(peak2_tau, peaks2["peak_heights"], 'rx', label='Peaks')
    plt.savefig('build/CP.pdf')
    plt.clf()

def t1_2():
    tau, M = np.genfromtxt('rohdaten/halbwertsbreite.csv', unpack=True)
    peak_index, peaks = find_peaks(M, height=0.6)
    print(peaks["peak_heights"][4])
    FWHM = (2.0545-1.969)*10**(-3)

    plt.plot(tau*10**3, M, 'b.', label='Messdaten')
    plt.vlines(2.0545, -0.1, 0.7, linestyle='dashed')
    plt.vlines(1.969, -0.1, 0.7, linestyle='dashed')
    plt.hlines(peaks["peak_heights"][4]/2, 2.0545, 1.969, colors='red', linestyle='dashed', label=r't$_{12}$')
    plt.ylabel(r'$M\:/\:\si{\milli\volt}$')
    plt.xlim(1.5, 2.5)
    plt.xlabel(r'$\tau\:/\:\si{\milli\second}$')
    plt.legend(loc='best')
    plt.savefig('build/halbwertsbreite.pdf')
    plt.clf()

def diffusion():
    tau, M = np.genfromtxt('rohdaten/diffusion.txt', unpack=True)

    params, cov = curve_fit(diffusionskonstante, tau, M)
    errors = np.sqrt(np.diag(cov))
    print('Diffusionskonstante: D = ', params, ' +/- ',  errors)

    x_range = np.linspace(min(tau), max(tau))
    plt.plot(tau, M, 'kx', label='Messwerte')
    #plt.plot(x_range, diffusionskonstante(x_range, params), 'b-', label='Fit')
    plt.ylabel(r'$M\:/\:\si{\milli\volt}$')
    plt.xlabel(r'$\tau\:/\:\si{\second}$')
    #  plt.xscale('log')
    plt.tight_layout()
    plt.savefig('build/diffussion')
    plt.clf()

    #  # save results
    #  make_table(header= ['$\delta s$ / \pixel', '$\Delta s$ / \pixel', '$\delta\lambda$ / \pico\meter', '$\zeta$'],
            #  places= [3.0, 3.0, 2.2, (1.2, 1.2)],
            #  data = [delta_s, del_s, d_lambda*1e12, delta_mg],
            #  caption = 'Werte zur Bestimmung des Lande-Faktors für die rote Spektrallinie.',
            #  label = 'tab:rot_sigma',
            #  filename = 'build/rot_sigma.tex')


if __name__ == '__main__':

    if not os.path.isdir('build'):
        os.mkdir('build')

    #messung_T1()
    #diffusion()
    #T2_Meiboom_Gill()
    T2_Carr_purcell()
    #t1_2()
