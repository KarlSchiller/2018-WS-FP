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

gp = const.physical_constants["proton gyromag. ratio"]
print("Gyromagnetischer Faktor eines Protons:", gp)

k = const.physical_constants["Boltzmann constant"]
print("Boltzmann-Konstante: k = ", k)

#-------------------------Verwendete Funktionen--------------------
def formel_T1(tau, m0, T1):
    return m0 * (1 - 2*np.exp(-tau / T1)) # T1 = 2

def formel_T2(t, m0, T2):
    return m0 * np.exp(-t/T2)

def linear(x, m, b):
    return m*x+b

def viskos():
    alpha = 1.024*10**(-9) #m**2/s**2
    rho = 1 #Dichte von Wasser
    t = 920
    delta = [1.2, 0.9, 0.7, 0.5, 0.4]
    t2 = [600, 700, 800, 900, 1000]

    params, cov = curve_fit(linear, delta, t2 )
    errors = np.sqrt(np.diag(cov))

    delta_real = (t - params[1]) / params[0]
    #delta_real = 0.48
    x_range = np.linspace(0.4, 1.2)
    plt.plot(delta, t2, label='Daten')
    plt.plot(x_range, linear(x_range, *params), label='Fit')
    plt.xlabel(r'Parameter \delta / \si{second}')
    plt.ylabel(r'Zeit $t$ / $10^3$\si{\second}')
    plt.legend(loc='best')
    plt.savefig('build/test.pdf')
    plt.clf()
    print(delta_real)
    return rho*alpha*(t-delta_real)

def G(t12):
    return (4 * 2.2) /(440* gp[0] * t12)

def diff_konst(x, m0, D):
    t = 2 * x
    #print(T2, g)

    #print('Variablen des Fits: ', T2, gp[0], g)
    return m0 * np.exp(-t / T2[0]) * np.exp(-D * gp[0]**2 * g**2 * t**3 / 12)

#----------------------------Auswertungen---------------------------

def messung_T1():
    tau, M = np.genfromtxt('rohdaten/t1.txt', unpack=True)

    params, cov = curve_fit(formel_T1, tau, M, p0 = [-640, 2])
    errors = np.sqrt(np.diag(cov))
    #print('Anfangsbedingung der Magnetisierung M0 = ', params[0] , ' +/- ', errors[0])
    print('Relaxationszeit T1 = ', np.round(params[1], 3), ' +/- ', np.round(errors[1], 3))

    x_range = np.linspace(min(tau), max(tau), 100000)
    plt.plot(tau, -M, 'bx', label='Messwerte')
    plt.plot(x_range, -formel_T1(x_range, *params), 'r-', label='Fit')
    plt.ylabel(r'$M\:/\:\si{\milli\volt}$')
    plt.xlabel(r'$\tau\:/\:\si{\second}$')
    plt.xscale('log')
    plt.tight_layout()
    plt.legend(loc='best')
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
    T2 = [params[1], errors[1]]
    print('Zeitkonstante T2 ist gegeben durch: T2 = ', np.round(params[1], 3), '+/-', np.round(errors[1],3))

    #print(len(peaks["peak_heights"]), len(peak_tau))
    plt.plot(peak_tau, -peaks["peak_heights"], 'bx', label='Peaks')
    plt.plot(tau, formel_T2(tau, *params), 'r-', label='Messwerte')
    plt.ylabel(r'$M\:/\:\si{\milli\volt}$')
    plt.xlabel(r'$\tau\:/\:\si{\second}$')
    plt.legend(loc='best')
    plt.savefig('build/MG.pdf')
    plt.clf()
    return T2


def T2_Carr_purcell():
    tau, M = np.genfromtxt('rohdaten/cp_3.csv', unpack=True)
    peak_index, peaks = find_peaks(M, height=0.02)
    print(peaks["peak_heights"])
    peak_tau = []
    for i in peak_index:
        peak_tau.append(tau[i])

    params, cov = curve_fit(formel_T2, peak_tau[:5], peaks["peak_heights"][:5])
    errors = np.sqrt(np.diag(cov))
    print('Zeitkonstante T2 nach Cell_Purcell ist gegeben durch: T2 = ', params[1], '+/-', errors[1])

    #plt.plot(tau, M, 'bx' ,label='Messwerte')
    plt.plot(peak_tau[:10], peaks["peak_heights"][:10], 'kx', label='Peaks')
    plt.plot(tau, formel_T2(tau, *params), 'r-', label='Regression')
    #plt.plot(peak_tau, peaks["peak_heights"], 'rx', label='Peaks')
    plt.savefig('build/CP.pdf')
    plt.clf()



def t1_2():
    tau, M = np.genfromtxt('rohdaten/halbwertsbreite.csv', unpack=True)
    peak_index, peaks = find_peaks(M, height=0.6)
    #print(peaks["peak_heights"][4])
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

    g = G(FWHM)
    print('Der Faktor G zur Berechnung der Diffusionskonstante: G = ', np.round(g, 9))
    return g, FWHM



def diffusion(T2, g):
    tau, M = np.genfromtxt('rohdaten/diffusion.txt', unpack=True)

    params, cov = curve_fit(diff_konst, tau, M)
    errors = np.sqrt(np.diag(cov))
    D = [params[1], errors[1]]
    print('Diffusionskonstante: D = ', np.round(params[1], 3), ' +/- ',  np.round(errors[1], 3))

    x_range = np.linspace(min(tau), max(tau))
    plt.plot(tau, M, 'bx', label='Messwerte')
    plt.plot(x_range, diff_konst(x_range, *params), 'r-', label='Fit')
    plt.ylabel(r'$M\:/\:\si{\milli\volt}$')
    plt.xlabel(r'$\tau\:/\:\si{\second}$')
    #  plt.xscale('log')
    plt.tight_layout()
    plt.savefig('build/diffussion')
    plt.clf()

    return D


def r_molekuel(D, FWHM):
    T = 25 # celsius
    T = 273.15 + T
    eta = viskos()
    g = G(FWHM)
    #Berechnung des Molekülradius aus vorheriger Auswertung
    r_berechnet = k[0]*T/(6*np.pi)/(g*D[0]*np.pi*eta)
    print("Aus der voherigen Analyse berechnete Molekülredius: r = ", r_berechnet)

    #Berechnung des Molekülradius zum Vergleich
    #r_vergleich =

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

    messung_T1()
    T2 = T2_Meiboom_Gill()
    g , FWHM = t1_2()
    D = diffusion(T2, g)
    r_molekuel(D, FWHM)
    T2_Carr_purcell()
