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
import pint
from tab2tex import make_table
ureg = pint.UnitRegistry(auto_reduce_dimensions = True)
Q_ = ureg.Quantity
tugreen = '#80BA26'

#  c = Q_(const.value('speed of light in vacuum'), const.unit('speed of light in vacuum'))
#  h = Q_(const.value('Planck constant'), const.unit('Planck constant'))
c = const.c
h = const.h
muB = const.value('Bohr magneton')


def linear(x, a, b):
    '''Lineare Regressionsfunktion'''
    return a * x + b

def ui_characterstic():
    '''Strom-Spannungs-Kennlinie'''
    U, I = np.genfromtxt('rohdaten/ui-characteristic.txt', unpack=True)

    plt.axvspan(xmin=65, xmax=85, facecolor=tugreen, label=r'Mögliches $U_{\mathrm{Dep}}$')  # alpha=0.9
    plt.axvline(x=100, color='k', linestyle='--', linewidth=0.8, label=r'Anglegte $U_{\mathrm{Exp}}$')
    plt.plot(U, I, 'kx', label='Messwerte')
    plt.xlabel(r'$U\:/\:\si{\volt}$')
    plt.ylabel(r'$I\:/\:\si{\micro\ampere}$')
    plt.legend(loc='lower right')  # lower right oder best
    plt.tight_layout()
    plt.savefig('build/ui-characteristic.pdf')
    plt.clf()

    mid = len(U) // 2  # use modulo operator
    make_table(header= ['$U$ / \\volt', '$I$ / \\micro\\ampere', '$U$ / \\volt', '$I$ / \\micro\\ampere'],
            places= [3.0, 1.2, 3.0, 1.2],
            data = [U[:mid], I[:mid], U[mid:], I[mid:]],
            caption = 'Aufgenommene Strom-Spannungs-Kennlinie.',
            label = 'tab:ui-characteristic',
            filename = 'build/ui-characteristic.tex')


def eichung():
    '''Eichung der Magnetischen Flussdichte'''
    # Eichung des Elektromagneten
    I, B = np.genfromtxt('rohdaten/eichung.txt', unpack=True)
    params, covariance = curve_fit(eichfunktion, I, B)
    errors = np.sqrt(np.diag(covariance))
    print('Eichung')
    print(f'\ta_0 = {params[0]} ± {errors[0]}')
    print(f'\ta_1 = {params[1]} ± {errors[1]}')
    print(f'\ta_2 = {params[2]} ± {errors[2]}')
    print(f'\ta_3 = {params[3]} ± {errors[3]}')
    print(f'\ta_4 = {params[4]} ± {errors[4]}')

    # Plot
    x_plot = np.linspace(0, 20, 10000)
    plt.plot(I, B, 'kx', label='Messwerte')
    plt.plot(x_plot, eichfunktion(x_plot, *params), 'b-', label='Regression')
    plt.xlabel(r'$I\:/\:$A')
    plt.ylabel(r'$B\:/\:$mT')
    plt.tight_layout()
    plt.savefig('build/eichung.pdf')
    plt.clf()

    I_halb = len(I) // 2
    B_halb = len(B) // 2
    # Speichern der Messwerte
    make_table(header= ['$I$ / \\ampere', '$B$ / \milli\\tesla', '$I$ / \\ampere', '$B$ / \milli\\tesla'],
            places= [2.1, 4.0, 2.1, 4.0],
            data = [I[:I_halb], B[:B_halb], I[I_halb:], B[B_halb:]],
            caption = 'Magnetische Flussdichte in Abhängigkeit des angelegten Stroms.',
            label = 'tab:eichung',
            filename = 'build/eichung.tex')
    return params, errors


if __name__ == '__main__':

    if not os.path.isdir('build'):
        os.mkdir('build')

    ui_characteristic()
