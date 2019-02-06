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

c = Q_(const.value('speed of light in vacuum'), const.unit('speed of light in vacuum'))
h = Q_(const.value('Planck constant'), const.unit('Planck constant'))


def linear(x, a, b):
    '''Linear Regression'''
    return a*x+b


def test():
    # Import data from text files with numpy
    # x, y = np.genfromtxt('rohdaten/filename.txt', unpack=True)

    # test data
    U = np.array(range(20))
    I = np.linspace(1, 10, 20)*2 + np.pi

    params, covariance = curve_fit(linear, U, I)
    errors = np.sqrt(np.diag(covariance))
    print('Strom-Spannungs-Kennlinie')
    print('\ta = {} ± {}'.format(params[0], errors[0]))
    print('\tb = {} ± {}'.format(params[1], errors[1]))

    # plot test data
    U_plot = np.linspace(0, 19, 10000)
    plt.plot(U_plot, linear(U_plot, *params),
            color=tugreen, linestyle='-', label='Regression', linewidth=0.8)
    plt.plot(U, I, 'kx', label='Messwerte')
    plt.xlabel(r'$U\:/\:\si{\volt}$')
    plt.ylabel(r'$I\:/\:\si{\milli\ampere}$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('build/ui-ui_characteristic.pdf')
    plt.clf()

    # save test data in latex table
    make_table(header= ['$U$ / \\volt', '$I$ / \\micro\\ampere'],
            places= [2.0, 2.2],
            data = [U, I],
            caption = 'Aufgenommene Strom-Spannungs-Kennlinie.',
            label = 'tab:ui-characteristic',
            filename = 'build/ui-characteristic.tex')


if __name__ == '__main__':

    if not os.path.isdir('build'):
        os.mkdir('build')

    test()
