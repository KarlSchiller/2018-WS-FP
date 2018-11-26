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


def eichfunktion(I, a0, a1, a2, a3, a4):
    '''Regressionsfunktion zur Kalibrierung des magnetischen
    Feldes in Abhängigkeit des angelegten Stromes I'''
    return a0 + a1*I + a2*I**2 + a3*I**3 + a4*I**4


def dispersionsgebiet(wellenlaenge, d, n):
    '''Berechnet Dispersionsgebiet einer Lummer-Gehrcke-Platte
    für eine Wellenlänge, Abstand der Platten d und
    Brechungsindex n'''
    return wellenlaenge**2 / (2*d) * (1 / (n**2 -1))**(0.5)


def aufloesung(wellenlaenge, L, n):
    '''Auflösung einer Lummer-Gehrcke-Platte in Abhängigkeit
    der Wellenlänge, Länge der Platte L und Brechungsindex n'''
    return L / wellenlaenge * (n**2 -1)


def lande(S, L, J):
    '''Berechnet für ein Tupel S, L und J den Lande-Faktor'''
    return (3*J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))


def lande_factors():
    '''Die theoretischen Lande-Faktoren'''
    print('Lande-Faktoren Theoriewerte')
    print(f'\t1D_2  {lande(0, 2, 2)}')
    print(f'\t1P_1  {lande(0, 1, 1)}')
    print(f'\t3S_1  {lande(1, 0, 1)}')
    print(f'\t3P_1  {lande(1, 1, 1)}')


def lummer_gehrke_platte():
    d = Q_(4, 'mm')  # Durchmesser der Platte
    L = Q_(120, 'mm')  # Laenge der Platte
    lambda_1 = Q_(644, 'nm')
    lambda_2 = Q_(480, 'nm')
    n_1 = 1.4567  # Brechungsindex bei 644nm
    n_2 = 1.4635  # Brechungsindex bei 480nm
    d_lambda_1 = dispersionsgebiet(lambda_1, d, n_1).to('pm')
    d_lambda_2 = dispersionsgebiet(lambda_2, d, n_2).to('pm')
    A_1 = aufloesung(lambda_1, L, n_1)
    A_2 = aufloesung(lambda_2, L, n_2)
    print(f'Wellenlänge {lambda_1}')
    print(f'\tDispersionsgebiet  {d_lambda_1}')
    print(f'\tAuflösung          {A_1}')
    print(f'Wellenlänge {lambda_2}')
    print(f'\tDispersionsgebiet  {d_lambda_2}')
    print(f'\tAuflösung          {A_2}')


def eichung():
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


def auswertung_blau(params, erros):
    ## Image 0
    im_0 = imageio.imread('rohdaten/blau_sigma_0A.JPG')
    im_0 = im_0[:,:,2]  # r g b  also ist blau an position 2
    mitte_0 = im_0[len(im_0) // 2]
    peaks_0 = find_peaks(mitte_0, height=20, distance=50, prominence=20)
    peak_indices_0 = peaks_0[0]
    peak_diffs_0 = np.diff(peak_indices_0)
    print(peak_indices_0)
    print(peak_diffs_0)

    # plot
    x_plot_0 = np.array(range(len(mitte_0)))
    plt.plot(x_plot_0, mitte_0, 'k.')
    plt.plot(peak_indices_0, mitte_0[peak_indices_0], 'rx')
    plt.savefig('build/blau_sigma_0A.pdf')
    plt.clf()

    ## Image 1
    im_1 = imageio.imread('rohdaten/blau_sigma_6A.JPG')
    im_1 = im_1[:,:,2]  # r g b  also ist blau an position 2
    mitte_1 = im_1[len(im_1) // 2]
    peaks_1 = find_peaks(mitte_1, height=20, distance=5, prominence=10)
    peak_indices_1 = peaks_1[0]
    peak_diffs_1 = np.diff(peak_indices_1)
    print(peak_indices_1)
    print(peak_diffs_1)

    # plot
    x_plot_1 = np.array(range(len(mitte_1)))
    plt.plot(x_plot_1, mitte_1, 'k.')
    plt.plot(peak_indices_1, mitte_1[peak_indices_1], 'rx')
    plt.savefig('build/blau_sigma_6A.pdf')
    plt.clf()

    ## Image 2
    im_2 = imageio.imread('rohdaten/blau_pi_0A.JPG')
    im_2 = im_2[:,:,2]  # r g b  also ist blau an position 2
    mitte_2 = im_2[len(im_2) // 2]
    peaks_2 = find_peaks(mitte_2, height=20, distance=50, prominence=20)
    peak_indices_2 = peaks_2[0]
    print(peak_indices_2)
    print(np.diff(peak_indices_2))

    # plot
    x_plot_2 = np.array(range(len(mitte_2)))
    plt.plot(x_plot_2, mitte_2, 'k.')
    plt.plot(peak_indices_2, mitte_2[peak_indices_2], 'rx')
    plt.savefig('build/blau_pi_0A.pdf')
    plt.clf()


if __name__ == '__main__':

    if not os.path.isdir('build'):
        os.mkdir('build')

    lande_factors()
    lummer_gehrke_platte()
    p, e = eichung()
    auswertung_blau(p, e)
