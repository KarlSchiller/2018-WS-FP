import os
import numpy as np
import matplotlib.pyplot as plt
#  import uncertainties.unumpy as unp
#  from uncertainties.unumpy import nominal_values as noms
#  from uncertainties.unumpy import std_devs as stds
#  from uncertainties import ufloat
from scipy.optimize import curve_fit
#  import scipy.constants as const
#  import imageio
from scipy.signal import find_peaks
#  import pint
import pandas as pd
from tab2tex import make_table
#  ureg = pint.UnitRegistry(auto_reduce_dimensions = True)
#  Q_ = ureg.Quantity
tugreen = '#80BA26'

#  c = Q_(const.value('speed of light in vacuum'), const.unit('speed of light in vacuum'))
#  h = Q_(const.value('Planck constant'), const.unit('Planck constant'))
#  c = const.c
#  h = const.h
#  muB = const.value('Bohr magneton')


def linear(x, a, b):
    '''Lineare Regressionsfunktion'''
    return a * x + b


def umrechnung(x, a0, a1, a2, a3, a4):
    '''Polynom 4.Grades zur Umrechnung ADCC in eV'''
    return a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0


def ui_characteristic():
    '''Strom-Spannungs-Kennlinie'''
    U, I = np.genfromtxt('rohdaten/ui-characteristic.txt', unpack=True)

    print('\tPlot UI-Characteristic')
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


def pedestal_run():
    '''Auswertung des Pedestals, Noise und Common Mode'''
    # adc counts
    adc = np.genfromtxt('rohdaten/Pedestal.txt',
            unpack=True,
            delimiter=';')
    # pedestal, mean of adc counts without external source
    pedestal = np.mean(adc, axis=0)
    # common mode shift, global noise during a measurement
    common_mode = np.mean(adc-pedestal, axis=1)
    # temporary variable to compute adc - pedestal - common_mode
    difference = ((adc - pedestal).T - common_mode).T
    # noise, the 'signal' of the measurement without ext source
    noise = np.sqrt(np.sum((difference)**2, axis=0)/(len(adc)-1))

    print('\tPlot Pedestal and Noise')
    stripe_indices = np.array(range(128))
    fig, ax1 = plt.subplots()
    #  plt.bar(stripe_indices,
            #  height = pedestal,
            #  width = 0.8)
    ax1.errorbar(x=stripe_indices,
            y=pedestal,
            xerr=0.5,
            yerr=0.2,
            elinewidth=0.7,
            fmt='none',
            color='k',
            label='Pedestal')
    ax1.set_ylabel(r'Pedestal\:/\:ADCC', color='k')
    ax1.set_xlabel('Kanal')
    ax1.tick_params('y', colors='k')
    ax1.set_ylim(500.5, 518.5)
    ax2 = ax1.twinx()
    ax2.errorbar(x=stripe_indices,
            y=noise,
            xerr=0.5,
            yerr=0.01,
            elinewidth=0.7,
            fmt='none',
            color=tugreen,
            label='Noise')
    ax2.set_ylabel(r'Noise\:/\:ADCC', color=tugreen)
    ax2.tick_params('y', colors=tugreen)
    ax2.set_ylim(1.75, 2.55)
    #  fig.legend()
    fig.tight_layout()
    fig.savefig('build/pedestal.pdf')
    fig.clf()

    print('\tPlot Common Mode Shift')
    plt.hist(common_mode, histtype='step', bins=30, color='k')
    plt.xlabel('Common Mode Shift\:/\:ADCC')
    plt.ylabel('Anzahl Messungen')
    plt.tight_layout()
    plt.savefig('build/common-mode.pdf')
    plt.clf()


def kalibration():
    '''Kalibration zur Umrechnung ADC Counts in Energie'''
    print('\tPlot Delay Scan')
    #  delay, y = np.genfromtxt('rohdaten/Delay_Scan', unpack=True)
    df_delay = pd.read_table('rohdaten/Delay_Scan', skiprows=1, decimal=',')
    df_delay.columns = ['delay', 'adc']
    best_delay_index = df_delay['adc'].idxmax(axis=0)
    print('\tBest Delay at {} ns'.format(df_delay['delay'][best_delay_index]))
    plt.bar(df_delay['delay'].drop(index=best_delay_index),
            df_delay['adc'].drop(index=best_delay_index), color='k')
    plt.bar(df_delay['delay'][best_delay_index], df_delay['adc'][best_delay_index],
            color=tugreen, label='Maximum')
    plt.xlabel(r'Verzögerung\:/\:\si{\nano\second}')
    plt.ylabel('Durchschnittliche ADC Counts')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('build/delay-scan.pdf')
    plt.clf()

    print('\tCompute Kalibration')
    # Energie zur Erzeugung eines Elektron-Loch-Paares in Silizium in eV
    energy_eh_couple = 3.6
    # Importiere Kalibrationsmessungen der Channel 10, 35, 60, 90, 120
    channel_indices = [10, 35, 60, 90, 120]
    df_channel = pd.DataFrame()
    for index, channel in enumerate(channel_indices):
        if index==0:  # get injected pulse and adc
            df_channel = pd.read_table(
                    'rohdaten/Calib/channel_{}.txt'.format(channel),
                    skiprows=1, decimal=',')
            df_channel.columns = ['pulse', '{}'.format(channel)]
            # transform pulse to injected eV
            df_channel['pulse'] *= energy_eh_couple
        else:
            # the injected pulses are the same
            df_channel['{}'.format(channel)] = pd.read_table(
                    'rohdaten/Calib/channel_{}.txt'.format(channel),
                    skiprows=1, decimal=',')['Function_0']
    # mean adc
    df_channel['mean'] = df_channel.drop(columns='pulse').mean(axis=1)
    # Fit from @start to @stop index
    start = 0
    stop = 90  # Maximum of 254
    params, covariance = curve_fit(umrechnung, df_channel['mean'][start:stop],
            df_channel['pulse'][start:stop])
    errors = np.sqrt(np.diag(covariance))
    print('\tFit von {} bis {} ADCC (Index {} bis {})'.format(df_channel['mean'][start],
        df_channel['mean'][stop], start, stop))
    print(f'\ta_0 = {params[0]} ± {errors[0]}')
    print(f'\ta_1 = {params[1]} ± {errors[1]}')
    print(f'\ta_2 = {params[2]} ± {errors[2]}')
    print(f'\ta_3 = {params[3]} ± {errors[3]}')
    print(f'\ta_4 = {params[4]} ± {errors[4]}')

    print('\tPlot Kalibration')
    plt.subplots(2, 2, sharex=True, sharey=True)
    ax_1 = plt.subplot(2, 2, 1)
    ax_2 = plt.subplot(2, 2, 2)
    ax_3 = plt.subplot(2, 2, 3, sharex=ax_1)
    ax_4 = plt.subplot(2, 2, 4, sharex=ax_2)
    ax_1.plot(df_channel['pulse'], df_channel['10'], 'k-', label='Kanal 10')
    ax_2.plot(df_channel['pulse'], df_channel['35'], 'k-', label='Kanal 35')
    ax_3.plot(df_channel['pulse'], df_channel['90'], 'k-', label='Kanal 90')
    ax_4.plot(df_channel['pulse'], df_channel['120'], 'k-', label='Kanal 120')
    ax_1.legend(loc='lower right')
    ax_2.legend(loc='lower right')
    ax_3.legend(loc='lower right')
    ax_4.legend(loc='lower right')
    ax_1.set_ylabel('ADCC')
    ax_3.set_ylabel('ADCC')
    ax_3.set_xlabel(r'Injizierte Energie$\:/\:$\si{\electronvolt}')
    ax_4.set_xlabel(r'Injizierte Energie$\:/\:$\si{\electronvolt}')
    plt.tight_layout()
    plt.savefig('build/calibration.pdf')
    plt.clf()

    # mean adc with regression
    adcc_plot = np.linspace(df_channel['mean'][start],
            df_channel['mean'][stop], 10000)
    plt.plot(df_channel['mean'], df_channel['pulse'], 'k-', label='Mittelwert')
    plt.plot(adcc_plot, umrechnung(adcc_plot, *params), color=tugreen, label='Regression')
    plt.axvline(x=df_channel['mean'][start], color='k', linestyle='--', linewidth=0.8,
            label='Regressionsbereich')
    plt.axvline(x=df_channel['mean'][stop], color='k', linestyle='--', linewidth=0.8)
    plt.xlabel('ADCC')
    plt.ylabel(r'Injizierte Energie$\:/\:$\si{\electronvolt}')
    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.savefig('build/umrechnung.pdf')
    plt.clf()

    print('\tPlot Vergleich')
    df_channel['vgl'] = pd.read_table('rohdaten/Calib/channel_60_null_volt.txt',
            skiprows=1, decimal=',')['Function_0']
    plt.plot(df_channel['pulse'], df_channel['60'], 'k-', label=r'\SI{100}{\volt}')
    plt.plot(df_channel['pulse'], df_channel['vgl'], color=tugreen, label=r'\SI{0}{\volt}')
    plt.xlabel(r'Injizierte Energie$\:/\:$\si{\electronvolt}')
    plt.ylabel('ADCC')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('build/vergleich.pdf')
    plt.clf()

    # Parameter zur Umrechnung ADCC in eV
    return params, errors


def vermessung():
    '''Vermessung der Streifensensoren mittels des Lasers'''
    print('\tPlot Laser Delay')
    df_delay = pd.read_table('rohdaten/laser_sync.txt', skiprows=1, decimal=',')
    df_delay.columns = ['delay', 'adc']
    best_delay_index = df_delay['adc'].idxmax(axis=0)
    print('\tBest Laser delay at {} ns'.format(df_delay['delay'][best_delay_index]))
    plt.bar(df_delay['delay'].drop(index=best_delay_index),
            df_delay['adc'].drop(index=best_delay_index), color='k')
    plt.bar(df_delay['delay'][best_delay_index], df_delay['adc'][best_delay_index],
            color=tugreen, label='Maximum')
    plt.xlabel(r'Verzögerung\:/\:\si{\nano\second}')
    plt.ylabel('ADC Counts')
    plt.ylim(0, 150)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('build/laser-delay.pdf')
    plt.clf()

    print('\tPlot Heatmap')
    df_laser =  pd.read_csv('rohdaten/Laserscan.txt',
            sep = '\t',
            names = ['stripe {}'.format(i) for i in range(128)],
            skiprows=1)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(df_laser, cmap='binary', edgecolors='k', linewidths=0.1)
    cbar = plt.colorbar(heatmap)
    cbar.set_label('ADC Counts')
    ax.set_xlabel('Streifen')
    ax.set_ylabel('Messposition')
    fig.tight_layout()
    fig.savefig('build/streifen-uebersicht.pdf')
    fig.clf()

    print('\tAnalyse single stripes 81 and 82')
    peaks_81, peakheights = find_peaks(df_laser['stripe 81'], height=130)
    peaks_82, peakheights = find_peaks(df_laser['stripe 82'], height=130)
    # warning: the array starts at zero, but the axis label starts at one!
    peaks_81 += 1
    peaks_82 += 1
    streifendicke = np.mean(np.concatenate((np.diff(peaks_81), np.diff(peaks_82)), axis=0))
    print('\tmean stripe width {} pm 10 microns'.format(streifendicke*10))

    measure_indices = np.arange(35)+1
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    #  ax1.plot(measure_indices, df_laser['stripe 81'], marker='x', color='k',
            #  linestyle=':', linewidth=0.3, label='Streifen 81')
    ax1.plot(measure_indices, df_laser['stripe 81'], 'kx', label='Streifen 81')
    for peak in peaks_81:
        ax1.axvline(x=peak, color='k', linestyle='--', linewidth=0.8)
    ax2.plot(measure_indices, df_laser['stripe 82'], 'kx', label='Streifen 82')
    #  ax2.plot(measure_indices, df_laser['stripe 82'], marker='x', color=tugreen,
            #  linestyle=':', linewidth=0.3, label='Streifen 82')
    for peak in peaks_82:
        ax2.axvline(x=peak, color='k', linestyle='--', linewidth=0.8)
    ax1.set_ylabel('ADC Counts')
    ax2.set_ylabel('ADC Counts')
    ax2.set_xlabel('Messposition')
    ax1.set_title('Streifen 81')
    ax2.set_title('Streifen 82')
    fig.tight_layout()
    #  fig.legend()
    fig.savefig('build/streifen.pdf')
    fig.clf()
    return None


if __name__ == '__main__':

    if not os.path.isdir('build'):
        os.mkdir('build')

    print('UI-Characteristic')
    ui_characteristic()
    print('Pedestal Run')
    pedestal_run()
    print('Kalibration')
    params, errors = kalibration()
    print('Laser Vermessung')
    vermessung()
