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
tuorange = '#E36913'

#  c = Q_(const.value('speed of light in vacuum'), const.unit('speed of light in vacuum'))
#  h = Q_(const.value('Planck constant'), const.unit('Planck constant'))
#  c = const.c
#  h = const.h
#  muB = const.value('Bohr magneton')

# Estimation depletion voltage. Could not fit ...
Udepletion = 80


def linear(x, a, b):
    '''Lineare Regressionsfunktion'''
    return a * x + b


def umrechnung(x, a0, a1, a2, a3, a4):
    '''Polynom 4.Grades zur Umrechnung ADCC in eV'''
    return a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0


def cce(U, a, Udep):
    '''CCE at U < Udep'''
    D = 300e-6  # sensor thickness in meter
    dc = D*np.sqrt(U/Udep)
    return 1-np.exp(-dc/a) / (1-np.exp(-D/a))


def cce_2(U, a):
    '''CCE at U < Udep'''
    D = 300e-6
    dc = D*np.sqrt(U/Udepletion)
    return 1-np.exp(-dc/a) / (1-np.exp(-D/a))


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

    #  mean adc with regression
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


def ccel():
    '''Estimation of the Charge Collection Efficiency with the use of a Laser'''
    temp = pd.DataFrame()  # help Dataframe to import Data
    for voltage in np.arange(0, 201, 10):
        temp['{}'.format(voltage)] = np.genfromtxt('rohdaten/CCEL/{}CCEL.txt'.format(voltage),
                unpack=True)

    #  To get an idea, which strip is to analize
    #  WARNING: Applied voltage values not correct in figure!
    #  print('\tPlot Heatmap')
    #  fig, ax = plt.subplots()
    #  heatmap = ax.pcolor(temp, cmap='binary', edgecolors='k', linewidths=0.1)
    #  cbar = plt.colorbar(heatmap)
    #  cbar.set_label('ADC Counts')
    #  ax.set_ylabel('Streifen')
    #  ax.set_xlabel('Spannung')
    #  fig.tight_layout()
    #  fig.savefig('build/ccel-uebersicht.pdf')
    #  fig.clf()

    # analyse strip 84
    strip_84 = temp.iloc[83]
    df = pd.DataFrame()
    df['ccel'] = strip_84 / strip_84.max()
    applied_voltage = np.arange(0, 201, 10)
    start = 0
    stop = 8
    #  params, covariance = curve_fit(cce, applied_voltage[start:stop], df['ccel'][start:stop],
            #  p0 = [1e-4, 80],
            #  bounds = ([1e-6, 80],  # lower bounds
                      #  [1, 81])) # upper bounds
    params, covariance = curve_fit(cce_2, applied_voltage[start:stop], df['ccel'][start:stop],
            p0 = [1e-4],
            bounds = (1e-6, 1)) # lower and upper bound
    errors = np.sqrt(np.diag(covariance))
    print('\tFit CCE from {} to {} volt (indices {} to {})'.format(applied_voltage[start],
        applied_voltage[stop], start, stop))
    print('\tUdep set to {} volt'.format(Udepletion))
    print(f'\ta    = {params[0]} ± {errors[0]}')
    #  print(f'\tUdep = {params[1]} ± {errors[1]}')


    print('\tPlot CCEL')
    voltage_plot = np.linspace(0, 200, 10000)
    plt.axvspan(xmin=65, xmax=85, facecolor=tugreen, label=r'Ermitteltes $U_{\mathrm{Dep}}$')  # alpha=0.9
    plt.plot(applied_voltage[stop:], df['ccel'][stop:], 'kx')
    plt.plot(applied_voltage[start:stop], df['ccel'][start:stop], 'x', color=tuorange)
    #  plt.plot(voltage_plot, cce(voltage_plot, *params), 'r-')
    plt.plot(voltage_plot, cce_2(voltage_plot, *params), color=tuorange)
    plt.xlabel(r'$U\:/\:\si{\volt}$')
    plt.ylabel(r'Normiertes Messsignal')
    plt.tight_layout()
    #  plt.legend()
    plt.savefig('build/ccel.pdf')
    plt.clf()

    # return ccel-values to compare them with the cceq measurement
    return df['ccel']


def cceq():
    '''Estimation of the Charge Collection Efficiency with the use of a beta source'''
    #  array to save the mean counts of the clusters
    mean_counts = np.array([])
    applied_voltage = np.arange(0, 201, 10)
    for voltage in applied_voltage:
        temp =  pd.read_csv('rohdaten/CCEQ/{}_Cluster_adc_entries.txt'.format(voltage),
            sep = '\t',
            # Maximum Number of columns possible is the number of stripes
            names = ['{}'.format(i) for i in range(128)],
            skiprows=1)
        #  mean_counts['{}'.format(voltage)] = temp.sum(axis=1).mean()
        mean_counts = np.append(mean_counts, temp.sum(axis=1).mean())
    # norm the signal
    mean_counts = mean_counts / np.max(mean_counts)

    # Not necessary, because the same data is displayed in cce-verlgeich.pdf
    #  print('\tPlot CCEQ')
    #  voltage_plot = np.linspace(0, 200, 10000)
    #  plt.plot(applied_voltage, mean_counts, 'kx')
    #  plt.xlabel(r'$U\:/\:\si{\volt}$')
    #  plt.ylabel(r'Normiertes Messsignal')
    #  plt.tight_layout()
    #  #  plt.legend()
    #  plt.savefig('build/cceq.pdf')
    #  plt.clf()

    # return cceq-values to compare them with the ccel measurement
    return mean_counts


def cce_vergleich(ccel_data, cceq_data):
    '''Compare CCE-measurements with laser and source'''
    print('\tPlot Vergleich')
    applied_voltage = np.arange(0, 201, 10)
    plt.plot(applied_voltage, ccel_data, 'kx', label='Laser')
    plt.plot(applied_voltage, cceq_data, 'x', color=tugreen, label='Quelle')
    plt.xlabel(r'$U\:/\:\si{\volt}$')
    plt.ylabel(r'Normiertes Messsignal')
    plt.legend(loc='lower right')
    plt.tight_layout()
    #  plt.legend()
    plt.savefig('build/cce-vergleich.pdf')
    plt.clf()

    return None


def source_scan(calib_params, calib_errors):
    '''Characterization of a beta source'''
    # number of clusters per event
    df_cluster_number = pd.read_csv('rohdaten/number_of_clusters.txt',
            names=['adcc'],
            skiprows=1)
    # Dataframe indexed from 0 to 127, but number of clusters starts with 1
    df_cluster_number.set_index(np.arange(1, 129), inplace=True)
    # extract only the cluster numbers not equal zero
    df_cluster_number = df_cluster_number[df_cluster_number != 0].dropna()
    df_cluster_number['log'] = np.log10(df_cluster_number['adcc'])
    print('\tPlot Number of Clusters')
    plt.bar(df_cluster_number.index.values, df_cluster_number['log'], edgecolor='k', color='w')
    plt.xlabel('Anzahl Cluster')
    plt.ylabel('Logarithmierte Häufigkeit')
    plt.tight_layout()
    plt.savefig('build/number-of-clusters.pdf')
    plt.clf()

    # number of channels per cluster
    df_channel_number = pd.read_csv('rohdaten/cluster_size.txt',
            names=['adcc'],
            skiprows=1)
    #  extract only the number of channels not equal zero
    df_channel_number = df_channel_number[df_channel_number != 0].dropna()
    df_channel_number['log'] = np.log10(df_channel_number['adcc'])
    print('\tPlot Number of Channels')
    plt.bar(df_channel_number.index.values, df_channel_number['log'], edgecolor='k', color='w')
    plt.xlabel('Anzahl Kanäle')
    plt.ylabel('Logarithmierte Häufigkeit')
    plt.tight_layout()
    plt.savefig('build/number-of-channels.pdf')
    plt.clf()

    # hitmap, number of events per channel
    df_hitmap = pd.read_csv('rohdaten/hitmap.txt',
            names=['adcc'],
            skiprows=1)
    print('\tPlot Number of Events per Channel (Hitmap)')
    plt.bar(df_hitmap.index.values, df_hitmap['adcc'], color='k', width=0.2)
    plt.xlabel('Kanal')
    plt.ylabel('Anzahl Ereignisse')
    plt.tight_layout()
    plt.savefig('build/hitmap.pdf')
    plt.clf()

    # energy spectrum
    print('\tImport cluster adc entries')
    df_spectrum = pd.read_csv('rohdaten/Cluster_adc_entries.txt',
            sep='\t',
            names=['{}'.format(i) for i in range(128)],
            skiprows=1)
    # drop empty columns
    df_spectrum.dropna(axis=1, how='all', inplace=True)
    # energy spectrum in adc
    df_spectrum['adcc'] = df_spectrum.sum(axis=1)
    # energy spectrum in eV
    df_spectrum['eV'] = umrechnung(df_spectrum.drop(columns='adcc'), *params).sum(axis=1)
    df_spectrum['keV'] = df_spectrum['eV']*1e-3

    print('\tEnergy spectrum adc')
    adc_bins = np.arange(start=0, stop=300, step=1)
    plt.hist(df_spectrum['adcc'], histtype='step', bins=adc_bins, color='k', density=True)
    plt.xlabel('ADC Counts')
    plt.ylabel('relative Häufigkeit')
    plt.tight_layout()
    plt.savefig('build/spectrum-adc.pdf')
    plt.clf()

    print('\tEnergy spectrum eV')
    eV_bins = np.arange(start=0, stop=300, step=1)
    eV_values, bins, patches = plt.hist(df_spectrum['keV'], histtype='step', bins=eV_bins, color='k', density=True)
    mpv = np.argmax(eV_values)  # most probable value
    mel = df_spectrum['keV'].mean()  # mean energy loss
    mel_error = df_spectrum['keV'].std()*(1/np.sqrt(len(df_spectrum.index)-1))  # mean standard deviation
    plt.xlabel(r'Energie\:/\:\si{\kilo\electronvolt}')
    plt.ylabel('relative Häufigkeit')
    plt.axvline(mpv, color=tugreen, label='MPV', linestyle='--', linewidth=0.8)
    plt.axvline(mel, color=tuorange, label='MEL', linestyle='--', linewidth=0.8)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('build/spectrum-eV.pdf')
    plt.clf()
    print('\tMPV {} keV at index {}'.format(eV_bins[mpv], mpv))
    print('\tMEL {} keV +- {} keV'.format(mel, mel_error))

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
    print('CCEL')
    ccel_data = ccel()
    print('CCEQ')
    cceq_data = cceq()
    cce_vergleich(ccel_data, cceq_data)
    print('Quellenscan')
    source_scan(params, errors)
