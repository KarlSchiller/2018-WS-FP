import os
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as sdevs)
import matplotlib.pyplot as plt
import datetime
from astropy.io import ascii
from scipy.optimize import curve_fit
import scipy.constants as codata
from tab2tex import make_table
import pint
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
from scipy.signal import find_peaks


#Zur Berechnung der linearen Ausgleichsrechnung der Energieeichung
def linear(x, m, b):
    return m*x + b

#Gauss-Funktion für Fit
def gauss(x, sigma, h, a, mu):
    return a+h*np.exp(-(x-mu)**2/(2*sigma**2))

#Exponentialfunktion für Fit der Deketor-Effizienz
def potenz(x, a, b, c, e):
    return a*(x-b)**e + c

#-------------Aufgabenteil a)
#
def eichung():
    # Berechnung der Aktivität am Messtag
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


    #finde die peaks mit Energie E, der Binhöhe Bin_index, der Peakhöhe und der Emissions-Wahrscheinlichkeit
    E, Bin_index, pH, emis_wahr= np.genfromtxt("data/Eu_sortiert.txt", unpack=True)
    #E = E * ureg.kiloelectron_volts
    #Erstelle Tabelle mit Daten aus der
    print(emis_wahr)
    make_table(
        header = ['Energie $E$ / \kilo\electronvolt', 'Emis.-Wahr. W', 'Zugeordnete Bin-Index $i$', 'Peakhöhe' ],
        data = [E, emis_wahr, Bin_index, pH],
        places = [4.0, 2.1, 4.0, 4.0],
        caption = 'Gegebene Werte zur Kalibrierung des Germanium-Detektors \cite{anleitung}.',
        label = 'tab:anleitung_eu',
        filename = 'build/tables/anleitung_eu.tex'
        )
    counts = np.genfromtxt("data/Eu.txt", unpack=True)
    peaks = find_peaks(counts, height=10, distance=10)

    indexes = peaks[0]
    peak_heights = peaks[1]
    #print(indexes)
    #print(peak_heights)

    #Ausgleichsrechnung der Energieeichung
    x_plot = np.linspace(0, 1450)
    params, cov = curve_fit(linear, E, Bin_index)
    errors = np.sqrt(np.diag(cov))
    print('Werte zur Energiekallibrierung:')
    m =[]
    m = ufloat(params[0], errors[0])
    print('m = ', params[0], '+/-', errors[0])
    b = []
    b = ufloat(params[1], errors[1])
    print('b = ', params[1], '+/-', errors[1])

    plt.plot(E, Bin_index, 'bx', label='Eu-Gamma-Spektrum')
    plt.plot(x_plot, linear(x_plot, *params), 'r-', label='Ausgleichsrechnung')
    plt.legend()
    plt.xlabel(r'E$_\gamma$ / keV')
    plt.ylabel(r'Bineinträge')
    plt.savefig('build/Referenz.pdf')
    plt.clf()

    return E, Bin_index, pH, counts, eu_aktivitaet, emis_wahr, m, b

    #Berechnung des Raumwinkels
def raumwinkel():
    r = Q_(22.5, 'mm')
    a = Q_(ufloat(7.30, 1), 'mm')
    a = a + Q_(1.50, 'cm')
    a.to('mm')
    w = 0.5*(1-a*(a**2+r**2)**(-0.5))
    print(f'Der vorliegende Raumwinkel mit a = {a} und r = {r} beträgt Omega/4*pi = {w}.')
    return w

    #Führe den Gauss-Fit für die Peakinhalte Z
def gauss_fit(Bin_index, counts):
    peak_inhalt = []
    index_fit = []
    hoehe = []
    unter = []
    sigma = []
    for i in Bin_index:
        #Bin_index are floating points, so there have to be a typecast
        i = int(i)
        #Verwende mehrere Bins der Orginaldaten um Fit zu erlauben
        a=i-40
        b=i+40
        params_gauss,cov_gauss = curve_fit(gauss, np.arange(a,b+1), counts[a:b+1], p0=[1, counts[i], 0, i-0.1])
        errors_gauss = np.sqrt(np.diag(cov_gauss))

        #Verbinde Parameter mit zugehörigen Fehlern
        sigma_fit=ufloat(params_gauss[0],errors_gauss[0])
        h_fit=ufloat(params_gauss[1],errors_gauss[1])
        a_fit=ufloat(params_gauss[2],errors_gauss[2])
        mu_fit=ufloat(params_gauss[3],errors_gauss[3])

        #Gebe Fit-Parameter für jeden Bin-Fit in Array an
        index_fit.append(mu_fit)
        hoehe.append(h_fit)
        unter.append(a_fit)
        sigma.append(sigma_fit)
        peak_inhalt.append(h_fit*sigma_fit*np.sqrt(2*np.pi))

    #Erstelle Tabelle der Fitparameter des Gauss-Fits
    make_table(
        header= ['$\mu_i$', '$\sigma_i$', '$h_i$', '$a$'],
        data= [index_fit, sigma, hoehe, unter],
        caption = 'Parameter des durchgeführten Gauss-Fits pro Bin. Dabei ist $\mu$ der Mittelwert, $\sigma$ die Standardabweichnug, $h$ die Höhe und a der Energieoffset.',
        label = 'tab:gauss_parameter',
        places = [(1.2, 1.2), (1.2, 1.2), (1.2, 1.2), (1.2, 1.2)],
        filename = 'build/tables/Gauss-Fit-Parameter.tex'
    )


    #print(f'Der Mittelwert mu beträgt {index_fit}.')
    #print(f'Die Höhe h beträgt {hoehe}.')
    #print(f'Der Offset beträgt {unter}.')
    #print(f'Die Standardabweichung sigma beträgt {sigma}.')
    #print(f'Der Peakinhalt Z beträgt {peak_inhalt}.')
    return index_fit, peak_inhalt, hoehe, unter, sigma

# Berechnung der Detektor-Effizenz Q des Germanium Detektors
def detector_efficiency(Omega, Z, A, W, E, m, b):
    Q = []
    A = A/ureg.becquerel
    #for i in range(len(W)):
    Q = Z/(Omega*A*W)

    #Erstelle Tabelle der Peakhöhen, zugeordneten energien und daraus berechneten Detektor-Effizienzen
    make_table(
        data= [Z, E, Q],
        header = ['$Z_i$', 'E_i' ,'$Q \ \si{becquerel}$'],
        caption = 'Peakhöhe, Energie und Detektoreffizenz als Ergebnis des Gaußfits.',
        label = 'tab:det_eff',
        places = [ (1.2, 1.2), 1.2, (1.2, 1.2)],
        filename = 'build/tables/det_eff.tex'
    )

    #Potenz-fit der Detektoreffizienz mit Energien über 150 keV
    E1 = []
    Q1 = []
    for i in range(len(Q)):
        if E[i] > 150:
            E1.append(E[i])
            Q1.append(Q[i].magnitude)
    print(len(E1), len(Q1))
    print(E1,noms(Q1))


    print(sdevs(Q1))
    #params, cov = curve_fit(potenz, noms(E1), noms(Q1), sigma=sdevs(Q1))
    #error = (np.diag(cov))**(0.5)

    #print('Parameter der Potenzfunktion:')
    #print(f'Steigung a = {params[0]} +/- {error[0]}')
    #print(f'Verschiebung b = {params[1]} +/- {error[1]}')
    #print(f'Verschiebung c = {params[2]} +/- {error[2]}')
    #print(f'Exponent e = {params[3]} +/- {error[3]}')
    #x_plot = np.linspace(200, 1600)
    plt.plot(E, noms(Q), 'bx', label='Effizienz-Energie')
    #plt.plot(x_plot, potenz(x_plot, *params), 'r-', label='Effizenz-Energie-Fit')
    plt.legend(loc='best')
    plt.xlabel(r'$E\,/\,keV$')
    plt.ylabel(r'$Q(E)$')
    plt.savefig('build/det_eff.pdf')
    plt.clf()
    return Q


#----------Aufgabenteil b)
#



if __name__ == '__main__':

    if not os.path.isdir('build'):
        os.mkdir('build')
    if not os.path.isdir('build/tables'):
        os.mkdir('build/tables')
    #Teilaufgabe a
    E, Bin_index, peak_height, counts, eu_aktivitaet, emis_wahr, m, b = eichung()
    w = raumwinkel()
    index_fit, peak_inhalt, hoehe, unter, sigma = gauss_fit(Bin_index, counts)
    detector_efficiency(w, peak_inhalt, eu_aktivitaet, emis_wahr, E, m, b)
