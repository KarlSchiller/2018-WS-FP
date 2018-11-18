# coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
import sympy
from uncertainties import correlated_values, correlation_matrix
from scipy.integrate import quad
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as sdevs
import scipy.constants as con
from scipy.constants import physical_constants as pcon
from scipy.signal import find_peaks
from astropy.io import ascii
from tab2tex import make_table

#------------------------Aufgabenteil a) {Untersuchung des Eu-Spektrums}
data = np.genfromtxt('data/Eu.txt', unpack=True)
E, peaks_ind, W = np.genfromtxt('data/2_0/Eu.txt', unpack=True)
#make_table(
#        header = ['Energie $E / \si{\kilo\electronvolt}$', 'Bin-Index $i$', 'Emis.-Wahr. W'],
#        data = [E, W, peaks_ind],
#        places = [1.0, 2.1, 1.0],
#        caption = 'Gegebene Werte zur Kalibrierung des Germanium-Detektors \cite{anleitung}.',
#        label = 'tab:anleitung_eu',
#        filename = 'tables/anleitung_eu.tex'
#        )

peaks = find_peaks(data, height=5, distance=10)
indexes = peaks[0]
peak_heights = peaks[1]


#Energieeichung: Wird bei jeder Betrachtung eines Spektrums benötigt
#Lineare Funktion für Ausgleichsgeraden
def lin(x,m,b):
    return m*x+b

#Linearer Fit mit Augabe der Parameter
params, covariance= curve_fit(lin,peaks_ind,E)
errors = np.sqrt(np.diag(covariance))
print('Kalibrationswerte:')
print('Steigung m =', params[0], '±', errors[0])
print('Achsenabschnitt b =', params[1], '±', errors[1])
#Zusammenfassen der Werte und Ungenauigkeiten der Fit-Parameter
m=ufloat(params[0],errors[0])
b=ufloat(params[1],errors[1])

#Plotten des vom Detektor aufgenommenen Spektrums + logarithmische y-Achse
#x=np.linspace(1,8192,8192)
#plt.bar(lin(x,*params), data, label='Messdaten' )
#plt.bar(lin(indexes,*params),data[indexes])
#plt.xlim(0, 3450)
#plt.xlabel(r'Kanäle')
#plt.ylabel(r'Zählrate $N$')
#plt.legend(loc='best')
#plt.tight_layout()
#plt.savefig('build/orginal_Eu.pdf')
#plt.yscale('log')
#plt.savefig('build/orginal_Eu_log.pdf')
#plt.clf()
#
#Plotten der Eichung/Kalibrierung am Eu-Spektrum
#x=np.linspace(1,8192,8192)
#plt.plot(x, lin(x,*params),'r-',label='Fit')
#plt.plot(peaks_ind,E,'bx',label='Daten')
#plt.xlim(0,4000)
#plt.tight_layout()
#plt.xlabel(r'Bin-Index')
#plt.grid()
#plt.ylabel(r'E$_\gamma\:/\: \mathrm{keV}$')
#plt.legend()
#plt.savefig('build/kalibation.pdf')
#plt.clf()

#------------Berechnen der Detektoreffizenz
#Berechnung der Aktivität am Messtag
A=ufloat(4130,60) #Aktivität Europium-Quelle am 01.10.2000
t_halb = ufloat(4943,5) #Halbwertszeit Europium in Tagen
dt = 18*365.25 + 43 #Zeitintervall in Tagen
A_jetzt=A*unp.exp(-unp.log(2)*dt/t_halb)#Aktivität Versuchstag
print('Aktivität zum Messzeitpunkt',A_jetzt)

#Gauß-Funktion für Peakhoehenbestimmung
def gauss(x,sigma,h,a,mu):
    return a+h*np.exp(-(x-mu)**2/(2*sigma**2))

#Verwende Gauß-Fit in jedem Bin des Spektrums um Peakhöhe zu erhalten
def gaussian_fit_peaks(test_ind):
    peak_inhalt = []
    index_fit = []
    hoehe = []
    unter = []
    sigma = []
    for i in test_ind:
        a=i-40
        b=i+40

        params_gauss,covariance_gauss=curve_fit(gauss,np.arange(a,b+1),data[a:b+1],p0=[1,data[i],0,i-0.1])
        errors_gauss = np.sqrt(np.diag(covariance_gauss))

        sigma_fit=ufloat(params_gauss[0],errors_gauss[0])
        h_fit=ufloat(params_gauss[1],errors_gauss[1])
        a_fit=ufloat(params_gauss[2],errors_gauss[2])
        mu_fit=ufloat(params_gauss[3],errors_gauss[3])
        #print(h_fit*sigma_fit*np.sqrt(2*np.pi))

        index_fit.append(mu_fit)
        hoehe.append(h_fit)
        unter.append(a_fit)
        sigma.append(sigma_fit)
        peak_inhalt.append(h_fit*sigma_fit*np.sqrt(2*np.pi))
    return index_fit, peak_inhalt, hoehe, unter, sigma

index_f, peakinhalt, hoehe, unter, sigma = gaussian_fit_peaks(peaks_ind.astype('int'))


E_det =[]
for i in range(len(index_f)):
    E_det.append(lin(index_f[i],*params))

#Berechnung des Raumwinkels
a=ufloat(7.3+1.5, 0.1) #in cm
r=ufloat(2.25, 0) #in cm
omega_4pi = (1-a/(a**2+r**2)**(0.5))/2
print('Raumwinkel',omega_4pi)

#Berechnung Detektoreffizienz für jeden Energiepeak
Q=[peakinhalt[i]/(omega_4pi*A_jetzt*W[i]) for i in range(len(W))]

##Erstellen einer Tabelle der Fit-Parameter des Gauß-Fits
#make_table(
#    header= ['$a$', '$h_i$', '$\mu_i$', '$\sigma_i$'],
#    data=[unter, hoehe, index_f, sigma],
#    caption='Parameter des durchgeführten Gauss-Fits pro Bin. Dabei ist $\mu$ der Mittelwert, $\sigma$ die Standardabweichnug, $h$ die Höhe und a der Energieoffset.',
#    label='tab:gauss_parameter',
#    places=[(1.2, 1.2), (1.2, 1.2), (1.2, 1.2), (1.2, 1.2)],
#    filename='tables/Gauss-Fit-Parameter.tex'
#    )
#
##Erstellen einer Tabelle der Detektoreffizenz und den dazu wverwendeten Werten
#make_table(
#    header=['$Z_i$', 'E_i' ,'$Q \ \si{becquerel}$'],
#    data=[peakinhalt, E_det, Q],
#    caption = 'Peakhöhe, Energie und Detektoreffizenz als Ergebnis des Gaußfits.',
#    label = 'tab:det_eff',
#    places = [ (1.2, 1.2), 1.2, (1.2, 1.2)],
#    filename = 'tables/det_eff.tex'
#    )


#Betrachte Exponential-Fit für Beziehnung zwischen Effizienz und Energie
#Lasse erste Werte weg
Q=Q[1:]
E=E[1:]
E_det=E_det[1:]

#Potenzfunktion für Fit
def potenz(x,a,b,c,e):
    return a*(x-b)**e+c

#Durchführung des Exponential-Fits und Ausgabe der Parameter
params2, covariance2= curve_fit(potenz,noms(E_det),noms(Q),sigma=sdevs(Q))
errors2 = np.sqrt(np.diag(covariance2))
#Zusammenfassen der Fit-Parameter
a=ufloat(params2[0],errors2[0])
b=ufloat(params2[1],errors2[1])
c=ufloat(params2[2],errors2[2])
e=ufloat(params2[3],errors2[3])
#Ausgabe der Fit-Parameter
print('Kalibrationswerte Potenzfunktion:')
print(f'Steigung a = {a}')
print(f'Verschiebung b = {b}')
print(f'Verschiebung c = {c}')
print(f'Exponent e = {e}')

#Plotten der Effizenz gegen die Energie mit Exponential-Fit-Funktion
#x=np.linspace(200,1600,10000)
#plt.plot(x, potenz(x,*params2),'r-',label='Fit')
#plt.plot(E,noms(Q),'b.',label='Daten')
#plt.legend()
#plt.xlabel(r'$E \:/\: keV$')
#plt.grid()
#plt.ylabel(r'$Q(E)$')
#plt.savefig('build/efficiency.pdf')
#plt.clf()

#-----------------------Teilaufgabe b) {Untersuchung des Cs-Spektrums}
data_b = np.genfromtxt('data/Cs.txt', unpack=True)
x_plot = np.linspace(0, len(data_b), len(data_b))

##Plotten des vom Detektor aufgenommenen Cs-Spektrums + logarithmische y_Achse
#plt.bar(x_plot, data_b)
#plt.xlim(0, 1800)
#plt.xlabel(r'Kanäle')
#plt.ylabel(r'Ausschläge')
#plt.tight_layout()
#plt.savefig('build/spektrum_Cs.pdf')
#plt.yscale('log')
#plt.savefig('build/spektrum_Cs_log.pdf')

#Finde Peaks in Spektrum und ordne sie der Energie zu
peaks_2 = find_peaks(data_b, height=60, distance=20)
indexes_2 = peaks_2[0]
peak_heights_2 = peaks_2[1]
energie_2 = lin(indexes_2, *params)
#print(indexes_2)
#print(energie_2)

#Identifiziere die charakteristischen Energie-Peaks
e_rueck=energie_2[-4]
e_compton=energie_2[-2]
e_photo=energie_2[-1]

##Fasse Ergebnisse der Peaksuche in Tabelle zusammen
#make_table(
#    header=['$E_\text{rueck}$', '$E_\text{compton}$', '$E_\text{photo}$'],
#    data=[e_rueck, e_compton, e_photo],
#    places=[1.2, 1.2, 1.2],
#    caption='Bestimmte Werte für den Rückstreupeak, den Comptonpeak und des Vollenergiepeaks.',
#    label='tab:peaks',
#    filename ='tables/peaks_Cs.tex'
#    )

#Vergleiche zwischen gemessenen und theoretischen Werten der Peaks
e_compton_theo = 2*e_photo**2/(con.m_e*(1+2*e_photo/con.m_e))
vgl_compton = 1-e_compton/e_compton_theo
print(f'Ein Vergleich des theoretischen E_compton {e_compton_theo} mit dem gemessenen E_compton {e_compton}, beträgt: {vgl_compton} ')

e_rueck_theo = e_photo/(1+2*e_photo/con.m_e)
vgl_rueck = 1-e_rueck/e_rueck_theo
print(f'Ein Vergleich des theoretischen E_compton {e_rueck_theo} mit dem gemessenen E_compton {e_rueck}, beträgt: {vgl_rueck} ')

#Betrachte Bereich um Vollenergiepeak herum und führe seperat eine lineare Regression von beiden Seiten durch
left = 1638
right = 1658
print(data_b[left], data_b[right])

params_l, cov_l = curve_fit(lin, data_b[left:indexes_2[-1]+1], np.arange(left, indexes_2[-1]+1))
errors_l = np.sqrt(np.diag(cov_l))
m_l = ufloat(params_l[0], errors_l[0])
b_l = ufloat(params_l[1], errors_l[1])
print(f'Für die Betrachtung der linken Seite des Vollenergiepeaks beträgt die Steigung {params_l[0]} +/-{errors_l[0]} und der Setoff {params_l[1]} +/- {params_l[1]}')

params_r, cov_r = curve_fit(lin,data_b[indexes_2[-1]:right+1],np.arange(indexes_2[-1],right+1))
errors_r = np.sqrt(np.diag(cov_r))
m_r = ufloat(params_r[0], errors_r[0])
b_r = ufloat(params_r[1], errors_r[1])
print(f'Für die Betrachtung der rechten Seite des Vollenergiepeaks beträgt die Steigung {params_r[0]} +/-{errors_r[0]} und der Setoff {params_r[1]} +/- {params_r[1]}')

#Berechne die Halbwertes und Zehntelbreite des Vollenergiepeaks und gebe Ergebnisse aus
halb = m_r*1/2*data_b[indexes_2[-1]]+b_r - (m_l*1/2*data_b[indexes_2[-1]]+b_l)
zehntel = m_r*1/10*data_b[indexes_2[-1]]+b_r - (m_l*1/10*data_b[indexes_2[-1]]+b_l)

print('Vergleich Halb- zu Zehntelwertsbreite:')
#lin beschreibt noch die lineare Regression vom beginn der Auswertung
print('Halbwertsbreite', lin(halb,*params))
print('Zehntelbreite', lin(zehntel,*params))
print('Zehntel- nach Halbwertsbreit', lin(1.823*halb,*params))
print('Verhältnis der beiden:', 1- lin(zehntel,*params)/lin((1.823*halb),*params))

##Plotte das zugeordnete Cs-Spektrum und setze Horizontale bei Zehntel- und Harlbwertsbreite
#x=np.linspace(1,8192,8192)
#plt.plot(x, data_b,'r-',label='Fit')
#plt.plot(indexes_2,data_b[indexes_2],'bx',label='Peaks')
#plt.axhline(y=0.5*data_b[indexes_2[-1]], color='g',linestyle='dashed')
#print('Halbwertshöhe', 0.5*data_b[indexes_2[-1]])
#print('Zehntelwertshöhe', 0.1*data_b[indexes_2[-1]])
#plt.axhline(y=0.1*data_b[indexes_2[-1]], color='r',linestyle='dashed')
#plt.xlim(0,2000)
#plt.xlabel(r'E$_\gamma\:/\: \mathrm{keV}$')
#plt.ylabel(r'Zählrate $N$')
#plt.grid()
#plt.legend()
#plt.tight_layout()
#plt.savefig('build/Cs.pdf')
#plt.yscale('log')
#plt.savefig('build/Cs_log.pdf')
#plt.clf()

#Führe wieder Gausß-Fit für den Vollenergiepeak durch, um Peakhöhe bestimmen zu können
a=indexes_2[-1].astype('int')-50
b=indexes_2[-1].astype('int')+50

params_gauss_b,covariance_gauss_b=curve_fit(gauss,np.arange(a,b+1),data_b[a:b+1],p0=[1,data_b[indexes_2[-1]],0,indexes_2[-1]-0.1])
errors_gauss_b = np.sqrt(np.diag(covariance_gauss_b))
#Fasse Wert und Ungenauigkeit der Fit-Parameter wieder jeweils zusammen
sigma_fit=ufloat(params_gauss_b[0],errors_gauss_b[0])
h_fit=ufloat(params_gauss_b[1],errors_gauss_b[1])
a_fit=ufloat(params_gauss_b[2],errors_gauss_b[2])
mu_fit=ufloat(params_gauss_b[3],errors_gauss_b[3])

inhalt_photo=h_fit*sigma_fit*np.sqrt(2*np.pi)
print(f'Der Inhalt/ die Höhe des Vollenergiepeaks liegt bei {inhalt_photo} keV.')


def compton(E,eps):
    a_c = data_b[indexes_2[-2]] / (1/eps**2 *(2+ e_compton**2/(e_compton-e_photo)**2*(1/eps**2+(e_photo-e_compton)/e_photo-2/eps*(e_photo-e_compton)/e_photo)))
    return a_c/eps**2 *(2+ E**2/(E-e_photo)**2*(1/eps**2+(e_photo-E)/e_photo-2/eps*(e_compton-e_photo)/e_photo))

params_compton,covariance_compton=curve_fit(compton,lin(np.arange(1,indexes_2[-2]+1),*params),data_b[0:indexes_2[-2]])
errors_compton = np.sqrt(np.diag(covariance_compton))

eps=ufloat(params_compton[0],errors_compton[0])
def compton2(E):
    eps2 = noms(eps)
    a_c = data_b[indexes_2[-2]] / (1/eps2**2 *(2+ e_compton**2/(e_compton-e_photo)**2*(1/eps2**2+(e_photo-e_compton)/e_photo-2/eps2*(e_photo-e_compton)/e_photo)))
    return a_c/eps2**2 *(2+ E**2/(E-e_photo)**2*(1/eps2**2+(e_photo-E)/e_photo-2/eps2*(e_compton-e_photo)/e_photo))
print(eps)

inhalt_comp = quad(compton2,a=lin(0,*params),b=lin(indexes_2[-2],*params))
print(inhalt_comp[0])

mu_ph = 0.002 #in cm^-1
mu_comp = 0.38
l=3.9
abs_wahrsch_ph = 1-np.exp(-mu_ph*l)
abs_wahrsch_comp = 1-np.exp(-mu_comp*l)
print(f'Die absolute Wahrscheinlichkeit eine Vollenergiepeaks liegt bei: {abs_wahrsch_ph} Prozent')
print(f'Die absolute Wahrscheinlichkeit eine Comptonpeaks liegt bei: {abs_wahrsch_comp} Prozent')


#Aufgabenteil d)
data_d = np.genfromtxt('data/Sb_Ba.txt', unpack=True) #Das sollte Barium sein
#x_plot = np.linspace(1, 8192, 8192)
#plt.bar(x_plot, data_d)
#plt.savefig('build/Ba_Sb_orginal.pdf')
#plt.yscale('log')
#plt.savefig('build/Ba_Sb_orginal_log.pdf')
#plt.clf()

peaks_3 = find_peaks(data_d, height=90, distance=15)
indexes_3 = peaks_3[0]
peak_heights_3 = peaks_3[1]
energie_3 = lin(indexes_3,*params)
print(indexes_3)
print(energie_3)


x=np.linspace(1,8192,8192)
plt.plot(x, data_d,'r-',label='Detektor')
plt.plot(indexes_3,data_d[indexes_3],'bx',label='Peaks')
plt.xlabel(r'E$_\gamma\:/\: \mathrm{keV}$')
plt.ylabel(r'Zählrate $N$')
plt.xlim(0, 1800)
plt.grid()
plt.legend()
plt.savefig('build/Ba_Sb.pdf')
plt.clf()


def gaussian_fit_peaks_d(test_ind):
    peak_inhalt = []
    index_fit = []
    hoehe = []
    sigma = []
    unter = []
    for i in test_ind:
        a=i-40
        b=i+40

        params_gauss,covariance_gauss=curve_fit(gauss,np.arange(a,b+1),data_d[a:b+1],p0=[1,data_d[i],0,i-0.1])
        errors_gauss = np.sqrt(np.diag(covariance_gauss))

        sigma_fit=ufloat(params_gauss[0],errors_gauss[0])
        h_fit=ufloat(params_gauss[1],errors_gauss[1])
        a_fit=ufloat(params_gauss[2],errors_gauss[2])
        mu_fit=ufloat(params_gauss[3],errors_gauss[3])

        hoehe.append(h_fit)
        unter.append(a_fit)
        sigma.append(sigma_fit)
        index_fit.append(mu_fit)
        peak_inhalt.append(h_fit*sigma_fit*np.sqrt(2*np.pi))
    return index_fit,peak_inhalt, hoehe, unter, sigma

#E_ba, W_ba, peaks_ind_ba = np.genfromtxt('Ba.txt', unpack=True)
#index_ba, peakinhalt_ba, hoehe_ba, unter_ba, sigma_ba = gaussian_fit_peaks_d(peaks_ind_ba.astype('int'))
#ascii.write(
#    [E_ba,W_ba,peaks_ind_ba,lin(peaks_ind_ba,*params)],
#    'ba.tex', format='latex',overwrite='True')
#
#E_ba_det = []
#for i in range(len(index_ba)):
#    E_ba_det.append(lin(index_ba[i],*params))
#
#A=peakinhalt_ba[4:]/(omega_4pi*W_ba[4:]*potenz(E_ba_det[4:],*params2)) #nur die mit E>150keV mitnehmen
#A_det = [0,0,0,0]
#for i in A:
#    A_det.append(i)
#
#make_table(
#    header=,
#    data=[index_ba, E_ba_det, hoehe_ba, sigma_ba],
#    places=,
#    caption =,
#    label =,
#    filename =
#)
##ascii.write(
##    [index_ba,E_ba_det,hoehe_ba, sigma_ba],
##    'd.tex', format='latex',overwrite='True')
##ascii.write(
##    [unter_ba, peakinhalt_ba, A_det],
##    'd2.tex', format='latex',overwrite='True')
#A_gem = ufloat(np.mean(noms(A)),np.mean(sdevs(A)))
#print('gemittelte Aktivität',A_gem)
#
##Aufgabenteil e)
#data_e = np.genfromtxt('Daten/04.txt', unpack=True)
#
#peaks_4 = find_peaks(data_e, height=100, distance=15)
#indexes_4 = peaks_4[0]
#peak_heights_4 = peaks_4[1]
#energie_4 = lin(indexes_4,*params)
#ascii.write(
#    [indexes_4,data_e[indexes_4],energie_4],
#    'e.tex', format='latex', overwrite='True')
#print(energie_4)
#
#x=np.linspace(1,8192,8192)
#plt.plot(lin(x,*params), data_e,'b--',label='Fit Bins - Energien')
#plt.plot(lin(indexes_4,*params),data_e[indexes_4],'rx',label='Detektierte Peaks')
#plt.xlabel(r'E $\:/\: \mathrm{keV}$')
#plt.ylabel(r'Zählrate $N$')
#plt.legend()
#plt.savefig('e.pdf')
#plt.clf()
#
