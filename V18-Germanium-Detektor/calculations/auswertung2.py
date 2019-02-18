# coding=utf-8
import os
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

if not os.path.isdir('build'):
    os.mkdir('build')
if not os.path.isdir('build/tables'):
    os.mkdir('build/tables')

#------------------------Aufgabenteil a) {Untersuchung des Eu-Spektrums}
data = np.genfromtxt('data/Eu.txt', unpack=True)
E, peaks_ind, W = np.genfromtxt('data/2_0/Eu.txt', unpack=True)
make_table(
        header = [' $E$ / \kilo\electronvolt', ' $W$ / \%', 'Bin-Index $i$'],
        data = [E, W, peaks_ind],
        places = [4.0, 2.1, 4.0],
        caption = 'Gegebene Werte zur Kalibrierung des Germanium-Detektors \cite{anleitung}. Aufgelistet sind die jeweilige Energie, die Emissionswahrscheinlichkeit $W$ und der Bin Index $i$.',
        label = 'tab:anleitung_eu',
        filename = 'build/tables/anleitung_eu.tex'
        )

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

##Plotten des vom Detektor aufgenommenen Spektrums + logarithmische y-Achse
#x=np.linspace(1,8192,8192)
#plt.bar(lin(x,*params), data, label='Messdaten' )
#plt.bar(lin(indexes,*params),data[indexes])
#plt.xlim(0, 1600)
#plt.xlabel(r'Bin-Indizes')
#plt.ylabel(r'Zählrate $N$')
#plt.legend(loc='best')
#plt.savefig('build/orginal_Eu.pdf')
#plt.yscale('log')
#plt.savefig('build/orginal_Eu_log.pdf')
#plt.clf()

#Plotten der Eichung/Kalibrierung am Eu-Spektrum
x=np.linspace(250,3700,3450)
plt.plot(x, lin(x,*params),'r-',label='Fit')
#plt.plot(peaks_ind,E,'bx',label='Daten')
plt.errorbar(peaks_ind, E, xerr=100, fillstyle= None, fmt=' x', label='Daten')
plt.ylim(0,1500)
plt.xlim(0, 4000)
plt.xlabel(r'Kanal')
plt.grid()
plt.ylabel(r'E$_\gamma\:/\: \mathrm{keV}$')
plt.legend()
plt.savefig('build/kalibation.pdf')
plt.clf()

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
#print('Peakinhalte: ', peak_inhalt, '', 'Autrittswahrscheinlichkeit: ', W)
Q=[peakinhalt[i]/(omega_4pi*A_jetzt*W[i]) for i in range(len(W))]
print('Peakinhalte: ', peakinhalt)
#Erstellen einer Tabelle der Fit-Parameter des Gauß-Fits
make_table(
    header= ['$a$', '$h_i$', '$\mu_i$', '$\sigma_i$ / \kilo\electronvolt'],
    data=[unter, hoehe, index_f, sigma],
    caption='Parameter des durchgeführten Gauss-Fits pro Bin. Dabei ist $\mu$ der Mittelwert, $\sigma$ die Standardabweichnug, $h$ die Höhe und a der Zählraten-Offset.',
    label='tab:gauss_parameter',
    places=[(2.2, 1.2), (4.2, 6.2), (4.2, 3.2), (3.2, 3.2)],
    filename='build/tables/Gauss-Fit-Parameter.tex'
    )

#Erstellen einer Tabelle der Detektoreffizenz und den dazu verwendeten Werten
make_table(
    header=['$Z_i$', '$E_i$ / \kilo\electronvolt' ,'$Q_i$ / \\becquerel '],
    data=[peakinhalt, E_det, Q],
    caption = 'Peakhöhe, Energie und Detektoreffizenz als Ergebnis des Gaußfits.',
    label = 'tab:det_eff',
    places = [ (5.2, 6.2), (4.2, 3.2), (3.2, 3.2)],
    filename = 'build/tables/det_eff.tex'
    )


#Betrachte Exponential-Fit für Beziehnung zwischen Effizienz und Energie
#Lasse erste Werte weg
#Q=Q[1:]
#E=E[1:]
#E_det=E_det[1:]

#Potenzfunktion für Fit
def potenz(x,a,b,c,e):
    return a*(x-b)**e+c

#Durchführung des Exponential-Fits und Ausgabe der Parameter
print('Daten für den Exponentialfit:')
print(noms(Q), noms(E_det))
params2, covariance2= curve_fit(potenz,noms(E_det),noms(Q),sigma=sdevs(Q), p0=[1, 150, 13, -2])
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
x=np.linspace(30,1600,10000)
plt.plot(x, potenz(x,*params2),'r-',label='Fit')
plt.errorbar(E,noms(Q), xerr=50,fmt=' x', ecolor='b',label='Daten')
plt.legend()
plt.xlabel(r'$E \:/\: keV$')
plt.grid()
plt.ylabel(r'$Q(E) \:/\: \frac{keV}{Bq}$')
plt.savefig('build/efficiency.pdf')
plt.clf()

#-----------------------Teilaufgabe b) {Untersuchung des Cs-Spektrums}
data_b = np.genfromtxt('data/Cs.txt', unpack=True)
x_plot = np.linspace(0, len(data_b), len(data_b))

#Plotten des vom Detektor aufgenommenen Cs-Spektrums + logarithmische y_Achse
plt.bar(x_plot, data_b)
plt.xlim(0, 1800)
plt.xlabel(r'Energie')
plt.ylabel(r'Rate')
plt.savefig('build/spektrum_Cs.pdf')
plt.yscale('log')
plt.savefig('build/spektrum_Cs_log.pdf')
plt.clf()

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
#print(len(energie_2), len(indexes_2))
#print(e_rueck, e_compton, e_photo)
#print(indexes_2[-4], indexes_2[-2], indexes_2[-1])


e_photo = 661.59
m_e = 511000
#Vergleiche zwischen gemessenen und theoretischen Werten der Peaks
e_compton_theo = 2*e_photo*(e_photo**2/m_e*(1+2*e_photo/m_e))
vgl_compton = 1-e_compton/e_compton_theo
print(f'Ein Vergleich des theoretischen E_compton {e_compton_theo} mit dem gemessenen E_compton {e_compton}, beträgt: {vgl_compton} ')

e_rueck_theo = e_photo/(1+2*e_photo/m_e)
vgl_rueck = 1-e_rueck/e_rueck_theo
print(f'Ein Vergleich des theoretischen E_rueck {e_rueck_theo} mit dem gemessenen E_compton {e_rueck}, beträgt: {vgl_rueck} ')

#Betrachte Bereich um Vollenergiepeak herum und führe seperat eine lineare Regression von beiden Seiten durch
left = 1638
right = 1658
#print(data_b[left], data_b[right])

params_l, cov_l = curve_fit(lin, data_b[left:indexes_2[-1]+1], np.arange(left, indexes_2[-1]+1))
errors_l = np.sqrt(np.diag(cov_l))
m_l = ufloat(params_l[0], errors_l[0])
b_l = ufloat(params_l[1], errors_l[1])
print(f'Für die Betrachtung der linken Seite des Vollenergiepeaks beträgt die Steigung {m_l} und der Setoff {b_l}')

params_r, cov_r = curve_fit(lin,data_b[indexes_2[-1]:right+1],np.arange(indexes_2[-1],right+1))
errors_r = np.sqrt(np.diag(cov_r))
m_r = ufloat(params_r[0], errors_r[0])
b_r = ufloat(params_r[1], errors_r[1])
print(f'Für die Betrachtung der rechten Seite des Vollenergiepeaks beträgt die Steigung {m_r} und der Setoff {b_r}')

#Berechne die Halbwertes und Zehntelbreite des Vollenergiepeaks und gebe Ergebnisse aus
halb = m_r*1/2*data_b[indexes_2[-1]]+b_r - (m_l*1/2*data_b[indexes_2[-1]]+b_l)
zehntel = m_r*1/10*data_b[indexes_2[-1]]+b_r - (m_l*1/10*data_b[indexes_2[-1]]+b_l)

print('Vergleich Halb- zu Zehntelwertsbreite:')
#lin beschreibt noch die lineare Regression vom beginn der Auswertung
print('Halbwertsbreite: ', lin(halb,*params))
print('Zehntelbreite: ', lin(zehntel,*params))
print('Halbwertes- nach Zehntelbreite : ', 1.823*lin(halb,*params))
print('Verhältnis der beiden:', 1- lin(zehntel,*params)/(1.832*lin((halb),*params)))

#Plotte das zugeordnete Cs-Spektrum und setze Horizontale bei Zehntel- und Harlbwertsbreite
x=np.linspace(1,8192,8192)
plt.plot(x, data_b,'r-',label='Fit')
plt.plot(indexes_2,data_b[indexes_2],'bx',label='Peaks')
plt.axhline(y=0.5*data_b[indexes_2[-1]], color='g',linestyle='dashed')
print('Halbwertshöhe', 0.5*data_b[indexes_2[-1]])
print('Zehntelwertshöhe', 0.1*data_b[indexes_2[-1]])
plt.axhline(y=0.1*data_b[indexes_2[-1]], color='r',linestyle='dashed')
plt.xlim(0,2000)
plt.xlabel(r'E$_\gamma\:/\: \mathrm{keV}$')
plt.ylabel(r'Zählrate $N$')
plt.grid()
plt.legend()
plt.savefig('build/Cs.pdf')
plt.yscale('log')
plt.savefig('build/Cs_log.pdf')
plt.clf()

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


inhalt_comp = quad(compton2,a=lin(0,*params),b=lin(indexes_2[-2],*params))
print(f'Der Inhalt des Compton-Kontinuums, liegt bei: {inhalt_comp[0]}')

mu_ph = 0.002 #in cm^-1
mu_comp = 0.38
l=3.9
abs_wahrsch_ph = 1-np.exp(-mu_ph*l)
abs_wahrsch_comp = 1-np.exp(-mu_comp*l)
print(f'Die absolute Wahrscheinlichkeit eine Vollenergiepeaks liegt bei: {abs_wahrsch_ph} Prozent')
print(f'Die absolute Wahrscheinlichkeit eine Comptonpeaks liegt bei: {abs_wahrsch_comp} Prozent')


#------------------Aufgabenteil d) {Barium oder Antimon? Wir werden es erfahren.}
#Betrachte zuerst Orginalaufnahmen des Detektors
data_d = np.genfromtxt('data/Sb_Ba.txt', unpack=True)
x_plot = np.linspace(1, 8192, 8192)
plt.bar(x_plot, data_d)
plt.xlim(0, 8192)
plt.savefig('build/Ba_Sb_orginal.pdf')
plt.yscale('log')
plt.savefig('build/Ba_Sb_orginal_log.pdf')
plt.clf()

#Finde höchste Peaks und ordne sie den passenden Energien des Spektrums zu
peaks_3 = find_peaks(data_d, height=70, distance=15)
indexes_3 = peaks_3[0]
peak_heights_3 = peaks_3[1]
energie_3 = lin(indexes_3,*params)
print(indexes_3)
print(energie_3)

x=np.linspace(1,8192,8192)
plt.plot(x, data_d,'r-',label='Detektor')
plt.plot(indexes_3,data_d[indexes_3],'bx',label='Peaks')
plt.xlabel(r'Kanäle$')
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

#Führe wieder einen Gauß-Fit in den Bins durch um den Peakinhalt zu bestimmen
E_ba, W_ba, peaks_ind_ba = np.genfromtxt('data/Sb_Ba_sortiert.txt', unpack=True)
index_ba, peakinhalt_ba, hoehe_ba, unter_ba, sigma_ba = gaussian_fit_peaks_d(peaks_ind_ba.astype('int'))

#Fasse Ergebnisse in Tabelle zusammen
make_table(
    header= ['$E$ / \kilo\electronvolt ', '$W$ / \%', 'Index $i$', '$E_i$ / \kilo\electronvolt '],
    data=[E_ba, W_ba, peaks_ind_ba, lin(peaks_ind_ba, *params)],
    places=[3.2, 2.1, 3.0, 3.2],
    caption ='Werte der zu erwartenden Peaks der Ba-Quelle. Dazu die erwarete Energie $E$, die Emissionswahrscheinlichkeit $W$, der zugeordnete Index $i$ und die gefittete Energie $E_i$.',
    label ='tab:Ba_erwartet',
    filename ='build/tables/Ba_erwartet.tex'
)

E_ba_det = []
for i in range(len(index_ba)):
    E_ba_det.append(lin(index_ba[i],*params))

#print(E_ba_det)
#Berechne aktivität der Quelle am Messtag
print(f'Daten zur Berechnung der Akivität: {E_ba_det}, {params2}')
A=peakinhalt_ba[2:]/(omega_4pi*W_ba[2:]*potenz(E_ba_det[2:],*params2)) #nur die mit E>150keV mitnehmen

A_det = []
for i in range(0,2):
    A_det.append(0)

for i in A:
    A_det.append(i)
print('A_det', A_det)
#print(unter_ba)
#print(peakinhalt_ba)

#Fasse Fit-Parameter in Tabelle zusammen
make_table(
    header= ['Bin-Index $i$', '$E_\gamma$ / \kilo\electronvolt', '$h_i$', '$\sigma_i$ / \kilo\electronvolt'],
    data=[index_ba, E_ba_det, hoehe_ba, sigma_ba],
    places=[(4.2, 1.2), (3.2, 1.2), (4.2, 2.2), (1.2, 1.2)],
    caption='Parameter des Gauß-Fits. Dabei ist $\sigma_i$ die Standardabweichung.',
    label='tab:Ba',
    filename='build/tables/Ba.tex'
)

#Trage Ergebnisse der Aktivitätsbestimmung in Tabelle ein
print(unter_ba, peakinhalt_ba, A_det)
#make_table(
#    header= ['$Z_i$', '$E_i$ / \kilo\electronvolt ', '$A_i$ / \\becquerel '],
#    data=[unter_ba, peakinhalt_ba, A_det],
#    places=[(2.2, 2.2), (4.2, 3.1), (4.0, 2.2)],
#    caption='Berechnete Aktivitäten für jeden Bin mit dazu benötigten Werten.',
#    label ='plt:aktivitaet_ba',
#    filename ='build/tables/aktivitaet_ba.tex'
#)

A_gem = ufloat(np.mean(noms(A)),np.mean(sdevs(A)))
print('gemittelte Aktivität',A_gem)


#-------------Aufgabenteil e) {Was das? Gucken wir mal}
data_e = np.genfromtxt('data/unbekannt.txt', unpack=True)

peaks_4 = find_peaks(data_e, height=50, distance=15)
indexes_4 = peaks_4[0]
peak_heights_4 = peaks_4[1]
energie_4 = lin(indexes_4,*params)
make_table(
    header=['Index $i$', '$Z_\\text{i}$', '$E_\\text{i}$ / \kilo\electronvolt'],
    data= [indexes_4, data_e[indexes_4], energie_4],
    places=[4.0, 3.1, 4.2],
    caption ='Zugeordnete Indizes, Zählrate $Z_\\text{i}$ und Energie $E_\\text{i}$ der gefundenen Peaks.',
    label='tab:last',
    filename ='build/tables/last.tex'
)


x=np.linspace(1,8192,8192)
plt.plot(lin(x,*params), data_e,'r-',label='Fit')
plt.plot(lin(indexes_4,*params),data_e[indexes_4],'bx',label='Peaks')
plt.xlabel(r'E$_\gamma\:/\: \mathrm{keV}$')
plt.ylabel(r'Zählrate $N$')
plt.xlim(0, 3300)
plt.legend()
plt.grid()
plt.savefig('build/unbekannt.pdf')
plt.clf()
