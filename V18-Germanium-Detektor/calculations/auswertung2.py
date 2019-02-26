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

#Plotten des vom Detektor aufgenommenen Spektrums + logarithmische y-Achse
x=np.linspace(1,8192,8192)
plt.bar(lin(x,*params), data, label='Messdaten' )
plt.bar(lin(indexes,*params),data[indexes])
plt.xlim(0, 1600)
plt.xlabel(r'Bin-Indizes')
plt.ylabel(r'Zählrate $N$')
plt.legend(loc='best')
plt.savefig('build/orginal_Eu.pdf')
plt.yscale('log')
plt.savefig('build/orginal_Eu_log.pdf')
plt.clf()

#Plotten der Eichung/Kalibrierung am Eu-Spektrum
x=np.linspace(250,3700,3450)
plt.plot(x, lin(x,*params),'r-',label='Fit')
#plt.plot(peaks_ind,E,'bx',label='Daten')
plt.errorbar(peaks_ind, E, yerr=20, fillstyle= None, fmt=' x', label='Daten')
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
    return a+h*np.exp(-((x-mu)/sigma)**2)

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


        params_gauss,covariance_gauss=curve_fit(gauss,np.arange(a,b+1),data[a:b+1],p0=[1,data[i],0,i-1])
        errors_gauss = np.sqrt(np.diag(covariance_gauss))

        sigma_fit=ufloat(params_gauss[0],errors_gauss[0])
        h_fit=ufloat(params_gauss[1],errors_gauss[1])
        a_fit=ufloat(params_gauss[2],errors_gauss[2])
        mu_fit=ufloat(params_gauss[3],errors_gauss[3])
        #print(h_fit*sigma_fit*np.sqrt(2*np.pi))
        if i == 311:
            plt.plot(np.arange(a, b+1), data[a:b+1], label='Daten')
            plt.plot(np.arange(a, b+1), gauss(np.arange(a, b+1), *params_gauss), label='Fit')
            plt.savefig('build/test.pdf')
            plt.clf()
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
a=ufloat(0.073+0.015, 0.001) #in m
r=ufloat(0.0225, 0) #in m
omega_4pi = (1-a/(a**2+r**2)**(0.5))/2
print('Raumwinkel',omega_4pi)

#Berechnung Detektoreffizienz für jeden Energiepeak
Q=[]
Z=[]
for i in range(len(W)):
    Z.append(np.sqrt(2*np.pi)*hoehe[i]*sigma[i])
    Q.append(Z[i]/(omega_4pi*A_jetzt*W[i]/100*3600))


#Erstellen einer Tabelle der Fit-Parameter des Gauß-Fits
make_table(
    header= ['$a$', '$h_i$', '$\mu_i$', '$\sigma_i$'],
    data=[unter, hoehe, index_f, sigma],
    caption='Parameter des durchgeführten Gauss-Fits pro Bin. Dabei ist $\mu$ der Mittelwert, $\sigma$ die Standardabweichnug, $h$ die Höhe und a der Zählraten-Offset.',
    label='tab:gauss_parameter',
    places=[(2.2, 1.2), (4.2, 6.2), (4.2, 3.2), (3.2, 3.2)],
    filename='build/tables/Gauss-Fit-Parameter.tex'
    )

#Erstellen einer Tabelle der Detektoreffizenz und den dazu verwendeten Werten
make_table(
    header=['$Z_i$', '$E_i$ / \kilo\electronvolt' ,'$Q_i$ / \\becquerel ', 'W/\%'],
    data=[Z, E_det, Q, W],
    caption = 'Peakhöhe, Energie und Detektoreffizenz als Ergebnis des Gaußfits.',
    label = 'tab:det_eff',
    places = [ (5.2, 6.2), (4.2, 3.2), (3.2, 3.2), 2.1],
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
#print('Daten für den Exponentialfit:')
#print(noms(Q), noms(E_det))
params2, covariance2= curve_fit(potenz,noms(E_det),noms(Q),sigma=sdevs(Q), p0=[1, 0.1, 0, 0.5])
errors2 = np.sqrt(np.diag(covariance2))
#Zusammenfassen der Fit-Parameter
a=ufloat(params2[0],errors2[0])
b=ufloat(params2[1],errors2[1])
c=ufloat(params2[2],errors2[2])
e=ufloat(params2[3],errors2[3])

#Ausgabe der Fit-Parameter
print('\nKalibrationswerte Potenzfunktion:')
print(f'Steigung a = {a}')
print(f'Verschiebung b = {b}')
print(f'Verschiebung c = {c}')
print(f'Exponent e = {e}')

#Plotten der Effizenz gegen die Energie mit Exponential-Fit-Funktion
x=np.linspace(1,1600,10000)
plt.plot(x, potenz(x,*params2),'r-',label='Fit')
plt.errorbar(E,noms(Q), yerr=sdevs(Q),fmt=' x', ecolor='b',label='Daten')
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
plt.xlabel(r'Energie $E$')
plt.ylabel(r'Zählrate $N$')
plt.savefig('build/spektrum_Cs.pdf')
plt.yscale('log')
plt.savefig('build/spektrum_Cs_log.pdf')
plt.clf()

#Finde Peaks in Spektrum und ordne sie der Energie zu
peaks_2 = find_peaks(data_b, height=60, distance=20)
indexes_2 = peaks_2[0]
peak_heights_2 = peaks_2[1]
energie_2 = lin(indexes_2, *params)
print('Indices der peaks: ', indexes_2)

#Identifiziere die charakteristischen Energie-Peaks
e_rueck=energie_2[-4]
print('Rueckstreu: ', energie_2[-4], indexes_2[-4])
e_compton=energie_2[-2]
print('Compton: ', energie_2[-2], indexes_2[-2])
e_photo=energie_2[-1]
print('Photo: ', energie_2[-1], indexes_2[-1])
#print(len(energie_2), len(indexes_2))
#print(e_rueck, e_compton, e_photo)
#print(indexes_2[-4], indexes_2[-2], indexes_2[-1])


e_photo = 661.59
m_e = 511
#Vergleiche zwischen gemessenen und theoretischen Werten der Peaks
e_compton_theo = 2*e_photo**2/(m_e*(1+2*e_photo/m_e))
vgl_compton = 1-e_compton/e_compton_theo
print(f'Ein Vergleich des theoretischen E_compton {e_compton_theo} mit dem gemessenen E_compton {e_compton}, beträgt: {vgl_compton} ')

e_rueck_theo = e_photo/(1+2*e_photo/m_e)
vgl_rueck = 1-e_rueck/e_rueck_theo
print(f'Ein Vergleich des theoretischen E_rueck {e_rueck_theo} mit dem gemessenen E_compton {e_rueck}, beträgt: {vgl_rueck} ')



#Führe wieder Gausß-Fit für den Vollenergiepeak durch, um Peakhöhe bestimmen zu können
a=indexes_2[-1].astype('int')-50
b=indexes_2[-1].astype('int')+50

params_gauss_b,covariance_gauss_b=curve_fit(gauss,lin(np.arange(a,b+1),*params),data_b[a:b+1],p0=[1,data_b[indexes_2[-1]],0,lin(indexes_2[-1]-0.1,*params)])
errors_gauss_b = np.sqrt(np.diag(covariance_gauss_b))


#Fasse Wert und Ungenauigkeit der Fit-Parameter wieder jeweils zusammen
sigma_fit=ufloat(params_gauss_b[0],errors_gauss_b[0])
h_fit=ufloat(params_gauss_b[1],errors_gauss_b[1])
a_fit=ufloat(params_gauss_b[2],errors_gauss_b[2])
mu_fit=ufloat(params_gauss_b[3],errors_gauss_b[3])

Z_photo=h_fit*sigma_fit*np.sqrt(2*np.pi)
print(f'Die Rate des Vollenergiepeaks liegt bei {Z_photo} keV.')

#-------------------------------------------------------------------------------

Z_3 = data_b[a:b+1]*sigma_fit*np.sqrt(2*np.pi)
plt.plot(lin(np.arange(a,b+1),*params), gauss(lin(np.arange(a,b+1),*params), *params_gauss_b)*noms(sigma_fit)*np.sqrt(2*np.pi), 'k-', label='Fit')
plt.errorbar(lin(np.arange(a,b+1),*params), noms(Z_3), yerr=sdevs(Z_3), fillstyle= None, fmt=' x', label='Daten')
plt.xlim(654, 660)
plt.axhline(y=0.5*data_b[indexes_2[-1]]*noms(sigma_fit)*np.sqrt(2*np.pi), xmin = 0.31, xmax = 0.675, color='g',linestyle='dashed', label='Halbwertsbreite')
plt.axhline(y=0.1*data_b[indexes_2[-1]]*noms(sigma_fit)*np.sqrt(2*np.pi), xmin = 0.158, xmax = 0.81,color='r',linestyle='dashed', label='Zehntelbreite')
plt.ylabel('Peakhöhe $Z$')
plt.xlabel('Energie $E$/keV')
plt.legend(loc='best')
plt.grid()
plt.savefig('build/test_2.pdf')
plt.clf()

print('\nVergleich Halb- zu Zehntelwertsbreite:')
#lin beschreibt noch die lineare Regression vom beginn der Auswertung
h_g = ufloat(2.2, 0.2)
print('Halbwertsbreite Gemessen: ', h_g)
h_t = np.sqrt(8*np.log(2))*sigma_fit
print('Halbwertsbreite Theorie: ', h_t)
z_g = ufloat(4.0, 0.3)
print('Zehntelbreite Gemessen: ', z_g)
z_t = np.sqrt(8*np.log(10))*sigma_fit
print('Zehntelbreite Theorie: ', z_t)

print('Verhältnis der Halbwertsbreiten Werte: ', (h_g-h_t)/h_t, 'und der Zehntelbreiten: ', (z_g-z_t)/z_t)
print('Verhältnis zwischen gemessener', z_g/h_g, '', z_t/h_t,'\n')



#-------------------------------------------------------------------------------
#Plotte das zugeordnete Cs-Spektrum und setze Horizontale bei Zehntel- und Harlbwertsbreite
x=np.linspace(1,8192,8192)
plt.plot(lin(x, *params), data_b,'r-',label='Fit')
plt.plot(lin(indexes_2, *params),data_b[indexes_2],'bx',label='Peaks')
plt.axhline(y=0.5*data_b[indexes_2[-1]], color='g',linestyle='dashed', label='Halbwertshöhe')
print('Halbwertshöhe', 0.5*data_b[indexes_2[-1]])
print('Zehntelwertshöhe', 0.1*data_b[indexes_2[-1]])
plt.axhline(y=0.1*data_b[indexes_2[-1]], color='r',linestyle='dashed', label='Zehntelhöhe')
plt.xlim(0,700)
plt.xlabel(r'E$_\gamma\:/\: \mathrm{keV}$')
plt.ylabel(r'Zählrate $N$')
plt.grid()
plt.legend(loc='upper right')
plt.savefig('build/Cs.pdf')
plt.yscale('log')
plt.savefig('build/Cs_log.pdf')
plt.clf()

inhalt_photo = ufloat(sum(data_b[1649-13:1649+9]*noms(sigma_fit)*np.sqrt(2*np.pi)), sum(np.sqrt(data_b[1649-13:1649+9]*noms(sigma_fit)*np.sqrt(2*np.pi))))
print('\nInhalt des Photo-Peaks: ', inhalt_photo)

min_ind_comp = 53
inhalt_comp = ufloat(sum(data_b[min_ind_comp:indexes_2[-2]]*noms(sigma_fit)*np.sqrt(2*np.pi)), sum(np.sqrt(data_b[min_ind_comp:indexes_2[-2]]*noms(sigma_fit)*np.sqrt(2*np.pi))))
print(f'Der Inhalt des Compton-Kontinuums, liegt bei: {inhalt_comp}')

mu_ph = ufloat(0.007, 0.003) #in cm^-1
mu_comp = ufloat(0.35, 0.07)
l=3.9
abs_wahrsch_ph = 1-unp.exp(-mu_ph*l)
abs_wahrsch_comp = 1-unp.exp(-mu_comp*l)
print(f'Die absolute Wahrscheinlichkeit eine Vollenergiepeaks liegt bei: {abs_wahrsch_ph} Prozent')
print(f'Die absolute Wahrscheinlichkeit eine Comptonpeaks liegt bei: {abs_wahrsch_comp} Prozent\n')


#------------------Aufgabenteil d) {Barium oder Antimon? Wir werden es erfahren.}
#Betrachte zuerst Orginalaufnahmen des Detektors
data_d = np.genfromtxt('data/Sb_Ba.txt', unpack=True)
x_plot = np.linspace(1, 8192, 8192)
plt.bar(x_plot, data_d)
plt.xlim(0, 8192)
plt.xlabel('')
plt.ylabel('Zählrate $Z$')
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
plt.plot(lin(x, *params), data_d,'r-',label='Detektor')
plt.plot(lin(indexes_3, *params),data_d[indexes_3],'bx',label='Peaks')
print('Peaks des Barium-Spektrums bei : ', lin(indexes_3, *params) , 'mit Inhalt: ', data_d[indexes_3])
plt.xlabel(r'Energie $E\:/\:$keV')
plt.ylabel(r'Zählrate $N$')
plt.xlim(0, 500)
plt.grid()
plt.legend()
plt.savefig('build/Ba_Sb.pdf')
plt.clf()


def gaussian_fit_peaks_d(test_ind):
    peak_inhalt = []
    index_fit = []
    hoehe = []
    unter = []
    sigma = []
    for i in test_ind:
        a=i-40
        b=i+40


        params_gauss_d,covariance_gauss_d=curve_fit(gauss,np.arange(a,b+1),data_d[a:b+1],p0=[1,data_d[i],0,i-1])
        errors_gauss_d = np.sqrt(np.diag(covariance_gauss_d))

        sigma_fit=ufloat(params_gauss_d[0],errors_gauss_d[0])
        h_fit=ufloat(params_gauss_d[1],errors_gauss_d[1])
        a_fit=ufloat(params_gauss_d[2],errors_gauss_d[2])
        mu_fit=ufloat(params_gauss_d[3],errors_gauss_d[3])
        #print(h_fit*sigma_fit*np.sqrt(2*np.pi))
        if i == 208:
            plt.plot(np.arange(a, b+1), data_d[a:b+1], label='Daten')
            plt.plot(np.arange(a, b+1), gauss(np.arange(a, b+1), *params_gauss_d), label='Fit')
            plt.savefig('build/test.pdf')
            plt.clf()
        index_fit.append(mu_fit)
        hoehe.append(h_fit)
        unter.append(a_fit)
        sigma.append(sigma_fit)
        peak_inhalt.append(h_fit*sigma_fit*np.sqrt(2*np.pi))
    return index_fit, peak_inhalt, hoehe, unter, sigma

#Führe wieder einen Gauß-Fit in den Bins durch um den Peakinhalt zu bestimmen
E_ba, W_ba, peaks_ind_ba, E_ba_echt= np.genfromtxt('data/Sb_Ba_sortiert.txt', unpack=True)
index_ba, peakinhalt_ba, hoehe_ba, unter_ba, sigma_ba = gaussian_fit_peaks_d(peaks_ind_ba.astype('int'))


#Fasse Ergebnisse in Tabelle zusammen
make_table(
    header= ['$E$ / \kilo\electronvolt ', '$W$ / \%', '$E_i$ / \kilo\electronvolt '],
    data=[E_ba, W_ba, E_ba_echt],
    places=[3.2, 2.1, 3.2],
    caption ='Werte der zu erwartenden Peaks der Ba-Quelle. Dazu die erwarete Energie $E$, die Emissionswahrscheinlichkeit $W$, der zugeordnete Index $i$ und die gefittete Energie $E_i$.',
    label ='tab:Ba_erwartet',
    filename ='build/tables/Ba_erwartet.tex'
)

E_ba_det = []
for i in range(len(index_ba)):
    E_ba_det.append(lin(index_ba[i],*params))

#print(E_ba_det)
#Berechne aktivität der Quelle am Messtag
#print(f'\nDaten zur Berechnung der Akivität: {E_ba_det}, {params2}')
A=peakinhalt_ba[2:]/(omega_4pi*W_ba[2:]/100*potenz(E_ba_det[2:],*params2)) #nur die mit E>150keV mitnehmen

A_det = []
for i in range(0,2):
    A_det.append(0)

for i in A:
    A_det.append(i)
#print('A_det', A_det)
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

#berechnung der neuen Effizienz
Z_d = [0, 0]
Q_d = []
for i in range(2, len(W_ba)):
    Z_d.append(np.sqrt(2*np.pi)*hoehe_ba[i]*sigma_ba[i])
    Q_d.append(Z[i]/(omega_4pi*A_det[i]*W_ba[i]/100*3600))
print('Z_d: ', Z_d)

#Trage Ergebnisse der Aktivitätsbestimmung in Tabelle ein
#make_table(
#    header= ['$W$\/\%', '$Z_i$', '$E_i$ / \kilo\electronvolt ', '$A_i$ / \\becquerel '],
#    data=[W_ba,  unter_ba, peakinhalt_ba, A_det],
#    places=[2.1, (2.2, 2.2), (4.2, 3.1), (4.0, 2.2)],
#    caption='Berechnete Aktivitäten für jeden Bin mit dazu benötigten Werten.',
#    label ='plt:aktivitaet_ba',
#    filename ='build/tables/aktivitaet-ba.tex'
#)
make_table(
    header= ['$W\/\%$', '$Z_i$', '$E_i\/$ \kilo\electronvolt', '$A_i\/$\\becquerel'],
    data=[W_ba, Z_d, peakinhalt_ba, A_det],
    places=[2.1, 4.2, 4.2, 4.0],
    caption='Berechnete Aktivität für jeden bin mit dazu benötigten Werten.',
    label='tab:aktivitaet_ba',
    filename='build/tables/aktivitaet_ba.tex'
)

A_gem = ufloat(np.mean(noms(A)),np.mean(sdevs(A)))
#print('gemittelte Aktivität',A_gem)


#-------------Aufgabenteil e) {Was das? Gucken wir mal}
data_e = np.genfromtxt('data/unbekannt.txt', unpack=True)

peaks_4 = find_peaks(data_e, height=50, distance=15)
indexes_4 = peaks_4[0]
peak_heights_4 = peaks_4[1]
energie_4 = lin(indexes_4,*params)
print(energie_4)
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
