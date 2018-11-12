import os
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
import matplotlib.pyplot as plt
import datetime
from astropy.io import ascii
from scipy.optimize import curve_fit
import scipy.constants as codata
from tab2tex import make_table
import pint
Q_ = pint.UnitRegistry().Quantity

def eichung():
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


if __name__ == '__main__':

    if not os.path.isdir('build'):
        os.mkdir('build')

    eichung()
