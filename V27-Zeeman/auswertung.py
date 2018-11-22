import os
import numpy as np
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import ufloat
import pint
from tab2tex import make_table
ureg = pint.UnitRegistry(auto_reduce_dimensions = True)
Q_ = ureg.Quantity


def dispersionsgebiet(wellenlaenge, d, n):
    return wellenlaenge**2 / (2*d) * (1 / (n**2 -1))**(0.5)


def aufloesung(wellenlaenge, L, n):
    return L / wellenlaenge * (n**2 -1)


def lande(S, L, J):
    return (3*J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))


def lande_factors():
    print('Lande-Faktoren')
    print(f'\t1D_2  {lande(0, 2, 2)}')
    print(f'\t1P_1  {lande(0, 1, 1)}')
    print(f'\t3S_1  {lande(1, 0, 1)}')
    print(f'\t3P_1  {lande(1, 1, 1)}')


def lummer_gehrcke_platte():
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


if __name__ == '__main__':

    if not os.path.isdir('build'):
        os.mkdir('build')

    lande_factors()
    lummer_gehrcke_platte()
