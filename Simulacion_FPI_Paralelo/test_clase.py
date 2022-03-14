#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 02:09:14 2022

@author: alejandro_goper
"""

"""
=============================================================================
Script para probar la simulacion de FPI 2GAP en paralelo
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from FabryPerot.Clase import FPI_2GAP_parallel
from FabryPerot.FFT_support import encontrar_FFT_dominio_en_OPL

"""
==============================================================================
Vamos a considerar una sistema:
    Fibra - n_0 = 1.45
    Aire - n_1 = 1.003
    Vidrio - n_2 = 1.65
    Fibra - n_2 = 1.45
==============================================================================
"""

# Definicion del dominio en longitudes de onda
T_muestreo_lambda = 0.005 # nm 
lambda_ = np.arange(1510,1590,0.005) #nanometros

# Indices del primer interferometro
n_1 = [1.45, 1.003, 1.65, 1.45]
# Indices del segundo interferometro
n_2 = [1.45, 1.003, 1.65, 1.45]

# Longitud de cavidades del primer interferometro [mm]
L_1 = [1, 0.5]
# Longitud de cavidades del segundo interferometro [mm]
L_2 = [2, 1.5]


# Parametros de perdida en el primer interferometro
A_1 = [0,0]
a_1 = [0,0]
# Parametros de perdida en el segundo interferomtro
A_2 = [0,0]
a_2 = [0,0]

obj = FPI_2GAP_parallel(lambda_inicial=1510, 
                        lambda_final = 1590, 
                        T_muestreo_lambda= T_muestreo_lambda, 
                        L_i1= L_1,
                        L_i2= L_2,
                        n_i1= n_1, 
                        n_i2= n_2, 
                        alpha_i1= a_1,
                        alpha_i2= a_2,
                        A_i1= A_1,
                        A_i2= A_2)

intensidad = obj.I_out()

opl, amp = encontrar_FFT_dominio_en_OPL(lambda_inicial=1510, lambda_final=1590, senal=intensidad)


fig, ax = plt.subplots(figsize=(40,20))
#plt.plot(lambda_,10*np.log10(reflectancia))

ax = plt.subplot(1,2,1)
ax.plot(lambda_,intensidad)

ax = plt.subplot(1,2,2)
ax.plot(opl,amp)
ax.set_xlim([0,4])
plt.show()