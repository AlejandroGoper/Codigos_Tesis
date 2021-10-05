#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:19:04 2021

@author: alejandro_goper


Codigo para la transformada de Fourier de la señal obtenida en main

"""

import numpy as np
import matplotlib.pyplot as plt
from FabryPerot.Clase import FabryPerot_2GAP
from FabryPerot.FFT_support import encontrar_FFT

"""
==============================================================================

Obtención de la serie discreta de fourier (FFT) de la señal obtenida con 
la Reflectancia R como funcion de la longitud de onda en un 
Interferometro de Fabry-Perot en serie de 2 cavidades (GAP).

==============================================================================
"""

lambda_inicial = 1500 #nm

lambda_final = 1600 #nm

T_muestreo_lambda = 0.01 #nm

# Definicion del dominio en longitudes de onda
lambda_ = np.arange(lambda_inicial,lambda_final, T_muestreo_lambda) #nanometros


# Construyendo señal a analizar en el dominio de fourier
obj = FabryPerot_2GAP(lambda_inicial=lambda_inicial,lambda_final= lambda_final,L_medio_1 = 0.4, L_medio_2=0.8, eta_medio_1 = 1.0, eta_medio_2 = 1.332, eta_medio_3=1.48)
reflectancia = obj.R()

x,y = encontrar_FFT(lambda_inicial=lambda_inicial, T_muestreo_lambda=T_muestreo_lambda, Reflectancia=reflectancia)

# Graficando FFT

plt.figure()
plt.plot(x,y)
plt.title(label="Espectro de magnitud de Fourier")
plt.xlabel(xlabel=r"OPL [$mm$] ")
plt.ylabel(ylabel=r"$R$")
#plt.plot(lambda_,10*np.log10(reflectancia))
#plt.plot(lambda_,reflectancia)
plt.xlim([-0.1,4])
plt.show()

