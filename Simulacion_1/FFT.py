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
from numpy.fft import fft,fftfreq
from FabryPerot.FFT_support import recorte_frec_negativas_fft,calcular_verdadera_amplitud

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


fft_reflectancia = fft(reflectancia)

# Cambio de variable 

beta = 1/lambda_ # Unidades de beta [beta] = 1/nm

# Periodo de muestreo 

T_muestreo_beta = beta[0] - beta[1]
 

"""
Esta funcion calcula el vector de "frecuencias" de la transformada de fourier, vease

https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html

Es el analogo a la funcion vfreq realizada en mi github: 
    
    https://github.com/AlejandroGoper/Fundamento_de_Procesamiento_Digital_de_Senales/blob/main/Tarea5/Codigo/Tarea5.ipynb

Es importante notar el escalamiento de los ejes, dado que en esta nueva variable beta
el espaciado entre cada frecuencia es del orden de 10**8 nanometros, por lo que debemos
dividir el vector de frecuencias por un factor de 10**6 para que convierta los nanometros 
a milimetros, ademas dado que en general tenemos 

x(beta) = cos[2pi * (2OPL) * beta] donde beta es la variable independiente beta = 1/lambda

en el espectro de fourier los picos de "frecuencias" estaran ubicados en +-2OPL 

si queremos que cada pico de frecuencia diga el OPL directo, debemos agregar un factor de
1/2 adicional al vector de frecuencias

Todo esto podemos realizarlo multiplicando T_muestreo_beta*(2*10**6) 
"""

vfreq_np = fftfreq(len(fft_reflectancia),(T_muestreo_beta)*((2*10**6)))

#reflectancia_db = 10*np.log10(obj.R())


magnitud_fft = calcular_verdadera_amplitud(fft=fft_reflectancia)


vfreq_positivas = recorte_frec_negativas_fft(fft=vfreq_np)

# Graficando FFT

plt.figure()
plt.plot(vfreq_positivas,magnitud_fft)
plt.title(label="Espectro de magnitud de Fourier")
plt.xlabel(xlabel=r"OPL [$mm$] ")
plt.ylabel(ylabel=r"$R$")
#plt.plot(lambda_,10*np.log10(reflectancia))
#plt.plot(lambda_,reflectancia)
plt.xlim([-0.1,4])
plt.show()

