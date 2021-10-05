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
from numpy.fft import fft,fftshift,ifftshift


"""
==============================================================================

Simulación de la serie discreta de fourier (FFT) de la señal obtenida con 
la Reflectancia R como funcion de la longitud de onda en un 
Interferometro de Fabry-Perot en serie de 2 avidades (GAP).

==============================================================================
"""

# Periodo de muestreo
T_muestreo = 0.01 # nm

lambda_inicial = 1500 # nm
lambda_final = 1600 # nm

# Tiempo de grabacion
t = lambda_final - lambda_inicial # nm

# Definicion del dominio en longitudes de onda
lambda_ = np.arange(lambda_inicial,lambda_final, T_muestreo) #nanometros


# Construyendo señal a analizar en el dominio de fourier
obj = FabryPerot_2GAP(lambda_inicial=1500,lambda_final= 1600,L_medio_1 = 0.4, L_medio_2=0.8, eta_medio_1 = 1.0, eta_medio_2 = 1.332, eta_medio_3=1.48)
reflectancia = obj.R()


fft_reflectancia = fft(reflectancia)

# Cambio de variable 

beta = 1/lambda_ # Unidades de beta [beta] = 1/nm

# Periodo de muestreo 

T_muestreo_beta = beta[0] - beta[1]
 

def calcVecFrec(datos,PeriodoMuestreo):
    #Esta funcion calcula el vector de los valores de la frecuencia (en numero de datos/nm) en el
    #espacio de Fourier de una secuencia guardada en el vector datos,
    #PeriodoMuest, es el tiempo transcurrido entre medición y medición
    N = len(datos) # Calculamos la longitud de los datos
    TiempoCompleto=PeriodoMuestreo*N #Calculamos el tiempo total (de medición) de la señal
    #Verificamos si la cantidad de datos es par o impar
    if(N%2 == 0):
        # Construímos un vector que va de -int(N/2) hasta int(N/2)-1 en incrementos de 1 en 1
        vfreq = np.arange(-np.floor(N/2),np.floor(N/2),1)
    else:
        # Construimos un vector que va de -int(N/2) hasta int(N/2) en incrementos de 1
        vfreq = np.arange(-np.floor(N/2),np.floor(N/2)+1,1)
    # Hasta este punto las unidades de vfreq serán (repeticiones)
    # Queremos convertirlas a (1/(1/nm)) como en el caso de la fft, así que dividiremos
    # entre el tiempo completo de medicion
    vfreq /= (2*(1*10**6)*TiempoCompleto) # Aqui las unidades son (nm)
    
    # Corremos el arreglo de tal forma que quede identico a la parte del dominio de la gráfica de
    # la transformada de fourier

    # Lo que hace ifffshift es correr un arreglo hacia la izquierda por ejemplo:
    # si vfreq = [-2,-1,0,1,2] ---> [-1,0,1,2,-2] ---> [0,1,2,-2,-1]
    vfreq = ifftshift(vfreq) # Corremos el arreglo
    return vfreq

vfreq =  calcVecFrec(datos=fft_reflectancia, PeriodoMuestreo=T_muestreo_beta)


"""
    Por I. Alejandro Gómez Pérez.
    
    La lógica de esta función es la siguiente:
    Toma unicamente los valores positivos de la fft (indices desde 0 hasta int(n/2)-1 [si n es par] 
    o desde 0 hasta int(n/2) [si n es impar]) duplica los valores de todo este arreglo (para así
    tomar en cuenta los valores negativos de las frecuencias) y finalmente dividimos entre el numero
    total de datos de la fft para encontrar así el valor de la amplitud correcta.
    
    ** Para construir el vector se uso una propiedad llamada slicing de los arreglos de python.
    Parámetros:
    fft: array que contiene la fft de alguna señal
    Regresa:
    vector: array con las contribuciones positivas de la fft con sus amplitudes correctas
"""
from numpy import floor,abs

def calculaVerdaderaAmplitud(fft):
    n = len(fft)
    if(n%2 == 0):
        lim = floor(n/2)
    else:
        lim = floor(n/2)+1
    vector = 2*abs(fft[0:int(lim)])/n
    vector[0] /= 2
    return vector


#reflectancia_db = 10*np.log10(obj.R())


v = calculaVerdaderaAmplitud(fft=fft_reflectancia)


n = len(fft_reflectancia)
if(n%2 == 0):
    lim = floor(n/2)
else:
    lim = floor(n/2)+1
vfreqpos = vfreq[0:int(lim)]


# Graficando FFT

plt.figure()
plt.stem(vfreqpos,v)
plt.title(label="FFT de la reflectancia R (Normalizada)")
plt.xlabel(xlabel=r"$mm$")
plt.ylabel(ylabel=r"$|R|$")
#plt.plot(lambda_,10*np.log10(reflectancia))
#plt.plot(lambda_,reflectancia)
plt.xlim([-0.1,4])
plt.show()

