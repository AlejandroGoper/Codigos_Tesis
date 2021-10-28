#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:32:08 2021

@author: alejandro_goper

Script para aplicar filtrado a la señal de la reflectancia

"""

import numpy as np
import matplotlib.pyplot as plt
from FabryPerot.FFT_support import calcular_verdadera_amplitud



# Parametros del filtro
N = 81 # Orden del filtro
M = N-1 
w_c = 2.0 # frecuencia de corte OPL [mm]

# Dominio de la secuencia S[n]
n = np.arange(0,N)
# Calculando los coeficientes de la respuesta al impulso ideal de un filtro pasa bajos
# Es un seno cardinal
h_n = np.sin(w_c*(n-M/2))/(np.pi*(n-M/2))
# Agregando la contribucion central del seno cardinal (dado que no esta definida)
h_n[int(M/2)] = w_c/np.pi

# Graficando para ver que sucede
plt.figure()
plt.stem(h_n)
plt.title("Respuesta al impulso del filtro ideal")
plt.show()

# Segun la referencia: http://profesores.elo.utfsm.cl/~mzanartu/IPD414/Docs/ipd414-c05c.pdf
# La ventana de Hamming presenta una menor atenuacion que la de gauss asi que se probara 

ventana_hamming =  0.5 - 0.5*np.cos((2*np.pi*n)/M)
sigma = 0.1
ventan_gauss = np.exp(-0.5*((2*n-M)/(sigma*M))**2)

# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(1,2,1)
ax.stem(ventana_hamming)
ax.set_title("Ventana de Hamming")
ax = plt.subplot(1,2,2)
ax.stem(ventan_gauss)
ax.set_title(r"Ventana de gauss $\sigma = 0.1$")
plt.show()

# Multiplicar los espectros, la ventana con el seno cardinal

h_filtro_hamming = ventana_hamming*h_n
h_filtor_gauss = ventan_gauss*h_n

# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(1,2,1)
ax.stem(h_filtro_hamming)
ax.set_title("Respuesta a Ventana de Hamming")
ax = plt.subplot(1,2,2)
ax.stem(h_filtor_gauss)
ax.set_title(r" Respuesta a Ventana de gauss $\sigma = 0.1$")


plt.show()

# Graficando la respuesta en el espacio de las frecuencias FFT

fft_filtro_hamming = np.fft.fft(h_filtro_hamming)
fft_filtro_gauss = np.fft.fft(h_filtor_gauss)

# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(1,2,1)
ax.plot(calcular_verdadera_amplitud(fft_filtro_hamming))
ax.set_title("Espectro amp de Hamming")
ax = plt.subplot(1,2,2)
ax.plot(calcular_verdadera_amplitud(fft_filtro_gauss))
ax.set_title(r"Espectro amp de gauss $\sigma = 0.1$")

plt.show()


# Creacion de una señal cualquiera 

t = np.arange(0,100,0.1)
signal = np.cos(2*np.pi*0.5*t) + np.cos(2*np.pi*1.3*t)
fft_senal = calcular_verdadera_amplitud(np.fft.fft(signal))

senal_filtrada = fft_filtro_hamming*np.fft.fft(signal)

# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(1,2,1)
ax.plot(t,signal)
ax.set_title("Signal")
ax = plt.subplot(1,2,2)
ax.plot(calcular_verdadera_amplitud(senal_filtrada))
ax.set_title(r"Espectro de signal")
plt.show()



