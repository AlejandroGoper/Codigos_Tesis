#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:32:08 2021

@author: alejandro_goper

Script para aplicar filtrado a la señal de la reflectancia

"""

import numpy as np
import matplotlib.pyplot as plt
from FabryPerot.FFT_support import calcular_verdadera_amplitud, recorte_frec_negativas_fft


# Parametros del filtro
N = 81 # Orden del filtro
M = N-1 
w_c = 80 # frecuencia de corte OPL [mm]

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
plt.title("Respuesta en el tiempo del filtro ideal")
plt.show()

# Segun la referencia: http://profesores.elo.utfsm.cl/~mzanartu/IPD414/Docs/ipd414-c05c.pdf
# La ventana de Hanning presenta una menor atenuacion que la de gauss asi que se probara 

# Construyendo las ventanas centradas en M/2 muestras

w_n_hanning =  0.5 - 0.5*np.cos((2*np.pi*n)/M)
sigma = 0.4 # sigma menor que 0.5
w_n_gauss = np.exp(-0.5*((2*n-M)/(sigma*M))**2)

# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
fig.suptitle("Definicion de las ventanas")
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(1,2,1)
ax.stem(w_n_hanning)
ax.set_title("Ventana de Hanning")
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"w[$n$]")
ax = plt.subplot(1,2,2)
ax.stem(w_n_gauss)
ax.set_title(r"Ventana de gauss $\sigma = 0.4$")
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"w[$n$]")
plt.show()


# Multiplicar la ventana w_n con la respuesta en el tiempo del filtro ideal para truncar
s_n_hanning = w_n_hanning*h_n
s_n_hanning /= sum(s_n_hanning)


s_n_gauss = w_n_gauss*h_n
s_n_gauss /= sum(s_n_gauss)



# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
fig.suptitle("Eventanado")
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(1,2,1)
ax.stem(s_n_hanning)
ax.set_title("Truncamiento con hanning")
ax.set_xlabel(r"$n$")
# Aqui s[n] = w[n]h[n]
ax.set_ylabel(r"s[$n$]")
ax = plt.subplot(1,2,2)
ax.stem(s_n_gauss)
ax.set_title(r"Truncamiento con gauss $\sigma = 0.4$")
ax.set_xlabel(r"$n$")
# Aqui s[n] = w[n]h[n]
ax.set_ylabel(r"s[$n$]")
plt.show()

print(sum(s_n_gauss), sum(s_n_hanning))

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

t = np.arange(0,15,0.01)
senal = np.cos(2*np.pi*0.5*t) + np.cos(2*np.pi*1.3*t) +1*np.cos(2*np.pi*5*t)

vfreq = recorte_frec_negativas_fft(np.fft.fftfreq(len(senal),0.01))

fft_senal = calcular_verdadera_amplitud(np.fft.fft(senal)) 

#from scipy.signal import filtfilt
#senal_filtrada = filtfilt(b=h_filtor_gauss, a=1, x=senal)

senal_filtrada = np.convolve(h_filtor_gauss/sum(h_filtor_gauss), senal, mode="same")

fft_senal_filtered = calcular_verdadera_amplitud(np.fft.fft(senal_filtrada))

vfreq_f = recorte_frec_negativas_fft(np.fft.fftfreq(len(senal_filtrada),0.01))

# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(3,2,1)
ax.plot(t,senal)
ax.set_title("Señal")
ax = plt.subplot(3,2,2)
ax.plot(vfreq,fft_senal)
ax.set_title(r"Espectro de amplitud")
ax.set_xlim([0,6])
ax = plt.subplot(3,2,3)
ax.plot(h_filtor_gauss)
ax.set_title(r"Ventana de Gauss")
ax = plt.subplot(3,2,4)
ax.plot(t,senal_filtrada)
ax.set_title(r"Señal filtrada")
ax = plt.subplot(3,2,5)
ax.plot(fft_senal_filtered)
ax.set_title(r"Espectro de amplitud SF")
ax.set_xlim([0,80])
plt.show()





"""
import scipy.signal as sc

# Creacion de una señal cualquiera 

t = np.arange(0,15,0.01)
senal = np.cos(2*np.pi*0.5*t) + np.cos(2*np.pi*1.3*t)

fft_senal = calcular_verdadera_amplitud(np.fft.fft(senal)) 

h = sc.windows.gaussian(81,4)

h_n = h/sum(h)

filtered = np.convolve(h_n,senal,mode="valid")

fft_senal_filtered = calcular_verdadera_amplitud(filtered)

# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(3,2,1)
ax.plot(t,senal)
ax.set_title("Señal")
ax = plt.subplot(3,2,2)
ax.plot(fft_senal)
ax.set_title(r"Espectro de amplitud")
ax.set_xlim([0,50])
ax = plt.subplot(3,2,3)
ax.plot(h_n)
ax.set_title(r"Ventana de hann")
ax = plt.subplot(3,2,4)
ax.plot(filtered)
ax.set_title(r"Señal filtrada")
ax = plt.subplot(3,2,5)
ax.plot(fft_senal_filtered)
ax.set_title(r"Espectro de amplitud SF")
ax.set_xlim([0,50])
plt.show()
"""


