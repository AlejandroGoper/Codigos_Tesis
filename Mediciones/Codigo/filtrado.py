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
from scipy.signal import filtfilt


# Parametros del filtro
N = 91 # Orden del filtro
M = N-1 
w_c = 8*np.pi/(50)  # frecuencia de corte OPL [mm]
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
# Normalizando para que la suma sea 1
s_n_hanning /= sum(s_n_hanning)


s_n_gauss = w_n_gauss*h_n
# Normalizando para que la suma sea 1
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

# Graficando la respuesta en el espacio de las frecuencias FFT

fft_s_n_hanning = np.fft.fft(s_n_hanning)
fft_s_n_gauss = np.fft.fft(s_n_gauss)

# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(1,2,1)
ax.plot(recorte_frec_negativas_fft(np.fft.fftfreq(len(fft_s_n_hanning),0.01)),np.log10(calcular_verdadera_amplitud(fft_s_n_hanning)))
ax.set_xlabel("w")
ax.set_title("Espectro amp de Hamming")
ax = plt.subplot(1,2,2)
ax.plot(recorte_frec_negativas_fft(np.fft.fftfreq(len(fft_s_n_gauss),0.01)) ,np.log10(calcular_verdadera_amplitud(fft_s_n_gauss)))
ax.set_xlabel("w")
ax.set_title(r"Espectro amp de gauss $\sigma = 0.4$")

plt.show()


# Creacion de una señal cualquiera 

t = np.arange(0,15,0.01)
senal = np.cos(2*np.pi*1*t) + np.cos(2*np.pi*2*t) + 1*np.cos(2*np.pi*10*t)

vfreq = recorte_frec_negativas_fft(np.fft.fftfreq(len(senal),0.01))

fft_senal = calcular_verdadera_amplitud(np.fft.fft(senal)) 

#from scipy.signal import filtfilt
#senal_filtrada = filtfilt(b=h_filtor_gauss, a=1, x=senal)

senal_filtrada_hanning = np.convolve(s_n_hanning, senal, mode="same")
#senal_filtrada_hanning = filtfilt(s_n_hanning,1,senal, method="gust")

fft_senal_filtrada_hanning = calcular_verdadera_amplitud(np.fft.fft(senal_filtrada_hanning))

vfreq_filt_hanning = recorte_frec_negativas_fft(np.fft.fftfreq(len(senal_filtrada_hanning),0.01))


senal_filtrada_gauss = np.convolve(s_n_gauss, senal, mode="same")
#senal_filtrada_gauss = filtfilt(s_n_gauss,1,senal,method="gust")

fft_senal_filtrada_gauss = calcular_verdadera_amplitud(np.fft.fft(senal_filtrada_gauss))
vfreq_filt_gauss = recorte_frec_negativas_fft(np.fft.fftfreq(len(senal_filtrada_gauss),0.01))



# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(3,2,1)
ax.plot(t,senal)
ax.set_title("Señal original")
ax = plt.subplot(3,2,2)
ax.plot(vfreq,fft_senal)
ax.set_title(r"Espectro de amplitud")
#ax.set_xlim([0,10])
ax = plt.subplot(3,2,3)
ax.plot(t,senal_filtrada_hanning)
ax.set_title(r"Señal filtrada Hanning")
ax = plt.subplot(3,2,4)
ax.plot(vfreq_filt_hanning, fft_senal_filtrada_hanning)
ax.set_title(r"Espectro de amplitud Hanning")
#ax.set_xlim([0,10])
ax = plt.subplot(3,2,5)
ax.plot(t,senal_filtrada_gauss)
ax.set_title(r"Señal filtrada Gauss")
ax = plt.subplot(3,2,6)
ax.plot(vfreq_filt_gauss,fft_senal_filtrada_gauss)
ax.set_title(r"Espectro de amplitud Gauss")
#ax.set_xlim([0,10])
plt.show()


# Probando clase 

from FabryPerot.Filtros_support import Filtro

filt = Filtro(_senal=senal , _T_muestreo=0.01, _frec_corte= 8, _orden=91)

senal_filt_clase_gauss = filt.filtrar_por_ventana_de_gauss(0.4)

senal_filt_clase_hanning = filt.filtrar_por_ventana_de_hanning()

fft_senal_filtrada_gauss_clase = calcular_verdadera_amplitud(np.fft.fft(senal_filt_clase_gauss))
vfreq_filt_gauss_clase = recorte_frec_negativas_fft(np.fft.fftfreq(len(senal_filt_clase_gauss),0.01))


fft_senal_filtrada_hanning_clase = calcular_verdadera_amplitud(np.fft.fft(senal_filt_clase_hanning))
vfreq_filt_hanning_clase = recorte_frec_negativas_fft(np.fft.fftfreq(len(senal_filt_clase_hanning),0.01))


# Graficando para ver que sucede
fig, ax = plt.subplots()
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
ax = plt.subplot(3,2,1)
ax.plot(t,senal)
ax.set_title("Señal original")
ax = plt.subplot(3,2,2)
ax.plot(vfreq,fft_senal)
ax.set_title(r"Espectro de amplitud")
#ax.set_xlim([0,10])
ax = plt.subplot(3,2,3)
ax.plot(t,senal_filt_clase_hanning)
ax.set_title(r"Señal filtrada Hanning clase")
ax = plt.subplot(3,2,4)
ax.plot(vfreq_filt_hanning_clase, fft_senal_filtrada_hanning_clase)
ax.set_title(r"Espectro de amplitud Hanning")
#ax.set_xlim([0,10])
ax = plt.subplot(3,2,5)
ax.plot(t,senal_filt_clase_gauss)
ax.set_title(r"Señal filtrada Gauss")
ax = plt.subplot(3,2,6)
ax.plot(vfreq_filt_gauss_clase,fft_senal_filtrada_gauss_clase)
ax.set_title(r"Espectro de amplitud Gauss")
#ax.set_xlim([0,10])
plt.show()