#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:28:54 2021

@author: alejandro_goper


Script para pruebas de filtrado de espectros obtenidos en el laboratorio


Documentacion para los colores de las graficas:
    - https://matplotlib.org/stable/tutorials/colors/colors.html


"""

from FabryPerot.Filtros_support import Filtro, ventana_de_gauss, ventana_de_hanning
from FabryPerot.FFT_support import encontrar_FFT
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt



# Importando un espectro del fabry perot: 2GAP-VIDRIO-AIRE-100um - ESPECTRO (5)

# Importando archivos 

fecha_medicion = "09-11-2021"

carpeta = "2GAP-VIDRIO-AIRE-100um"

ruta_directorio = "../" + fecha_medicion + "/" + carpeta

nombre_archivo = "Espectro (7).txt"

path = ruta_directorio + "/" + nombre_archivo

data = np.loadtxt(path, skiprows=0)

path = ruta_directorio + "/referencia.txt"

referencia = np.loadtxt(path)

# Seprando datos de referencia por columnas

lambda_ref, potencia_dBm_ref = referencia[:,0], referencia[:,1]

# Separando datos por columnas
lambda_ = data[:,0]
potencia_dBm = data[:,1]


# Normalizando respecto a la referencia

potencia_dB = potencia_dBm - potencia_dBm_ref

# Definiendo limite de busqueda en el espectro de Fourier
lim_inf = 0
lim_sup = 3

# Implementaremos un KNN para encontrar el vecino mas cercano al limite de
# busqueda que hemos obtenido en el array del opl en el dominio de fourier
nn = NearestNeighbors(n_neighbors=1)


# Creando figura
fig,ax = plt.subplots(figsize=(40,20))
fig.set_tight_layout(True)
# Pone lo mas juntas las graficas posibles
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)

# Cambiando el tamano de la fuente en todos los ejes
plt.rcParams.update({'font.size': 20})

# Graficando el espectro 
ax = plt.subplot(3,2,1)
espectro_graph, = ax.plot(lambda_,potencia_dB, linewidth=1.5, label= "Medición")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$dB$", fontsize=30)
ax.set_title(label="Dominio óptico", fontsize=30)
#ax.set_ylim([-40,-10])
ax.legend(loc="best",fontsize=30)

# Graficando la FFT
#T_muestreo_lambda = lambda_[3] - lambda_[2] # Approx 0.005 nm
T_muestreo_lambda = 0.005
       
# Calculando la FFT
opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_dB)    

# Encontrando el vecino mas cercano a los limites de busqueda en amp
nn.fit(opl.reshape((len(amp),1)))

index_lim_inf = nn.kneighbors([[lim_inf]], 1, return_distance=False)[0,0]
index_lim_sup = nn.kneighbors([[lim_sup]],1 , return_distance=False)[0,0]

lim_inf = opl[index_lim_inf]
lim_sup = opl[index_lim_sup]


ax = plt.subplot(3,2,2)
fft_graph, = ax.plot(opl,amp, linewidth=1.5,color="purple")
ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$|dB|$", fontsize=30)
ax.set_title(label="Dominio de Fourier", fontsize=30)
ax.set_xlim([lim_inf,lim_sup])
#ax.set_ylim([0,1])

lambda_inicial = lambda_[0]

# Al realizar el cambio de variable beta = 1/lambda, tenemos que 
T_muestreo_beta = T_muestreo_lambda / (lambda_inicial*(lambda_inicial+T_muestreo_lambda))

filtro = Filtro(_senal=potencia_dB, _T_muestreo=T_muestreo_beta*(2*10**6), _frec_corte=3, _orden=901)
senal_filtrada = filtro.filtrar_por_ventana_de_gauss(0.2)

# Graficando el espectro 
ax = plt.subplot(3,2,3)
espectro_graph, = ax.plot(lambda_,senal_filtrada, linewidth=1.5, label="Señal filtrada")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$dB$", fontsize=30)
ax.set_title(label="Dominio óptico", fontsize=30)
#ax.set_ylim([-40,-10])
ax.legend(loc="lower left",fontsize=30)

opl_, amp_ = encontrar_FFT(lambda_inicial, T_muestreo_lambda, senal_filtrada)

ax = plt.subplot(3,2,4)
fft_graph, = ax.plot(opl_,amp_, linewidth=1.5,color="teal")
ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$|dB|$", fontsize=30)
ax.set_title(label="Dominio de Fourier", fontsize=30)
ax.set_xlim([lim_inf,lim_sup])
#ax.set_ylim([0,1])
#plt.savefig("FIltro.png")
#plt.show()



# Cambiando a escala lineal

senal_filtrada_esc_lineal = 10**(senal_filtrada/10)

# Construyendo una ventana gaussiana de sigma = 0.1

#w_n = ventana_de_gauss(orden=len(senal_filtrada_esc_lineal), sigma=1.4)
w_n = ventana_de_hanning(orden=len(senal_filtrada_esc_lineal))

w_n /= sum(w_n)

# Enventanado de la senal en escala lineal

senal_enventanada = senal_filtrada_esc_lineal * w_n

# Mejorando la resolucion del espectro añadiendo 0 al inicio y al final del array

n_zeros = 10000
zeros = list(np.zeros(n_zeros))

senal_enventanada = zeros + list(senal_enventanada)

senal_enventanada = np.array(senal_enventanada + zeros)



lambda_ = np.arange(1510-n_zeros*T_muestreo_lambda, 1590 + n_zeros*T_muestreo_lambda + 0.001 ,T_muestreo_lambda) 


# Calculando la FFT de la señal enventanada

opl_env, amp_env = encontrar_FFT(lambda_inicial, T_muestreo_lambda, senal_enventanada)


# Eliminando la componenete de DC por medio de eliminar los 3 primeros indices

amp_env[:10] = 0,0,0,0,0,0,0,0,0,0


# Graficando el espectro 
ax = plt.subplot(3,2,5)
espectro_graph, = ax.plot(lambda_, senal_enventanada, linewidth=1.5, label="Señal-ventana")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a.]$", fontsize=30)
ax.set_title(label="Dominio óptico escala lineal", fontsize=30)
#ax.set_ylim([-40,-10])
ax.legend(loc="upper left",fontsize=30)

# Graficando FFT
ax = plt.subplot(3,2,6)
fft_graph, = ax.plot(opl_env,amp_env, linewidth=1.5,color="navy")
ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$|u.a|$", fontsize=30)
ax.set_title(label="Dominio de Fourier", fontsize=30)
ax.set_xlim([lim_inf,lim_sup])
#ax.set_ylim([0,2])
plt.savefig("Filtro.png")
plt.show()

#print(len(np.arange(0,50,T_muestreo_lambda)))

