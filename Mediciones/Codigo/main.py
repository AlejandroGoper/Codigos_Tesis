#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:20:42 2021

@author: alejandro_goper

Este script es para analizar las mediciones obtenidas en el laboratorio
adquiridas con el Interrogador.

Se agregan funcionalidades de filtrado y windowing, además de la normalizacion
con respecto a una señal de referencia (para aplanar la señal en el dominio -
                                        optico)

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import writers, FuncAnimation
import os
from FabryPerot.FFT_support import encontrar_FFT
from FabryPerot.Filtros_support import Filtro, ventana_de_gauss

# Creando figura
fig,ax = plt.subplots(figsize=(40,20))
# Pone lo mas juntas las graficas posibles
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
# Cambiando el tamano de la fuente en todos los ejes
plt.rcParams.update({'font.size': 20})


# Importando archivos 
fecha_medicion = "18-10-2021"
carpeta = "1GAP-CAPILAR-AIRE-10um"
# Incremento en la medicion
inc = 10 # um
ruta_directorio = "../" + fecha_medicion + "/" + carpeta
# Calculando el numero de archivos en la carpeta
n = len(os.listdir(ruta_directorio))
nombre_archivo = "Espectro (1).txt"
path = ruta_directorio + "/" + nombre_archivo

# Cargando el primer espectro de las mediciones
data = np.loadtxt(path, skiprows=58)

# Separando datos por columnas    
lambda_ = data[:,0]
potencia_dBm = data[:,1]

path_ref = "../Referencia/referencia.txt"
# Cargando el espectro de la referencia
data_ref = np.loadtxt(path_ref, skiprows=0)

# Separando datos por columnas    
potencia_dBm_ref = data_ref[:,1]


# Normalizando el espectro medido respecto a la referecia
# Debido a que estan en escala logaritmica, la division es una resta

potencia_dB = potencia_dBm - potencia_dBm_ref

# T_muestreo_lambda = lambda_[3] - lambda_[2] Approx 0.005 nm
T_muestreo_lambda = 0.005

lambda_inicial = lambda_[0]

# Al realizar el cambio de variable beta = 1/lambda, tenemos que 
T_muestreo_beta = T_muestreo_lambda / (lambda_inicial*(lambda_inicial+T_muestreo_lambda))

# Aplicando Filtro pasabajos en una frecuencia de corte proporcional al incremento en las mediciones
# medido en milimetros
f_c = 20*(inc*0.001)

filtro = Filtro(_senal=potencia_dB, _T_muestreo=T_muestreo_beta*(2*10**6), _frec_corte=f_c, _orden=901)

potencia_dB_filtrada = filtro.filtrar_por_ventana_de_gauss(0.1)

potencia_filtrada_lineal = 10**(potencia_dB_filtrada/10)

w_n = ventana_de_gauss(orden=len(potencia_filtrada_lineal), sigma=0.3)
w_n /= sum(w_n)

potencia_filtrada_limitada_lineal = w_n*potencia_filtrada_lineal


# Calculando la FFT
opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_filtrada_limitada_lineal)    
    
# Eliminando la componente de DC

for i in np.arange(3):
    amp[i] = 0

# Graficando el espectro 
    
ax = plt.subplot(1,2,1)
espectro_graph, = ax.plot(lambda_,potencia_filtrada_limitada_lineal, linewidth=1.5, label="Medicion filtrada y limitada")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=26)
ax.set_ylabel(ylabel=r"$u.a.$", fontsize=26)
ax.set_title(label="Dominio óptico - Escala Lineal", fontsize=30)
ax.grid()
ax.legend(loc="best", fontsize=26)
#ax.set_ylim([-40,-10])

# Graficando la FFT

ax = plt.subplot(1,2,2)
fft_graph, = ax.plot(opl,amp, linewidth=1.9,color="purple")
ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=26)
ax.set_ylabel(ylabel=r"$|dB|$", fontsize=26)
ax.set_title(label="Dominio de Fourier", fontsize=30)
ax.grid()
ax.set_xlim([0,1.2])
#ax.set_ylim([0,1])



# Frames = numero de Espectros
def actualizar(i):
    
    # fecha_medicion = "18-10-2021"
    # carpeta = "1GAP-CAPILAR-AIRE-10um"
    # ruta_directorio = "../" + fecha_medicion + "/" + carpeta
    
    numero_simulacion = str(i)
    # numero_simulacion = format(i,"0>2d") 
    
    nombre_archivo = "Espectro (" + numero_simulacion + ").txt"
    
    path = ruta_directorio + "/" + nombre_archivo
    
    data = np.loadtxt(path, skiprows=58)
    
    # Separando datos por columnas
    potencia_dBm = data[:,1]
    
    # Normalizando el espectro medido respecto a la referecia
    # Debido a que estan en escala logaritmica, la division es una resta

    potencia_dB = potencia_dBm - potencia_dBm_ref
    
    # Aplicando Filtro pasabajos en una frecuencia de corte proporcional al incremento en las mediciones
    # medido en milimetros
    f_c = i*5*(inc*0.001)
    
    filtro = Filtro(_senal=potencia_dB, _T_muestreo=T_muestreo_beta*(2*10**6), _frec_corte=f_c, _orden=901)

    potencia_dB_filtrada = filtro.filtrar_por_ventana_de_gauss(0.1)
    # Transformando a escala lineal 
    potencia_filtrada_lineal = 10**(potencia_dB_filtrada/10)
    # Definiendo ventana gaussiana
    w_n = ventana_de_gauss(orden=len(potencia_filtrada_lineal), sigma=0.3)
    w_n /= sum(w_n)
    # Limitando el ancho de banda espectral
    potencia_filtrada_limitada_lineal = w_n*potencia_filtrada_lineal
    
    # Calculando la FFT
    opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_filtrada_limitada_lineal)    
    
    # Eliminando la componente de DC de la amplitud de la fft    
    for i in np.arange(3):
        amp[i] = 0

    
    espectro_graph.set_ydata(potencia_filtrada_limitada_lineal)
    fft_graph.set_ydata(amp)
    fig.suptitle(nombre_archivo)
    
    
    
    return espectro_graph, fft_graph, ax


anim = FuncAnimation(fig = fig, func=actualizar, repeat= True, frames = np.arange(1,n+1), interval=1500)

Writer = writers["ffmpeg"]
writer = Writer(fps=3,metadata={"artist":"IAGP"},bitrate=1800)
anim.save(carpeta+".mp4",writer)
