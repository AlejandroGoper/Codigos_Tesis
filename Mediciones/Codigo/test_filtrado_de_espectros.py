#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:28:54 2021

@author: alejandro_goper


Script para pruebas de filtrado de espectros obtenidos en el laboratorio

"""

from FabryPerot.Filtros_support import Filtro
from FabryPerot.FFT_support import encontrar_FFT, calcular_verdadera_amplitud, recorte_frec_negativas_fft
import numpy as np
import matplotlib.pyplot as plt
 
# Importando un espectro del fabry perot: 1GAP-CAPILAR-AIRE-10um - ESPECTRO (105)

# Importando archivos 

fecha_medicion = "18-10-2021"

carpeta = "1GAP-CAPILAR-AIRE-10um"

ruta_directorio = "../" + fecha_medicion + "/" + carpeta

nombre_archivo = "Espectro (105).txt"

path = ruta_directorio + "/" + nombre_archivo

data = np.loadtxt(path, skiprows=58)

path = "../Referencia/referencia.txt"

referencia = np.loadtxt(path)

# Seprando datos de referencia por columnas

lambda_ref, potencia_dBm_ref = referencia[:,0], referencia[:,1]

# Separando datos por columnas
lambda_ = data[:,0]
potencia_dBm = data[:,1]


# Normalizando respecto a la referencia

potencia_dB = potencia_dBm - potencia_dBm_ref

# Creando figura
fig,ax = plt.subplots()
fig.set_tight_layout(True)
# Pone lo mas juntas las graficas posibles
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)

# Graficando el espectro 
ax = plt.subplot(2,2,1)
espectro_graph, = ax.plot(lambda_,potencia_dB, linewidth=0.6)
ax.set_xlabel(xlabel=r"$\lambda [nm]$")
ax.set_ylabel(ylabel=r"$dB$")
ax.set_title(label="Dominio óptico")
ax.set_ylim([-40,-10])

# Graficando la FFT
T_muestreo_lambda = lambda_[3] - lambda_[2] # Approx 0.005 nm
        
# Calculando la FFT
opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_dB)    
    
ax = plt.subplot(2,2,2)
fft_graph, = ax.plot(opl,amp, linewidth=0.9,color="purple")
ax.set_xlabel(xlabel=r"$OPL [mm]$")
ax.set_ylabel(ylabel=r"$|dB|$")
ax.set_title(label="Dominio de Fourier")
ax.set_xlim([0,5])
ax.set_ylim([0,2])

lambda_inicial = lambda_[0]

# Al realizar el cambio de variable beta = 1/lambda, tenemos que 
T_muestreo_beta = T_muestreo_lambda / (lambda_inicial*(lambda_inicial+T_muestreo_lambda))

filtro = Filtro(_senal=potencia_dB, _T_muestreo=T_muestreo_beta*(2*10**6), _frec_corte=1.5, _orden=801)
senal_filtrada = filtro.filtrar_por_ventana_de_gauss(0.2)

# Graficando el espectro 
ax = plt.subplot(2,2,3)
espectro_graph, = ax.plot(lambda_,senal_filtrada, linewidth=0.6)
ax.set_xlabel(xlabel=r"$\lambda [nm]$")
ax.set_ylabel(ylabel=r"$dB$")
ax.set_title(label="Dominio óptico")
ax.set_ylim([-40,-10])

opl_, amp_ = encontrar_FFT(lambda_inicial, T_muestreo_lambda, senal_filtrada)

ax = plt.subplot(2,2,4)
fft_graph, = ax.plot(opl_,amp_, linewidth=0.9,color="teal")
ax.set_xlabel(xlabel=r"$OPL [mm]$")
ax.set_ylabel(ylabel=r"$|dB|$")
ax.set_title(label="Dominio de Fourier")
ax.set_xlim([0,5])
ax.set_ylim([0,2])
