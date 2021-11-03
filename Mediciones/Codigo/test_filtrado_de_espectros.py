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

# Separando datos por columnas
lambda_ = data[:,0]
potencia_dBm = data[:,1]


# Creando figura
fig,ax = plt.subplots()
fig.set_tight_layout(True)

# Pone lo mas juntas las graficas posibles
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)

# Graficando el espectro 
ax = plt.subplot(3,2,1)
espectro_graph, = ax.plot(lambda_,potencia_dBm, linewidth=0.6)
ax.set_xlabel(xlabel=r"$\lambda [nm]$")
ax.set_ylabel(ylabel=r"Pot $[dBm]$")
ax.set_title(label="Dominio óptico")
ax.set_ylim([-40,-10])

# Graficando la FFT
T_muestreo_lambda = lambda_[3] - lambda_[2] # Approx 0.005 nm
        
# Calculando la FFT
opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_dBm)    
    
ax = plt.subplot(3,2,2)
fft_graph, = ax.plot(opl,amp, linewidth=0.9,color="purple")
ax.set_xlabel(xlabel=r"$OPL [mm]$")
ax.set_ylabel(ylabel=r"$Amp$")
ax.set_title(label="Dominio de Fourier")
ax.set_xlim([0,5])
ax.set_ylim([0,2])

lambda_inicial = lambda_[0]

# Al realizar el cambio de variable beta = 1/lambda, tenemos que 
T_muestreo_beta = T_muestreo_lambda / (lambda_inicial*(lambda_inicial+T_muestreo_lambda))

beta = 1/lambda_
beta_ = np.flip(beta) # [mm]

# Graficando el espectro 
ax = plt.subplot(3,2,3)
espectro_graph, = ax.plot(beta_,potencia_dBm, linewidth=0.6)
ax.set_xlabel(xlabel=r"$\beta [\frac{1}{nm}]$")
ax.set_ylabel(ylabel=r"Pot $[dBm]$")
ax.set_title(label="Dominio óptico")
ax.set_ylim([-40,-10])

fft_test = np.fft.fft(potencia_dBm)
vfreq = np.fft.fftfreq(len(potencia_dBm),T_muestreo_beta)

fft_amp = calcular_verdadera_amplitud(fft_test)
vfreq = recorte_frec_negativas_fft(vfreq)

ax = plt.subplot(3,2,4)
fft_graph, = ax.plot(vfreq,fft_amp, linewidth=0.9,color="purple")
ax.set_xlabel(xlabel=r"$vfreq [nm]$")
ax.set_ylabel(ylabel=r"$Amp |dBm|$")
ax.set_title(label="Dominio de Fourier")
ax.set_xlim([0,8*10**6])
ax.set_ylim([0,2])


filtro = Filtro(_senal=potencia_dBm, _T_muestreo=T_muestreo_beta*(2*10**6), _frec_corte=1.5, _orden=991)
senal_filtrada = filtro.filtrar_por_ventana_de_gauss(0.2)

# Graficando el espectro 
ax = plt.subplot(3,2,5)
espectro_graph, = ax.plot(beta_,senal_filtrada, linewidth=0.6)
ax.set_xlabel(xlabel=r"$\beta [\frac{1}{nm}]$")
ax.set_ylabel(ylabel=r"Pot $[dBm]$")
ax.set_title(label="Dominio óptico")
ax.set_ylim([-40,-10])


fft_senal_fitrada = np.fft.fft(senal_filtrada)
vfreq_filt = np.fft.fftfreq(len(senal_filtrada),T_muestreo_beta*(2*10**6))

fft_amp_filt = calcular_verdadera_amplitud(fft_senal_fitrada)
vfreq_filt = recorte_frec_negativas_fft(vfreq_filt)

ax = plt.subplot(3,2,6)
fft_graph, = ax.plot(vfreq_filt,fft_amp_filt, linewidth=0.9,color="black")
ax.set_xlabel(xlabel=r"$vfreq [mm]$")
ax.set_ylabel(ylabel=r"$Amp |dBm|$")
ax.set_title(label="Dominio de Fourier")
ax.set_xlim([0,5])
ax.set_ylim([0,2])
