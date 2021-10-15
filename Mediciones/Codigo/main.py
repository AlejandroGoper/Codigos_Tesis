#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:20:42 2021

@author: alejandro_goper

Este script es para analizar las mediciones obtenidas en el laboratorio
adquiridas con el Interrogador.

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from FabryPerot.FFT_support import encontrar_FFT

# Importando archivos 

ruta_directorio = "../1GAP-AIRE-8-10-2021"

contenido = os.listdir(ruta_directorio)


# Ordenando el array 
contenido_ordenado = sorted(contenido)


# Recorriendo todos los espectros

data_spectra = contenido_ordenado


for spectrum in data_spectra[:]:
    path = ruta_directorio + "/" + spectrum
    data = np.loadtxt(path,skiprows=58)
   
    # Separando datos por columnas
    
    lambda_ = data[:,0]
    potencia_dBm = data[:,1]
    
    T_muestreo_lambda = lambda_[3] - lambda_[2] # Approx 0.005 nm
        
    # Calculando la FFT
    opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_dBm)
    
    
    # Graficando el espectro
    
    fig, ax = plt.subplots()
    # Pone lo mas juntas las graficas posibles
    fig.set_tight_layout(True)
    # Para que no se empalmen los titulos en los ejes
    fig.subplots_adjust(wspace=1.2)
    # Poniendo titulo
    fig.suptitle("Archivo: "+ spectrum)
    
    # Graficando el espectro 
    
    ax = plt.subplot(1,2,1)
    ax.plot(lambda_,potencia_dBm, linewidth=0.6)
    ax.set_xlabel(xlabel=r"$\lambda [nm]$")
    ax.set_ylabel(ylabel=r"$Pot [dBm]$")
    ax.set_title(label="Dominio óptico")
    
    # Graficando la FFT
    
    ax = plt.subplot(1,2,2)
    ax.plot(opl,amp, linewidth=0.9,color="purple")
    ax.set_xlabel(xlabel=r"$OPL [mm]$")
    ax.set_ylabel(ylabel=r"$Amp$")
    ax.set_title(label="Dominio de Fourier")
    ax.set_xlim([0,3])
    ax.set_ylim([0,3])
    
    plt.show()