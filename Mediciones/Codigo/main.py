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


data_spectra = contenido_ordenado

for spectrum in data_spectra:
    path = ruta_directorio + "/" + spectrum
    data = np.loadtxt(path,skiprows=58)
    
    
    # Separando datos
    
    lambda_ = data[:,0]
    potencia_dBm = data[:,1]
    
    T_muestreo_lambda = lambda_[3] - lambda_[2] # Approx 0.005 nm
        
    
    opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_dBm)
    
    
    # Graficando
    
    plt.figure()
    plt.plot(lambda_,potencia_dBm, linewidth=0.6)
    #plt.plot(opl, amp)
    plt.xlabel(xlabel=r"$OPL [mm]$")
    plt.ylabel(ylabel=r"$Pot [dBm]$")
    plt.title(label=spectrum)
    #plt.xlim([0,5])
    plt.show()