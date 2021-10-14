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

# Importando archivos 

ruta_directorio = "../1GAP-AIRE-8-10-2021"

contenido = os.listdir(ruta_directorio)

data_spectra = contenido

for spectrum in data_spectra[0:1]:
    path = ruta_directorio + "/" + spectrum
    data = np.loadtxt(path,skiprows=58)

plt.figure()
plt.plot(data[:,0], data[:,1],linewidth = 0.5)
plt.xlabel(xlabel=r"$\lambda [nm]$")
plt.ylabel(ylabel=r"$Pot [dBm]$")
#plt.xlim([1510,1520])
plt.show()