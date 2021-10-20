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
from matplotlib.animation import writers, FuncAnimation
import os
from FabryPerot.FFT_support import encontrar_FFT

# Creando figura

fig,ax = plt.subplots()

fig.set_tight_layout(True)

# Pone lo mas juntas las graficas posibles
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)


# Importando archivos 

fecha_medicion = "18-10-2021"

carpeta = "2GAP-CAPILAR-AIRE-AGUA-100um"

ruta_directorio = "../" + fecha_medicion + "/" + carpeta

n = len(os.listdir(ruta_directorio))

nombre_archivo = "Espectro (1).txt"

path = ruta_directorio + "/" + nombre_archivo

data = np.loadtxt(path, skiprows=58)

 # Separando datos por columnas
    
lambda_ = data[:,0]
potencia_dBm = data[:,1]

# Graficando el espectro 
    
ax = plt.subplot(1,2,1)
espectro_graph, = ax.plot(lambda_,potencia_dBm, linewidth=0.6)
ax.set_xlabel(xlabel=r"$\lambda [nm]$")
ax.set_ylabel(ylabel=r"Pot $[dBm]$")
ax.set_title(label="Dominio óptico")
ax.set_ylim([-40,-10])

 # Graficando la FFT
 
 
T_muestreo_lambda = lambda_[3] - lambda_[2] # Approx 0.005 nm
        
# Calculando la FFT
opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_dBm)    
    
ax = plt.subplot(1,2,2)
fft_graph, = ax.plot(opl,amp, linewidth=0.9,color="purple")
ax.set_xlabel(xlabel=r"$OPL [mm]$")
ax.set_ylabel(ylabel=r"$Amp$")
ax.set_title(label="Dominio de Fourier")
ax.set_xlim([0,10])
ax.set_ylim([0,1])



# Frames = numero de Espectros
def actualizar(i):
    
    fecha_medicion = "18-10-2021"
    carpeta = "2GAP-CAPILAR-AIRE-AGUA-100um"
    ruta_directorio = "../" + fecha_medicion + "/" + carpeta
    
    numero_simulacion = str(i)
    # numero_simulacion = format(i,"0>2d") 
    
    nombre_archivo = "Espectro (" + numero_simulacion + ").txt"
    
    path = ruta_directorio + "/" + nombre_archivo
    
    data = np.loadtxt(path, skiprows=58)
    
    # Separando datos por columnas
    
    lambda_ = data[:,0]
    
    T_muestreo_lambda = lambda_[3] - lambda_[2] # Approx 0.005 nm
    
    potencia_dBm = data[:,1]
    
    # Calculando la FFT
    opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_dBm)    
    
    espectro_graph.set_ydata(potencia_dBm)
    fft_graph.set_ydata(amp)
    fig.suptitle(nombre_archivo)
    
    
    
    return espectro_graph, fft_graph, ax


anim = FuncAnimation(fig = fig, func=actualizar, repeat= True, frames = np.arange(1,n+1), interval=1500)

Writer = writers["ffmpeg"]
writer = Writer(fps=3,metadata={"artist":"IAGP"},bitrate=1800)
anim.save(carpeta+".mp4",writer)

"""

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
    ax.set_ylabel(ylabel=r"Pot $[dBm]$")
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
    
"""