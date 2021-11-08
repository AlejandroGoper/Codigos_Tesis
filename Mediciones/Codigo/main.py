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
from FabryPerot.Filtros_support import Filtro, ventana_de_gauss, ventana_de_hanning

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

# Creando ventana
w_n = ventana_de_gauss(orden=len(potencia_filtrada_lineal), sigma=0.45)
#w_n = ventana_de_hanning(orden=len(potencia_filtrada_lineal))
w_n /= sum(w_n)

potencia_filtrada_limitada_lineal = w_n*potencia_filtrada_lineal


# Calculando la FFT
opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_filtrada_limitada_lineal)    
    
# Eliminando la componente de DC

for i in np.arange(3):
    amp[i] = 0


# Resolucion en el dominio de la frecuencia

dw = opl[2] - opl[1]

lim_inf = dw * 0
lim_sup = dw * 100


# Limitando la busqeda a la ventana de interes deinida por [lim_inf, lim_sup]

index_lim_inf = int(np.where(opl == lim_inf)[0])
index_lim_sup = int(np.where(opl == lim_sup)[0])

# Buscaremos el maximo solo en la vetana de interes.

amp_temp = amp[index_lim_inf:index_lim_sup] 

# Dado que ya hay un solo maximo podemos encontrarlo facilmente
max_value = amp_temp.max()
index_max_value = int(np.where(amp_temp == max_value)[0])
OPL_value = round(opl[index_max_value+index_lim_inf],3)

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
ax.set_ylabel(ylabel=r"$|a.u.|$", fontsize=26)
ax.set_title(label="Dominio de Fourier", fontsize=30)
ax.grid()
ax.set_xlim([lim_inf,lim_sup])
#ax.set_ylim([0,1e-7])

# Creando cadena para caja de texto
textstr = r"$OPL_{max}$ = " + str(OPL_value) 
#textstr = ''.join((r'$OPL=%.3f$' % (OPL_value, )))

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='teal', alpha=0.5)

graph_text = ax.text(.7, 0.95, textstr, transform=ax.transAxes, fontsize=40,
        verticalalignment='top', bbox=props)

#ax.set_ylim([0,1])


# Frames = numero de Espectros
def actualizar(i):
    
    # Cargando datos de las mediciones
    
    numero_simulacion = str(i)
    
    nombre_archivo = "Espectro (" + numero_simulacion + ").txt"
    
    path = ruta_directorio + "/" + nombre_archivo
    
    data = np.loadtxt(path, skiprows=58)
    
    # Separando datos por columnas
    potencia_dBm = data[:,1]
    
    # Normalizando el espectro medido respecto a la referecia
    potencia_dB = potencia_dBm - potencia_dBm_ref
    
    # Aplicando Filtro pasabajos en una frecuencia de corte proporcional al incremento en las mediciones
    # medido en milimetros
    f_c = (20+i)*(inc*0.001)
    filtro = Filtro(_senal=potencia_dB, _T_muestreo=T_muestreo_beta*(2*10**6), _frec_corte=f_c, _orden=901)
    potencia_dB_filtrada = filtro.filtrar_por_ventana_de_gauss(0.1)
    #potencia_dB_filtrada = filtro.filtrar_por_ventana_de_hanning()
    # Transformando a escala lineal 
    potencia_filtrada_lineal = 10**(potencia_dB_filtrada/10)
    
    # Limitando el ancho de banda espectral
    potencia_filtrada_limitada_lineal = w_n*potencia_filtrada_lineal
    
    # Calculando la FFT
    opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_filtrada_limitada_lineal)    
    
    
    # Eliminando la componente de DC de la amplitud de la fft    
    for i in np.arange(3):
        amp[i] = 0
    
    # Buscaremos el maximo solo en la vetana de interes.

    amp_temp = amp[index_lim_inf:index_lim_sup] 
    
    # Dado que ya hay un solo maximo en esa region podemos encontrarlo facilmente
    max_value = amp_temp.max()
    index_max_value = int(np.where(amp_temp==max_value)[0])
    # Redondeamos a 3 c.s.
    OPL_value = round(opl[index_max_value+index_lim_inf],4)
    
    espectro_graph.set_ydata(potencia_filtrada_limitada_lineal)
    fft_graph.set_ydata(amp)
    # Creando cadena para caja de texto
    textstr = r"$OPL_{max}$ = " + str(OPL_value) 
    graph_text.set_text(textstr)
    fig.suptitle(nombre_archivo, fontsize=18)
    
    return espectro_graph, fft_graph,  graph_text, ax


anim = FuncAnimation(fig = fig, func=actualizar, repeat= True, frames = np.arange(1,n+1), interval=1500)

Writer = writers["ffmpeg"]
writer = Writer(fps=3,metadata={"artist":"IAGP"},bitrate=1800)
anim.save(carpeta+".mp4",writer)
