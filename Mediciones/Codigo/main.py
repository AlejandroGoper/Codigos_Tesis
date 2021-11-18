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
from FabryPerot.FFT_support import encontrar_FFT_dominio_en_OPL
from FabryPerot.Filtros_support import Filtro, ventana_de_gauss, ventana_de_hanning, ventana_flattop
from sklearn.neighbors import NearestNeighbors
from scipy.signal import find_peaks

# Creando figura
fig,ax = plt.subplots(figsize=(40,20))
# Pone lo mas juntas las graficas posibles
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
# Cambiando el tamano de la fuente en todos los ejes
plt.rcParams.update({'font.size': 20})


# Importando archivos 
fecha_medicion = "17-11-2021"
carpeta = "2GAP-CAPILAR-AIRE-AGUA-100um"
# Incremento en la medicion
inc = 100 # um

ruta_directorio = "../" + fecha_medicion + "/" + carpeta
# Calculando el numero de archivos en la carpeta
n = len(os.listdir(ruta_directorio)) - 1
nombre_archivo = "Espectro (1).txt"
path = ruta_directorio + "/" + nombre_archivo

# Cargando el primer espectro de las mediciones
data = np.loadtxt(path, skiprows=58)

# Separando datos por columnas    
lambda_ = data[:,0]
potencia_dBm = data[:,1]

path_ref = ruta_directorio + "/referencia.txt"
# Cargando el espectro de la referencia
data_ref = np.loadtxt(path_ref, skiprows=58)

# Separando datos por columnas    
potencia_dBm_ref = data_ref[:,1]


# Normalizando el espectro medido respecto a la referecia
# Debido a que estan en escala logaritmica, la division es una resta

potencia_dB = potencia_dBm - potencia_dBm_ref

# T_muestreo_lambda = (lambda_[-1] - lambda_[0])/len(lambda_) Approx 0.005 nm

lambda_inicial = lambda_[0]
lambda_final = lambda_[-1]
m = len(lambda_)
T_muestreo_lambda = (lambda_final-lambda_inicial)/m

# Limite del dominio OPL en Fourier

lim_inf = 0.0 # mm
lim_sup = 5.0 # mm

# Al realizar el cambio de variable beta = 1/lambda, tenemos que 
T_muestreo_beta = (1/lambda_inicial - 1/lambda_final)/m
T_muestreo_beta_opl = T_muestreo_beta*(2*10**6)
# Aplicando Filtro pasabajos en una frecuencia de corte proporcional al incremento en las mediciones
# medido en milimetros
#f_c = 20*(inc*0.001)
f_c = lim_sup
filtro = Filtro(_senal=potencia_dB, 
                _T_muestreo=T_muestreo_beta_opl, 
                _frec_corte=f_c, 
                _orden=901)

potencia_dB_filtrada = filtro.filtrar_por_ventana_de_gauss(0.1)

potencia_filtrada_lineal = 10**(potencia_dB_filtrada/10)

# Creando ventana
# w_n = ventana_de_gauss(orden=len(potencia_filtrada_lineal), sigma=0.45)
w_n = ventana_de_hanning(orden=len(potencia_filtrada_lineal))
# w_n = ventana_flattop(orden = len(potencia_filtrada_lineal))
# w_n = np.ones(len(potencia_dB_filtrada))
potencia_filtrada_limitada_lineal = w_n*potencia_filtrada_lineal


# Calculando la FFT
opl,amp = encontrar_FFT_dominio_en_OPL(lambda_inicial= lambda_inicial,
                                       lambda_final= lambda_final, 
                                       senal=potencia_filtrada_limitada_lineal)



"""
==============================================================================
Estableciendo rango de busqueda en el espacio de Fourier  
==============================================================================
"""

nn = NearestNeighbors(n_neighbors=1)

nn.fit(opl.reshape((len(opl),1)))

index_lim_inf = nn.kneighbors([[lim_inf]], 1, return_distance=False)[0,0]
index_lim_sup = nn.kneighbors([[lim_sup]], 1, return_distance=False)[0,0]

lim_inf_opl = opl[index_lim_inf]
lim_sup_opl = opl[index_lim_sup]


# Eliminando la componenete de DC hasta un margen fijo en el opl

dc_margen = 0.08  # mm

nn = NearestNeighbors(n_neighbors=1)

# buscamos el indice en el array opl mas cercano a dc_margen
nn.fit(opl.reshape((len(opl),1)))
index_dc_margen = nn.kneighbors([[dc_margen]], 1, return_distance=False)[0,0]
# eliminamos todas las contribuciones del espectro de fourier hasta dc_margen
amp[:index_dc_margen] = np.zeros(index_dc_margen)


"""
==============================================================================
Mejoramiento de la resolucion en Fourier post-windowing
==============================================================================
"""

# Numero de ceros a agregar en cada extremo
n_zeros = 1000
zeros = list(np.zeros(n_zeros))

# Agregamos los ceros a cada extremo de la señal
potencia_filtrada_limitada_lineal = zeros + list(potencia_filtrada_limitada_lineal)
potencia_filtrada_limitada_lineal = np.array(potencia_filtrada_limitada_lineal + zeros)

# Agregando las correspondientes longitudes de onda (virtuales) 
# correspondientes a los ceros añadidos a los extremos 
lambda_mejorada = np.arange(lambda_inicial-n_zeros*T_muestreo_lambda, 
                    lambda_final + n_zeros*T_muestreo_lambda-0.00001,
                    T_muestreo_lambda) 



# La funcion de find_peaks de Scipy ya tiene la capacidad de realizar esto

# Encontrando el vecino mas cercano a los limites de busqueda en opl_env

index_lim_inf = nn.kneighbors([[lim_inf]], 1, return_distance=False)[0,0]
index_lim_sup = nn.kneighbors([[lim_sup]], 1, return_distance=False)[0,0]

# Limitamos la busqueda
amp_temp = amp[index_lim_inf:index_lim_sup]

# Necesitamos definir un valor limite en altura en el grafico de la amplitud
# se buscaran los maximos que superen este valor
lim_amp = 0.00025

# Buscando maximos en la region limitada
picos, _ = find_peaks(amp_temp, height = lim_amp)

# Como limitamos la busqueda hay que compensar con los indices anteriores 
# en el array para obtener el valor verdadero 
picos += index_lim_inf 

maximos = opl[picos]

# Imprimiendo resultados
text = ""
for maximo in maximos: 
    text += "\nMaximo localizado en: %.3f mm" % maximo
# Eliminando el espacio en blanco inicial
text = text[1:]
# Graficando el espectro 
    
ax1 = plt.subplot(1,2,1)
espectro_graph, = ax1.plot(lambda_mejorada,
                          potencia_filtrada_limitada_lineal, 
                          linewidth=1.5, 
                          label="Medicion filtrada y limitada")
ax1.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=26)
ax1.set_ylabel(ylabel=r"$u.a.$", fontsize=26)
ax1.set_title(label="Dominio óptico - Escala Lineal", fontsize=30)
#ax1.grid()
ax1.legend(loc="best", fontsize=26)
#ax.set_ylim([-40,-10])

# Graficando la FFT

ax2 = plt.subplot(1,2,2)
fft_graph, = ax2.plot(opl,amp, linewidth=1.9,color="purple")
ax2.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=26)
ax2.set_ylabel(ylabel=r"$|a.u.|$", fontsize=26)
ax2.set_title(label="Dominio de Fourier", fontsize=30)
#ax2.grid()
ax2.set_xlim([lim_inf,lim_sup])
#ax.set_ylim([0,1e-7])

# Creando cadena para caja de texto
textstr = text 
#textstr = ''.join((r'$OPL=%.3f$' % (OPL_value, )))

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='teal', alpha=0.5)

graph_text = ax2.text(0.79, 1.05, textstr, transform=ax.transAxes, fontsize=35,
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
    filtro = Filtro(_senal=potencia_dB, 
                    _T_muestreo=T_muestreo_beta_opl, 
                    _frec_corte=f_c, 
                    _orden=901)
    #potencia_dB_filtrada = filtro.filtrar_por_ventana_de_gauss(0.1)
    potencia_dB_filtrada = filtro.filtrar_por_ventana_de_hanning()
    # Transformando a escala lineal 
    potencia_filtrada_lineal = 10**(potencia_dB_filtrada/10)
    
    # Limitando el ancho de banda espectral
    potencia_filtrada_limitada_lineal = w_n*potencia_filtrada_lineal
    
    # Calculando la FFT
    opl,amp = encontrar_FFT_dominio_en_OPL(lambda_inicial=lambda_inicial, 
                                           lambda_final=lambda_final, 
                                           senal=potencia_filtrada_limitada_lineal)
    
    
    zeros = list(np.zeros(n_zeros))

    # Agregamos los ceros a cada extremo de la señal
    potencia_filtrada_limitada_lineal = zeros + list(potencia_filtrada_limitada_lineal)
    potencia_filtrada_limitada_lineal = np.array(potencia_filtrada_limitada_lineal + zeros)

    
    
    # eliminamos todas las contribuciones del espectro de fourier hasta dc_margen
    amp[:index_dc_margen] = np.zeros(index_dc_margen)
    # Limitamos la busqueda
    amp_temp = amp[index_lim_inf:index_lim_sup]
    
    # Necesitamos definir un valor limite en altura en el grafico de la amplitud
    # se buscaran los maximos que superen este valor
    lim_amp = 0.00025
    
    # Buscando maximos en la region limitada
    picos, _ = find_peaks(amp_temp, height = lim_amp)
    
    # Como limitamos la busqueda hay que compensar con los indices anteriores 
    # en el array para obtener el valor verdadero 
    picos += index_lim_inf 
    
    maximos = opl[picos]
    
    # Imprimiendo resultados
    text = ""
    for maximo in maximos: 
        text += "\nMaximo localizado en: %.3f mm" % maximo
    
    espectro_graph.set_ydata(potencia_filtrada_limitada_lineal)
    ax1.set_ylim([0,max(potencia_filtrada_limitada_lineal)])
    fft_graph.set_ydata(amp)
    ax2.set_ylim([0,max(amp)])
    # Creando cadena para caja de texto
    textstr = text[1:] 
    graph_text.set_text(textstr)
    fig.suptitle(nombre_archivo, fontsize=18)
    return espectro_graph, fft_graph, graph_text, ax1, ax2


anim = FuncAnimation(fig = fig, func=actualizar, 
                     repeat= True, 
                     frames = np.arange(1,n+1), 
                     interval=1500)

Writer = writers["ffmpeg"]
writer = Writer(fps=3,metadata={"artist":"IAGP"},bitrate=1800)
anim.save(carpeta+".mp4",writer)
