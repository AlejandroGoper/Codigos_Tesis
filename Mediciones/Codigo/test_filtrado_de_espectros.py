#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:28:54 2021

@author: alejandro_goper


Script para pruebas de filtrado de espectros obtenidos en el laboratorio


Documentacion para los colores de las graficas:
    - https://matplotlib.org/stable/tutorials/colors/colors.html


"""

from FabryPerot.Filtros_support import Filtro, ventana_de_gauss
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
ax.set_ylim([-40,-10])
ax.legend(loc="best",fontsize=30)

# Graficando la FFT
T_muestreo_lambda = lambda_[3] - lambda_[2] # Approx 0.005 nm
        
# Calculando la FFT
opl,amp = encontrar_FFT(lambda_inicial=lambda_[0], T_muestreo_lambda=T_muestreo_lambda, Reflectancia=potencia_dB)    
    
ax = plt.subplot(3,2,2)
fft_graph, = ax.plot(opl,amp, linewidth=1.5,color="purple")
ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$|dB|$", fontsize=30)
ax.set_title(label="Dominio de Fourier", fontsize=30)
ax.set_xlim([0,5])
ax.set_ylim([0,2])

lambda_inicial = lambda_[0]

# Al realizar el cambio de variable beta = 1/lambda, tenemos que 
T_muestreo_beta = T_muestreo_lambda / (lambda_inicial*(lambda_inicial+T_muestreo_lambda))

filtro = Filtro(_senal=potencia_dB, _T_muestreo=T_muestreo_beta*(2*10**6), _frec_corte=1.5, _orden=801)
senal_filtrada = filtro.filtrar_por_ventana_de_gauss(0.2)

# Graficando el espectro 
ax = plt.subplot(3,2,3)
espectro_graph, = ax.plot(lambda_,senal_filtrada, linewidth=1.5, label="Señal filtrada")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$dB$", fontsize=30)
ax.set_title(label="Dominio óptico", fontsize=30)
ax.set_ylim([-40,-10])
ax.legend(loc="lower left",fontsize=30)

opl_, amp_ = encontrar_FFT(lambda_inicial, T_muestreo_lambda, senal_filtrada)

ax = plt.subplot(3,2,4)
fft_graph, = ax.plot(opl_,amp_, linewidth=1.5,color="teal")
ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$|dB|$", fontsize=30)
ax.set_title(label="Dominio de Fourier", fontsize=30)
ax.set_xlim([0,5])
ax.set_ylim([0,2])
#plt.savefig("FIltro.png")
#plt.show()



# Cambiando a escala lineal

senal_filtrada_esc_lineal = 10**(senal_filtrada/10)

# Construyendo una ventana gaussiana de sigma = 0.1

w_n = ventana_de_gauss(orden=len(senal_filtrada_esc_lineal), sigma=0.06)

w_n /= sum(w_n)

# Enventanado de la senal en escala lineal

senal_enventanada = senal_filtrada_esc_lineal * w_n

# Calculando la FFT de la señal enventanada

opl_env, amp_env = encontrar_FFT(lambda_inicial, T_muestreo_lambda, senal_enventanada)


# Graficando el espectro 
ax = plt.subplot(3,2,5)
espectro_graph, = ax.plot(lambda_,senal_enventanada, linewidth=1.5, label="Señal-ventana")
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
ax.set_xlim([0,5])
#ax.set_ylim([0,2])
plt.savefig("FIltro.png")

#plt.show()

"""
# Construimos una ventana gaussiana de la misma longitud que la senal 

obj = Filtro(_senal=None, _T_muestreo=1, _frec_corte=1, _orden=len(potencia_dB))
w_n = obj.ventana_de_gauss(0.1)

w_n /= sum(w_n)

#w_n = 10*np.log10(w_n)

plt.figure()
plt.plot(w_n)
plt.show()

potencia_escala_lineal = (10**potencia_dB)/10
senal_truncada = potencia_escala_lineal * w_n

#senal_truncada = potencia_dB+w_n

#senal_truncada = 10*np.log10(senal_truncada)

plt.figure()
plt.plot(lambda_, senal_truncada)
plt.show()



opl_truncada, amp_truncada = encontrar_FFT(lambda_inicial, T_muestreo_beta*(2*10**6), senal_truncada)

plt.figure()
plt.plot(opl_truncada, amp_truncada)
plt.xlabel("OPL [mm]")
plt.xlim([0,5])
plt.ylabel("u.a")
plt.show()


filtro_2 = Filtro(_senal=senal_truncada, _T_muestreo=T_muestreo_beta*(2*10**6), _frec_corte=1.5, _orden=851)
senal_filtrada_2 = filtro_2.filtrar_por_ventana_de_gauss(0.2)

opl_filtrada_2, amp_filtrada_2 = encontrar_FFT(lambda_inicial, T_muestreo_lambda, senal_filtrada_2)

plt.figure()
plt.plot(opl_filtrada_2,amp_filtrada_2)
plt.xlabel("OPL [mm]")
plt.xlim([0,5])
plt.show()

"""