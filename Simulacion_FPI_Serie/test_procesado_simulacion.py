#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:59:45 2021

@author: alejandro_goper

Este script es para realizar la prueba de resultados de la simulación de un
interferómetro Fabry-Perot de 2 cavidades con el procesamiento automatico de 
la señal para así evaluar la precisión de script

"""


import numpy as np
import matplotlib.pyplot as plt
from FabryPerot.Clase import FabryPerot_2GAP
from FabryPerot.FFT_support import encontrar_FFT_dominio_en_OPL
from FabryPerot.Filtros_support import Filtro, ventana_de_hanning, ventana_de_gauss, ventana_flattop, ventana_kaiser_bessel
from scipy.signal import find_peaks
from sklearn.neighbors import NearestNeighbors


"""
==============================================================================
Cargando simulación
==============================================================================
"""

lambda_inicial = 1510 #nm

lambda_final = 1590 #nm

T_muestreo_lambda = 0.005 #nm

# Definicion del dominio en longitudes de onda
lambda_ = np.arange(lambda_inicial,lambda_final+T_muestreo_lambda, T_muestreo_lambda) #nanometros


# Construyendo señal a analizar en el dominio de fourier
obj = FabryPerot_2GAP(lambda_inicial=lambda_inicial,
                      lambda_final= lambda_final,
                      T_muestreo_lambda= 0.005,
                      L_medio_1 = 0.187, 
                      L_medio_2= 1.0, 
                      eta_medio_1 = 1.0002926, 
                      eta_medio_2 = 1.667)

reflectancia = obj.Reflectancia()

"""
==============================================================================
Eliminando componente de DC
==============================================================================
"""

reflectancia -= np.mean(reflectancia)

"""
==============================================================================
Definicion de parametros
==============================================================================
"""
# Definiendo limite de busqueda en el espectro de Fourier (OPL en milimetros)
lim_inf = 0 # mm 
lim_sup = 6 # mm
       

"""
==============================================================================
Calculando la FFT
==============================================================================
"""
opl,amp = encontrar_FFT_dominio_en_OPL(lambda_inicial = lambda_inicial, 
                                       lambda_final = lambda_final, 
                                       senal=reflectancia)



"""
==============================================================================
Estableciendo rango de busqueda en el espacio de Fourier  
==============================================================================
"""

# Implementaremos un KNN de Machine Learning para encontrar el vecino 
# mas cercano al limite de busqueda 
"""
Se realiza este procedimiento porque, en general, el lim_inf y lim_sup
no son multiplos enteros de el periodo de muestreo (T_muestreo_lambda) y por 
lo tanto al tratar de buscar el indice en el arreglo del OPL, en el dominio 
de Fourier, no se encontrara el valor exacto y esto devuelve un valor Null
"""

# Creamos objeto de la clase NN 
# Como solo ocupamos el vecino mas cercano entonces el parametro n_neighbors=1
nn = NearestNeighbors(n_neighbors=1)

# Encontrando el vecino mas cercano a los limites de busqueda en opl
"""
Dado que el metodo de Machine Learning solo funciona para datos bidimensionales
entonces transformamos el array de OPL en Fourier en una matriz
"""
nn.fit(opl.reshape((len(opl),1)))

"""
Identificamos el indice en el arreglo OPL del vecino mas cercano al valor del
limite inferior y superior con el uso del metodo kneighbors
"""
index_lim_inf = nn.kneighbors([[lim_inf]], 1, return_distance=False)[0,0]
index_lim_sup = nn.kneighbors([[lim_sup]], 1, return_distance=False)[0,0]

"""
Una vez obtenido el indice ya podemos identificar de que valor se trata en el
array del OPL y por tanto tambien de AMP
"""
lim_inf_ = opl[index_lim_inf]
lim_sup_ = opl[index_lim_sup]


"""
==============================================================================
Aplicacion de filtro FIR pasa-bajos
==============================================================================
"""


# Al realizar el cambio de variable beta = 1/lambda, tenemos que 
T_muestreo_beta = (1/lambda_inicial - 1/lambda_final)/len(reflectancia)

T_muestreo_beta_opl = T_muestreo_beta*(2*10**6)

# Creando objeto de la clase Filtro
filtro = Filtro(_senal=reflectancia, # senal a filtrar
                _T_muestreo=T_muestreo_beta_opl, # Periodo de muestreo
                _frec_corte=lim_sup, # Frecuencia de corte en unidades de T_muestreo
                _orden=901) # Orden del Filtro

# Filtrando por el metodo de las ventanas
#senal_filtrada = filtro.filtrar_por_ventana_de_gauss(sigma=0.2)
senal_filtrada = filtro.filtrar_por_ventana_de_gauss(sigma=0.2)
"""
==============================================================================
Aplicando FFT a la señal filtrada
==============================================================================
"""
# opl_, amp_ = encontrar_FFT(lambda_inicial, T_muestreo_lambda, senal_filtrada)
#opl_,amp_ = encontrar_FFT(lambda_=lambda_, Reflectancia=senal_filtrada)    
opl_,amp_ = encontrar_FFT_dominio_en_OPL(lambda_inicial=lambda_inicial, 
                                         lambda_final=lambda_final, 
                                         senal=senal_filtrada)

"""
==============================================================================
Aplicando tecnica WINDOWING:
    Esta tecnica ayuda a mejorar la definicion de los picos en el espectro
    de Fourier
==============================================================================
"""

# Construyendo una ventana w_n del mismo tamaño que el array de la senal

#w_n = ventana_de_gauss(orden=len(senal_filtrada), sigma=0.1)
#w_n = ventana_de_hanning(orden=len(senal_filtrada))
#w_n = ventana_flattop(orden=len(senal_filtrada_esc_lineal))
w_n = ventana_kaiser_bessel(orden=len(senal_filtrada), beta=float(10))

# Enventanado de la senal en escala lineal
senal_enventanada = senal_filtrada * w_n

"""
==============================================================================
Mejoramiento de la resolucion en Fourier post-windowing
==============================================================================
"""

# Mejorando la resolucion del espectro añadiendo 0 a los extremos del array

# Numero de ceros a agregar en cada extremo
n_zeros = 1000
zeros = list(np.zeros(n_zeros))

# Agregamos los ceros a cada extremo de la señal
senal_enventanada = zeros + list(senal_enventanada)
senal_enventanada = np.array(senal_enventanada + zeros)

# Agregando las correspondientes longitudes de onda (virtuales) correspondientes
# a los ceros añadidos a los extremos, notese que T_muestreo no cambia 
lambda_mejorada = np.arange(1510-n_zeros*T_muestreo_lambda, 
                    1590 + n_zeros*T_muestreo_lambda + T_muestreo_lambda, # quiando +0.001
                    T_muestreo_lambda) 


"""
==============================================================================
Aplicando FFT a la señal mejorada
==============================================================================
"""

# Calculando la FFT de la señal enventanada

#opl_env, amp_env = encontrar_FFT(lambda_inicial,
#                                 T_muestreo_lambda, 
#                                 senal_enventanada)

# Primer indice del array
lambda_inicial = lambda_mejorada[0]
# Ultimo indice del array
lambda_final = lambda_mejorada[-1]
opl_env, amp_env = encontrar_FFT_dominio_en_OPL(lambda_inicial= lambda_inicial,
                                                lambda_final= lambda_final, 
                                                senal=senal_enventanada)

"""
==============================================================================
Eliminando componente de DC a la señal mejorada
==============================================================================
"""

# Eliminando la componenete de DC por medio de eliminar los primeros indices
# Cuantos componentes eliminados? 

#dc_margen = 0.15  # mm

nn.fit(opl_env.reshape((len(opl_env),1)))
#index_dc_margen = nn.kneighbors([[dc_margen]], 1, return_distance=False)[0,0]

#amp_env[:index_dc_margen] = np.zeros(index_dc_margen)


"""
==============================================================================
Encontrando los valores maximos
==============================================================================
"""

# La funcion de find_peaks de Scipy ya tiene la capacidad de realizar esto

# Encontrando el vecino mas cercano a los limites de busqueda en opl_env

index_lim_inf = nn.kneighbors([[lim_inf]], 1, return_distance=False)[0,0]
index_lim_sup = nn.kneighbors([[lim_sup]], 1, return_distance=False)[0,0]

# Limitamos la busqueda
amp_env_temp = amp_env[index_lim_inf:index_lim_sup]

# Necesitamos definir un valor limite en altura en el grafico de la amplitud
# se buscaran los maximos que superen este valor
lim_amp = 0.0015

# Buscando maximos en la region limitada
picos, _ = find_peaks(amp_env_temp, height = lim_amp)

# Como limitamos la busqueda hay que compensar con los indices anteriores 
# en el array para obtener el valor verdadero 
picos += index_lim_inf 

maximos = opl_env[picos]

# Imprimiendo resultados
print("====================================================================")
for maximo in maximos: 
    print("Maximo localizado en: %.3f mm" % maximo)

print("====================================================================")


"""
==============================================================================
Graficando resultados
==============================================================================
"""

# Creando figura
fig,ax = plt.subplots(figsize=(40,20))
fig.set_tight_layout(True)
# Pone lo mas juntas las graficas posibles
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)

# Cambiando el tamano de la fuente en todos los ejes
plt.rcParams.update({'font.size': 20})

# Graficando el espectro optico inicial
ax = plt.subplot(3,2,1)
espectro_graph, = ax.plot(lambda_,reflectancia, linewidth=1.5, 
                          label= "Medición")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"Reflectancia $[u.a.]$", fontsize=30)
ax.set_title(label="Dominio óptico", fontsize=30)
#ax.set_ylim([-40,-10])
ax.legend(loc="best",fontsize=30)

# Graficando la FFT del espectro inicial
ax = plt.subplot(3,2,2)
fft_graph, = ax.plot(opl,amp, linewidth=1.5,color="purple")
ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
ax.set_ylabel(ylabel=r"Magnitud $[u.a.]$", fontsize=30)
ax.set_title(label="Dominio de Fourier", fontsize=30)
ax.set_xlim([lim_inf_,lim_sup_])
#ax.set_ylim([0,1])

# Graficando el espectro optico de la señal filtrada
ax = plt.subplot(3,2,3)
espectro_graph, = ax.plot(lambda_,senal_filtrada, linewidth=1.5, 
                          label="Señal filtrada")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a.]$", fontsize=30)
ax.set_title(label="Dominio óptico", fontsize=30)
#ax.set_ylim([-40,-10])
ax.legend(loc="lower left",fontsize=30)

# Graficando la FFT de la señal filtrada
ax = plt.subplot(3,2,4)
fft_graph, = ax.plot(opl_,amp_, linewidth=1.5,color="teal")
ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a.]$", fontsize=30)
ax.set_title(label="Dominio de Fourier", fontsize=30)
ax.set_xlim([lim_inf_,lim_sup_])
#ax.set_ylim([0,1])


# Graficando el espectro optico de la señal tratada 
ax = plt.subplot(3,2,5)
espectro_graph, = ax.plot(lambda_mejorada, senal_enventanada, linewidth=1.5,
                          label="Señal mejorada")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a.]$", fontsize=30)
ax.set_title(label="Dominio óptico escala lineal", fontsize=30)
#ax.set_ylim([-40,-10])
ax.legend(loc="upper left",fontsize=30)

# Graficando FFT de la señal tratada
ax = plt.subplot(3,2,6)
fft_graph, = ax.plot(opl_env,amp_env, linewidth=1.5,color="navy")
ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a]$", fontsize=30)
ax.set_title(label="Dominio de Fourier", fontsize=30)
ax.set_xlim([lim_inf_,lim_sup_])
#ax.set_ylim([0,2])

# Guardando figura
plt.savefig("Filtro.png")
# Mostrando Figura
plt.show()


