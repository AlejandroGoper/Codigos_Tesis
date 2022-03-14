#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 19:06:57 2021

@author: alejandro_goper

Codigo para crear animacion de la simulacion FabryPerot

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from FabryPerot.Clase import FabryPerot_2GAP
from FabryPerot.Filtros_support import *
from FabryPerot.FFT_support import encontrar_FFT_dominio_en_OPL
from sklearn.neighbors import NearestNeighbors

"""
==============================================================================
Creamos la figura y definimos parametros importantes de ella
==============================================================================
"""

# Creamos la figura
fig,ax = plt.subplots(figsize=(40,20))

# Para que las graficas no se empalmen
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)
# Cambiando el tamano de la fuente en todos los ejes
plt.rcParams.update({'font.size': 25})


"""
==============================================================================
Definicion de parametros
==============================================================================
"""
# Dominio de la senal optica 
lambda_ = np.arange(1510,1590,0.01)


# Definiendo limite de busqueda en el espectro de Fourier (OPL en milimetros)
lim_inf = 0 # mm 
lim_sup = 3 # mm

#Periodo de muestreo = (lambda_[-1] - lambda_[0])/len(lambda_) Approx 0.005 nm

lambda_inicial = lambda_[0] # Valor inicial del arreglo
lambda_final = lambda_[-1] # Valor final del arreglo
n = len(lambda_) # Cantidad de datos en el arreglo

T_muestreo_lambda = (lambda_final-lambda_inicial)/n 
       


"""
==============================================================================
Obteniendo la senal simulada
==============================================================================
"""
obj = FabryPerot_2GAP(lambda_inicial=1510,lambda_final= 1590, 
                      T_muestreo_lambda = 0.01, L_medio_1 = .4, 
                      L_medio_2=.8, eta_medio_1 = 1.0, eta_medio_2 = 1.332, 
                      eta_medio_3=1.48)
reflectancia = obj.R()



"""
==============================================================================
Encontrando la transformada de Fourier de la senal simulada
==============================================================================
"""

opl, amp = encontrar_FFT_dominio_en_OPL(lambda_inicial=lambda_[0], 
                                            lambda_final = lambda_[-1],
                                            senal=reflectancia)



"""
==============================================================================
Estableciendo rango de busqueda en el espacio de Fourier  
==============================================================================
"""

# Implementaremos un KNN de Machine Learning para encontrar el vecino 
# mas cercano al limite de busqueda 
"""
******************************************************************************
Se realiza este procedimiento porque, en general, el lim_inf y lim_sup
no son multiplos enteros de el periodo de muestreo (T_muestreo_lambda) y por 
lo tanto al tratar de buscar el indice en el arreglo del OPL, en el dominio 
de Fourier, no se encontrara el valor exacto y esto devuelve un valor Null
******************************************************************************
"""

# Creamos objeto de la clase NearestNeighbors 
# Como solo ocupamos el vecino mas cercano entonces el parametro n_neighbors=1
nn = NearestNeighbors(n_neighbors=1)

# Encontrando el vecino mas cercano a los limites de busqueda en opl
"""
*******************************************************************************
Dado que el metodo de Machine Learning solo funciona para datos bidimensionales
entonces transformamos el array del OPL en una matriz
*******************************************************************************
"""
nn.fit(opl.reshape((len(opl),1)))

"""
******************************************************************************
Identificamos el indice en el arreglo OPL del vecino mas cercano al valor del
limite inferior y superior con el uso del metodo kneighbors
******************************************************************************
"""
index_lim_inf = nn.kneighbors([[lim_inf]], 1, return_distance=False)[0,0]
index_lim_sup = nn.kneighbors([[lim_sup]], 1, return_distance=False)[0,0]

"""
******************************************************************************
Una vez obtenido el indice ya podemos identificar de que valor se trata en el
array del OPL y por tanto tambien de AMP
******************************************************************************
"""
lim_inf_ = opl[index_lim_inf]
lim_sup_ = opl[index_lim_sup]


"""
==============================================================================
Aplicacion de filtro FIR pasa-bajos de fase lineal 
==============================================================================
"""

# Al realizar el cambio de variable beta = 1/lambda, tenemos que 
T_muestreo_beta = (1/lambda_inicial - 1/lambda_final)/n

# Multiplicamos todo el array por un factor de 2x10**6
"""
******************************************************************************
Se multiplica por este factor para que las unidades de T_muestreo esten en
milimetros y reflejen el valor del OPL, una explicacion más detallada se
encuentra en el archivo FFT_support 
******************************************************************************
"""
T_muestreo_beta_opl = T_muestreo_beta*(2*10**6)


# Creando objeto de la clase Filtro
filtro = Filtro(_senal=reflectancia, # senal a filtrar
                _T_muestreo=T_muestreo_beta_opl, # Periodo de muestreo
                _frec_corte=lim_sup, # Frecuencia de corte en unidades de T_muestreo
                _orden=901) # Orden del Filtro

# Filtrando por el metodo de las ventanas
senal_filtrada = filtro.filtrar_por_ventana_de_gauss(sigma=0.2)


"""
==============================================================================
Aplicando FFT a la señal filtrada
==============================================================================
"""

opl_filt,amp_filt = encontrar_FFT_dominio_en_OPL(lambda_inicial=lambda_inicial, 
                                                 lambda_final=lambda_final, 
                                                  senal=senal_filtrada)

"""
==============================================================================
Cambiando la señal filtrada a escala Lineal
==============================================================================
"""

# Cambiando a escala lineal

senal_filtrada_esc_lineal = 10**(senal_filtrada/10)

"""
==============================================================================
Aplicando tecnica WINDOWING:
    Esta tecnica ayuda a mejorar la definicion de los picos en el espectro
    de Fourier
==============================================================================
"""

# Construyendo una ventana w_n del mismo tamaño que el array de la senal

#w_n = ventana_de_gauss(orden=len(senal_filtrada_esc_lineal), sigma=0.05)
w_n = ventana_de_hanning(orden=len(senal_filtrada_esc_lineal))
# w_n = ventana_flattop(orden=len(senal_filtrada_esc_lineal))

"""
La ventana de Keiser-Bessel es similar a otras ventanas para distintos 
valores del parametro beta,  por ejemplo:
    - beta = 0 - Ventana cuadrada
    - beta = 5 - Ventana de Hamming 
    - beta = 6 - Ventana de Hanning
    - beta = 8.6 - Ventana de Blackman - Harris
"""
#w_n = ventana_kaiser_bessel(orden=len(senal_filtrada_esc_lineal), beta=6)
# Enventanado de la senal en escala lineal
senal_enventanada = senal_filtrada_esc_lineal * w_n



"""
==============================================================================
Mejoramiento de la resolucion en Fourier post-windowing
==============================================================================
"""

# Mejorando la resolucion del espectro añadiendo 0 a los extremos del array

# Numero de ceros a agregar en cada extremo
n_zeros = 10000
"""
******************************************************************************
Empiricamente se ha determinado que cuando n_zeros > 10 000 entonces
el espectro se desplaza por lo que se sugiere usar 0 < n_zeros < 1000
******************************************************************************
"""
zeros = list(np.zeros(n_zeros))

# Agregamos los ceros a cada extremo de la señal
senal_enventanada = zeros + list(senal_enventanada)
senal_enventanada = np.array(senal_enventanada + zeros)

# Agregando las correspondientes longitudes de onda (virtuales) 
# correspondientes a los ceros añadidos a los extremos 
lambda_mejorada = np.arange(lambda_inicial-n_zeros*T_muestreo_lambda, 
                    lambda_final + n_zeros*T_muestreo_lambda-0.00001,
                    T_muestreo_lambda) 

# Le agregamos un -0.00001 al final del segun parametro para asegurar que la
# longitud de lambda_mejorada sea la misma que la de la senal enventanada
# dado que si se lo quitamos en ocaciones la longitud difiere por un valor  

"""
==============================================================================
Aplicando FFT a la señal mejorada
==============================================================================
"""

# Dado que hemos extendido el dominio en lambda calculamos de nuevo
lambda_inicial = lambda_mejorada[0]
lambda_final = lambda_mejorada[-1]

# Calculando la FFT de la señal enventanada
opl_env, amp_env = encontrar_FFT_dominio_en_OPL(lambda_inicial=lambda_inicial, 
                                                lambda_final=lambda_final, 
                                                senal=senal_enventanada)

"""
==============================================================================
Eliminando componente de DC a la señal mejorada
==============================================================================
"""

# Eliminando la componenete de DC hasta un margen fijo en el opl
dc_margen = 0.05 # mm

# buscamos el indice en el array opl mas cercano a dc_margen
nn.fit(opl_env.reshape((len(opl_env),1)))
index_dc_margen = nn.kneighbors([[dc_margen]], 1, return_distance=False)[0,0]
# eliminamos todas las contribuciones del espectro de fourier hasta dc_margen
amp_env[:index_dc_margen] = np.zeros(index_dc_margen)


"""
==============================================================================
Definiendo graficas
==============================================================================
"""

ax = plt.subplot(1,2,1)

senal_dB = 10*np.log10(reflectancia)

graph, = ax.plot(lambda_,senal_dB, linewidth = 1.5)
ax.set_ylabel(r"R [dB]", fontsize = 30)
ax.set_xlabel(r"$\lambda$", fontsize=30)
ax.set_xlim([1510,1590])
ax.set_ylim([-40,-5])
ax.set_title("Simulación FP - 2 GAP", fontsize=40)

ax = plt.subplot(1,2,2)

graph_2, = ax.plot(opl_env,amp_env, c="green", linewidth=1.5)
#ax.title.set_text("FFT")

ax.set_ylabel(r" |R|",fontsize=30 )
ax.set_xlabel(r"OPL[$mm$]", fontsize=30)
ax.set_xlim([-0.1,3])
ax.set_ylim([0,0.01])
ax.set_title("FFT", fontsize=40)


def actualizar(i):
    di = 0.02
    L1 = round((i+1)*di,2)
    label = f"$L_{1}$ = {L1} mm"
    lambda_inicial = 1510
    lambda_final = 1590
    obj = FabryPerot_2GAP(lambda_inicial, lambda_final, 
                          T_muestreo_lambda= 0.01, L_medio_1 = L1, 
                          L_medio_2=0.8, eta_medio_1 = 1.0, 
                          eta_medio_2 = 1.332, eta_medio_3=1.48)
    senal = obj.R()
    senal_dB = 10*np.log10(senal)
    
    
    """
    ==============================================================================
    Encontrando la transformada de Fourier de la senal simulada
    ==============================================================================
    """
    
    opl, amp = encontrar_FFT_dominio_en_OPL(lambda_inicial, 
                                                lambda_final,
                                                senal=senal_dB)
        
    
    
    """
    ==============================================================================
    Estableciendo rango de busqueda en el espacio de Fourier  
    ==============================================================================
    """
    
    # Implementaremos un KNN de Machine Learning para encontrar el vecino 
    # mas cercano al limite de busqueda 
    """
    ******************************************************************************
    Se realiza este procedimiento porque, en general, el lim_inf y lim_sup
    no son multiplos enteros de el periodo de muestreo (T_muestreo_lambda) y por 
    lo tanto al tratar de buscar el indice en el arreglo del OPL, en el dominio 
    de Fourier, no se encontrara el valor exacto y esto devuelve un valor Null
    ******************************************************************************
    """
    
    # Creamos objeto de la clase NearestNeighbors 
    # Como solo ocupamos el vecino mas cercano entonces el parametro n_neighbors=1
    nn = NearestNeighbors(n_neighbors=1)
    
    # Encontrando el vecino mas cercano a los limites de busqueda en opl
    """
    *******************************************************************************
    Dado que el metodo de Machine Learning solo funciona para datos bidimensionales
    entonces transformamos el array del OPL en una matriz
    *******************************************************************************
    """
    nn.fit(opl.reshape((len(opl),1)))
    
    """
    ******************************************************************************
    Identificamos el indice en el arreglo OPL del vecino mas cercano al valor del
    limite inferior y superior con el uso del metodo kneighbors
    ******************************************************************************
    """
    index_lim_inf = nn.kneighbors([[lim_inf]], 1, return_distance=False)[0,0]
    index_lim_sup = nn.kneighbors([[lim_sup]], 1, return_distance=False)[0,0]
    
    """
    ******************************************************************************
    Una vez obtenido el indice ya podemos identificar de que valor se trata en el
    array del OPL y por tanto tambien de AMP
    ******************************************************************************
    """
    lim_inf_ = opl[index_lim_inf]
    lim_sup_ = opl[index_lim_sup]
    
    
    """
    ==============================================================================
    Aplicacion de filtro FIR pasa-bajos de fase lineal 
    ==============================================================================
    """
    
    # Al realizar el cambio de variable beta = 1/lambda, tenemos que 
    T_muestreo_beta = (1/lambda_inicial - 1/lambda_final)/n
    
    # Multiplicamos todo el array por un factor de 2x10**6
    """
    ******************************************************************************
    Se multiplica por este factor para que las unidades de T_muestreo esten en
    milimetros y reflejen el valor del OPL, una explicacion más detallada se
    encuentra en el archivo FFT_support 
    ******************************************************************************
    """
    T_muestreo_beta_opl = T_muestreo_beta*(2*10**6)
    
    
    # Creando objeto de la clase Filtro
    filtro = Filtro(_senal=senal_dB, # senal a filtrar
                    _T_muestreo=T_muestreo_beta_opl, # Periodo de muestreo
                    _frec_corte=lim_sup, # Frecuencia de corte en unidades de T_muestreo
                    _orden=901) # Orden del Filtro
    
    # Filtrando por el metodo de las ventanas
    senal_filtrada = filtro.filtrar_por_ventana_de_gauss(sigma=0.2)
    
    
    """
    ==============================================================================
    Aplicando FFT a la señal filtrada
    ==============================================================================
    """
    
    opl_filt,amp_filt = encontrar_FFT_dominio_en_OPL(lambda_inicial=lambda_inicial, 
                                                     lambda_final=lambda_final, 
                                                     senal=senal_filtrada)
    
    """
    ==============================================================================
    Cambiando la señal filtrada a escala Lineal
    ==============================================================================
    """
    
    # Cambiando a escala lineal
    
    senal_filtrada_esc_lineal = 10**(senal_filtrada/10)
    
    """
    ==============================================================================
    Aplicando tecnica WINDOWING:
    Esta tecnica ayuda a mejorar la definicion de los picos en el espectro
    de Fourier
    ==============================================================================
    """        
    # Construyendo una ventana w_n del mismo tamaño que el array de la senal
    
    #w_n = ventana_de_gauss(orden=len(senal_filtrada_esc_lineal), sigma=0.05)
    w_n = ventana_de_hanning(orden=len(senal_filtrada_esc_lineal))
    # w_n = ventana_flattop(orden=len(senal_filtrada_esc_lineal))
    
    """
    La ventana de Keiser-Bessel es similar a otras ventanas para distintos 
    valores del parametro beta,  por ejemplo:
        - beta = 0 - Ventana cuadrada
        - beta = 5 - Ventana de Hamming 
        - beta = 6 - Ventana de Hanning
        - beta = 8.6 - Ventana de Blackman - Harris
        """
    #w_n = ventana_kaiser_bessel(orden=len(senal_filtrada_esc_lineal), beta=6)
    # Enventanado de la senal en escala lineal
    senal_enventanada = senal_filtrada_esc_lineal * w_n
    
    
    
    """
    ==============================================================================
    Mejoramiento de la resolucion en Fourier post-windowing
    ==============================================================================
    """
    
    # Mejorando la resolucion del espectro añadiendo 0 a los extremos del array
    
    # Numero de ceros a agregar en cada extremo
    n_zeros = 10000
    """
    ******************************************************************************
    Empiricamente se ha determinado que cuando n_zeros > 10 000 entonces
    el espectro se desplaza por lo que se sugiere usar 0 < n_zeros < 1000
    ******************************************************************************
    """
    zeros = list(np.zeros(n_zeros))
    
    # Agregamos los ceros a cada extremo de la señal
    senal_enventanada = zeros + list(senal_enventanada)
    senal_enventanada = np.array(senal_enventanada + zeros)
            
    # Agregando las correspondientes longitudes de onda (virtuales) 
    # correspondientes a los ceros añadidos a los extremos 
    lambda_mejorada = np.arange(lambda_inicial-n_zeros*T_muestreo_lambda, 
                                lambda_final + n_zeros*T_muestreo_lambda-0.00001,
                                T_muestreo_lambda) 
    
    # Le agregamos un -0.00001 al final del segun parametro para asegurar que la
    # longitud de lambda_mejorada sea la misma que la de la senal enventanada
    # dado que si se lo quitamos en ocaciones la longitud difiere por un valor  
    
    """
    ==============================================================================
    Aplicando FFT a la señal mejorada
    ==============================================================================
    """
    
    # Dado que hemos extendido el dominio en lambda calculamos de nuevo
    lambda_inicial = lambda_mejorada[0]
    lambda_final = lambda_mejorada[-1]
    
    # Calculando la FFT de la señal enventanada
    opl_env, amp_env = encontrar_FFT_dominio_en_OPL(lambda_inicial=lambda_inicial, 
                                                    lambda_final=lambda_final, 
                                                    senal=senal_enventanada)
    
    """
    ==============================================================================
    Eliminando componente de DC a la señal mejorada
    ==============================================================================
    """
    
    # Eliminando la componenete de DC hasta un margen fijo en el opl
    dc_margen = 0.05 # mm
    
    # buscamos el indice en el array opl mas cercano a dc_margen
    nn.fit(opl_env.reshape((len(opl_env),1)))
    index_dc_margen = nn.kneighbors([[dc_margen]], 1, return_distance=False)[0,0]
    # eliminamos todas las contribuciones del espectro de fourier hasta dc_margen
    amp_env[:index_dc_margen] = np.zeros(index_dc_margen)
    
    
            
            
            
            
            
            
            
                
    
    
    
    #x_fft, y_fft = encontrar_FFT_dominio_en_OPL(lambda_inicial = lambda_[0],
    #                                            lambda_final = lambda_[-1],
    #                                            senal=senal_dB)
    
    fig.suptitle(label, fontsize = 50)
    
    graph.set_ydata(senal_dB)
    #ax.set_title(label)
    graph_2.set_ydata(amp_env)
    return graph, graph_2, ax


anim = FuncAnimation(fig, actualizar, repeat = True, frames= np.arange(0,50),
                     interval = 1000 )

#plt.show()

# Guardaremos la animacion
Writer = writers["ffmpeg"]
writer = Writer(fps=1, metadata={"artist":"IAGP"}, bitrate=1800)
anim.save("Variacion_L1_FFT.mp4",writer)
