#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:50:56 2021

@author: alejandro_goper

Este script es para analizar el efecto vernier, es un analisis similar al 
realizado en el script envolventes.py
"""

from FabryPerot.Filtros_support import Filtro, ventana_de_gauss, ventana_de_hanning, ventana_flattop, ventana_kaiser_bessel
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt

"""
==============================================================================
Importando Datos
==============================================================================
"""

# Importando archivos 

#fecha_medicion = "1-2-2022"

#carpeta = "3GAP-CAPILAR-AIRE-FIBRA-AIRE-100nm"

# ruta_directorio = "../" + fecha_medicion + "/" + carpeta

ruta_directorio = "../" + "Mediciones_Monse" + "/" + "4xLcav" + "/" 

nombre_archivo = "Espectro (1).txt"

path = ruta_directorio + "/" + nombre_archivo

data = np.loadtxt(path, skiprows=58)

path = ruta_directorio + "/Referencia.txt"

referencia = np.loadtxt(path, skiprows = 58)

# Separando datos de la referencia por columnas

lambda_ref, potencia_dBm_ref = referencia[:,0], referencia[:,1]
lambda_, potencia_dBm = data[:,0], data[:,1]

# Damos por hecho que lambda_ = lambda_ref

"""
==============================================================================
Normalizando respecto a la referencia
==============================================================================
"""

# Normalizando

potencia_dB = potencia_dBm  - potencia_dBm_ref

# Cambiando a escala lineal
potencia = 10**(potencia_dB/10)

"""
==============================================================================
Definicion de parametros
==============================================================================
"""

lambda_inicial = lambda_[0] # Valor inicial del arreglo
lambda_final = lambda_[-1] # Valor final del arreglo
n = len(lambda_) # Cantidad de datos en el arreglo

T_muestreo_lambda = (lambda_final-lambda_inicial)/n
       

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

# Frecuencia de corte en [mm] para el filtro pasa bajos
f_c = 2.0 # mm 

# Creando objeto de la clase Filtro
filtro = Filtro(_senal=potencia, # senal a filtrar
                _T_muestreo=T_muestreo_beta_opl, # Periodo de muestreo
                _frec_corte=f_c, # Frecuencia de corte en unidades de T_muestreo
                _orden=901) # Orden del Filtro

# Filtrando por el metodo de las ventanas
senal_filtrada = filtro.filtrar_por_ventana_de_gauss(sigma=0.2)



"""
==============================================================================
Encontrando las envolventes superior e inferior 
==============================================================================
"""

lim_amplitud = senal_filtrada.min()
# Buscando picos por encima del valor minimo
picos, _ = find_peaks(senal_filtrada, height = lim_amplitud)

envolvente_superior = senal_filtrada[picos]
lambda_envolvente_superior = lambda_[picos]


# Para encontrar los minimos multiplicamos la senal original por -1
senal_invertida = - senal_filtrada
lim_amplitud = senal_invertida.min()
picos, _ = find_peaks(senal_invertida,height= lim_amplitud)


envolvente_inferior = senal_filtrada[picos]
lambda_envolvente_inferior = lambda_[picos]




"""
==============================================================================
Empiricamente se descubrio que la envolvente superior era la unica rastrable 
y que habia un patron bien definido en el rango [1,3] asi que, debemos 
aplicar el procedimiento de identificar de nuevo la envolvente para esta 
señal
==============================================================================
"""

lim_inferior = 0.1
lim_superior = 0.5

prom = (lim_inferior + lim_superior)/2

indices_senal_a_seguir = np.where((envolvente_inferior<lim_superior) & 
                                  (envolvente_inferior >= lim_inferior))
senal_a_seguir = envolvente_inferior[indices_senal_a_seguir]
lambda_senal_a_seguir = lambda_envolvente_inferior[indices_senal_a_seguir] 


# Buscando la envolvente superior e inferior de la senal rastreable
# este debe ser aproximadamente el punto medio del rango de busqueda

index_env_sup_senal_rastrable = np.where(senal_a_seguir >= prom) 
env_sup_senal_rastreable = senal_a_seguir[index_env_sup_senal_rastrable]
lambda_env_sup_senal_rastreable = lambda_senal_a_seguir[index_env_sup_senal_rastrable]



index_env_inf_senal_rastrable = np.where(senal_a_seguir < prom) 
env_inf_senal_rastreable = senal_a_seguir[index_env_inf_senal_rastrable]
lambda_env_inf_senal_rastreable = lambda_senal_a_seguir[index_env_inf_senal_rastrable]


"""
==============================================================================
Encontrando los puntos de interseccion:
    Se tomaran los puntos maximos de la envolvente inferior y los minimos de
    la envolvente inferior y luego realizamos un promedio de ambos valores
    
* Para relacionar los maximos con los minimos correspondientes se realiza la 
la comparacion punto a punto entre la envolvente inferior y superior, ambos
puntos son interseccion si la distancia entre ellos es menor a tol en nm 
==============================================================================
"""

# Este parametro ha sido proporcionado empiricamente a partir de los espectros
# para la envolvente inferior
lim_amplitud = env_inf_senal_rastreable.min()

# Buscando picos por encima del valor minimo de la señal
picos, _ = find_peaks(env_inf_senal_rastreable, height = lim_amplitud)

intersecciones_inferior=lambda_env_inf_senal_rastreable[picos]


# Este parametros ha sido encontrado empiricamente
lim_amplitud = -env_sup_senal_rastreable.max()
# Buscando picos por encima de 0.045
picos, _ = find_peaks(-env_sup_senal_rastreable, height = lim_amplitud)
intersecciones_superior = lambda_env_sup_senal_rastreable[picos]


# Lista para almacenar las intersecciones
intersecciones = []

# Tolerancia en lambda
tol = 10 #nm

# Tomaremos siempre el arreglo con menos puntos como el arreglo principal
if(len(intersecciones_superior) < len(intersecciones_inferior)):
    # Para cada punto del arreglo de intersecciones superior
    for punto in intersecciones_superior:    
        # calculamos la distancia a todos los puntos del arreglo de 
        # intersecciones inferior
        distancia = np.abs(punto - intersecciones_inferior)
        # Buscamos los indices de los puntos cuya distancia sea 
        index =np.where(distancia < tol)
        # Primero verificamos si el arreglo index no esta vacio 
        if(np.size(index)):
            # Si no esta vacio se realiza el promedio entre los puntos
            
            # Inciamos la sumatoria de los puntos
            sumatoria = punto
            # Encontramos los puntos cercanos al punto en cuestion 
            puntos_cercanos = intersecciones_inferior[index]
            # variable que controla el numero total de puntos
            m = len(puntos_cercanos) + 1
            # Realizamos la sumatoria de los puntos
            for p in puntos_cercanos:
                sumatoria += p
            # La interseccion es el promedio de los puntos cercanos
            interseccion = sumatoria/m
            intersecciones.append(interseccion)
            
elif(len(intersecciones_superior) > len(intersecciones_inferior)):
   for punto in intersecciones_inferior:    
        distancia = np.abs(punto - intersecciones_superior)
        index =np.where(distancia < tol)
        if(np.size(index)):
            sumatoria = punto
            puntos_cercanos = intersecciones_superior[index]
            m = len(puntos_cercanos) + 1
            for p in puntos_cercanos:
                sumatoria += p
            interseccion = sumatoria / m 
            intersecciones.append(interseccion)
    
else: 
    for punto in intersecciones_superior:    
        distancia = np.abs(punto - intersecciones_inferior)
        index =np.where(distancia < tol)
        if(np.size(index)):
            sumatoria = punto
            puntos_cercanos = intersecciones_inferior[index]
            m = len(puntos_cercanos) + 1
            for p in puntos_cercanos:
                sumatoria += p
            
            interseccion = sumatoria/m
            intersecciones.append(interseccion)

# Convirtiendo a array numpy
np.array(intersecciones)
    

"""
==============================================================================
Graficando resultados
==============================================================================
"""

# Creando figura
fig,ax = plt.subplots(figsize=(40,20))
# Pone lo mas juntas las graficas posibles
fig.set_tight_layout(True)
# Para que no se empalmen los titulos en los ejes
fig.subplots_adjust(wspace=1.2)

# Cambiando el tamano de la fuente en todos los ejes
plt.rcParams.update({'font.size': 20})

# Graficando el espectro optico inicial
ax = plt.subplot(2,2,1)
espectro_graph, = ax.plot(lambda_,potencia, linewidth=1.5, 
                          label= "Medición")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a]$", fontsize=30)
ax.set_title(label="Dominio óptico", fontsize=30)
#ax.set_ylim([-40,-10])
ax.legend(loc="best",fontsize=30)



# Graficando espectro filtrado
ax = plt.subplot(2,2,2)
espectro_graph, = ax.plot(lambda_,senal_filtrada, linewidth=1.5,color="purple",
                          label="Señal filtrada")
ax.scatter(lambda_env_sup_senal_rastreable, 
                            env_sup_senal_rastreable, 
                             s=150, c="red", label="Envolvente")
ax.scatter(lambda_env_inf_senal_rastreable, 
                            env_inf_senal_rastreable, 
                            s=150, c="red")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a]$", fontsize=30)
ax.set_title(label="Dominio óptico", fontsize=30)
ax.legend(loc="best")
#ax.set_xlim([lim_inf_,lim_sup_])
#ax.set_ylim([0,1])


"""
# Graficando envolvente superior
ax = plt.subplot(2,2,3)
espectro_graph = ax.scatter(lambda_env_sup_senal_rastreable, 
                            env_sup_senal_rastreable, 
                             s=150, c="red", label="Envolvente")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a]$", fontsize=30)
ax.set_title(label="Envolvente Superior", fontsize=30)
ax.legend(loc="best")
#ax.set_xlim([lim_inf_,lim_sup_])
#ax.set_ylim([0,1])


# Graficando envolvente inferior
ax = plt.subplot(2,2,4)
espectro_graph = ax.scatter(lambda_env_inf_senal_rastreable, 
                            env_inf_senal_rastreable, 
                            s=150, c="black", label="Envolvente a seguir")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a]$", fontsize=30)
ax.set_title(label="Envolvente Inferior", fontsize=30)
ax.legend(loc="best")
#ax.set_xlim([lim_inf_,lim_sup_])
#ax.set_ylim([0,1])

"""

# Graficando envolvente superior
ax = plt.subplot(2,2,3)
espectro_graph = ax.scatter(lambda_envolvente_superior, 
                            envolvente_superior, 
                             s=150, c="red", label="Envolvente")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a]$", fontsize=30)
ax.set_title(label="Envolvente Superior", fontsize=30)
ax.legend(loc="best")
#ax.set_xlim([lim_inf_,lim_sup_])
#ax.set_ylim([0,1])


# Graficando envolvente inferior
ax = plt.subplot(2,2,4)
espectro_graph = ax.scatter(lambda_envolvente_inferior, 
                            envolvente_inferior, 
                            s=150, c="black", label="Envolvente a seguir")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a]$", fontsize=30)
ax.set_title(label="Envolvente Inferior", fontsize=30)
ax.legend(loc="best")
#ax.set_xlim([lim_inf_,lim_sup_])
#ax.set_ylim([0,1])



#Creando caja de texto para mostrar los resultados en la imagen


# Concatenando las intersecciones
text = ""
for interseccion in intersecciones: 
    text += "\nInterseccion en: %.3f mm" % interseccion
# Eliminando el espacio en blanco inicial
text = text[1:]

# Creando cadena para caja de texto
textstr = text 

# Estas son propiedades de matplotlib.patch.Patch
props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)

graph_text = ax.text(0.05, 0.25, textstr, transform=ax.transAxes, fontsize=35,
        verticalalignment='top', bbox=props, color="black")

# Guardando figura

plt.savefig(ruta_directorio + "-" + nombre_archivo + ".png")
# Mostrando Figura

plt.show()

"""
Guardando datos de las envolventes superior e inferior.
"""


x_ = lambda_env_sup_senal_rastreable.reshape(len(lambda_env_sup_senal_rastreable),1)
xy = np.append(x_, env_sup_senal_rastreable.reshape(len(env_sup_senal_rastreable),1), axis=1)

np.savetxt(ruta_directorio + "/" + "Envolvente_superior_" + nombre_archivo, xy, fmt="%f" , header="Lambda, Potencia_escala_lineal", delimiter = ",")

x = lambda_env_inf_senal_rastreable.reshape(len(lambda_env_inf_senal_rastreable),1)
xy = np.append(x, env_inf_senal_rastreable.reshape(len(env_inf_senal_rastreable),1), axis=1)

np.savetxt(ruta_directorio + "/" + "Envolvente_inferior_" + nombre_archivo, xy, fmt="%f" , header="Lambda, Potencia_escala_lineal", delimiter = ",")
