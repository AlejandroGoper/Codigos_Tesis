#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:50:56 2021

@author: alejandro_goper

Este script es para realizar pruebas de la obtencion de los puntos de 
interseccion entre señales particulares

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

# Importando un espectro del fabry perot: 2GAP-VIDRIO-AIRE-0.1um

# Importando archivos 

fecha_medicion = "9-11-2021"

carpeta = "2GAP-VIDRIO-AIRE-0.1um"

ruta_directorio = "../" + fecha_medicion + "/" + carpeta

nombre_archivo = "Espectro (12).txt"

path = ruta_directorio + "/" + nombre_archivo

data = np.loadtxt(path, skiprows=0)

path = ruta_directorio + "/referencia.txt"

referencia = np.loadtxt(path)

# Separando datos de la referencia por columnas

lambda_ref, potencia_dBm_ref = referencia[:,0], referencia[:,1]
lambda_, potencia_dBm = data[:,0], data[:,1]

# Damos por hecho que lambda_ = lambda_ref

"""
==============================================================================
Cambiando todo a escala lineal y realizando la resta de las señales
==============================================================================
"""

potencia_ref = 10**(potencia_dBm_ref/10)

potencia_ = 10**(potencia_dBm/10)


# Restando ambas señales

potencia = potencia_  - potencia_ref

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
lim_amplitud = 0
# Buscando picos por encima de 0
picos, _ = find_peaks(senal_filtrada, height = lim_amplitud)

envolvente_superior = senal_filtrada[picos]
lambda_envolvente_superior = lambda_[picos]

# Para encontrar los minimos multiplicamos la senal original por -1
senal_invertida = - senal_filtrada
picos, _ = find_peaks(senal_invertida,height= lim_amplitud)

envolvente_inferior = senal_filtrada[picos]
lambda_envolvente_inferior = lambda_[picos]


"""
==============================================================================
Encontrando los puntos de interseccion:
    Se tomaran los puntos maximos de la envolvente inferior y los minimos de
    la envolvente inferior y luego realizamos un promedio de ambos valores
    
* Para relacionar los maximos con los minimos correspondientes se realiza la 
la comparacion punto a punto entre la envolvente inferior y superior, ambos
puntos son interseccion si la distancia entre ellos es menor a 1.5 mm 
==============================================================================
"""

lim_amplitud = -0.0002
# Buscando picos por encima de -0.0002
picos, _ = find_peaks(envolvente_inferior, height = lim_amplitud)

intersecciones_inferior=lambda_envolvente_inferior[picos]


# Buscando picos por encima de -0.0002
picos, _ = find_peaks(-envolvente_superior, height = lim_amplitud)
intersecciones_superior = lambda_envolvente_superior[picos]


# Tomaremos siempre el arreglo con menos puntos como el arreglo principal

if(len(intersecciones_superior) < len(intersecciones_inferior)):
    for punto in intersecciones_superior:    
        distancia = np.abs(punto - intersecciones_inferior)
        index =np.where(distancia < 1.8)[0]
        if(index != None):
            interseccion = 0.5*(punto + intersecciones_inferior[int(index[0])])
            print("Intersección en: ", interseccion)
    
            
elif(len(intersecciones_superior) > len(intersecciones_inferior)):
   for punto in intersecciones_inferior:    
        distancia = np.abs(punto - intersecciones_superior)
        index =np.where(distancia < 1.8)[0]
        if(index != None):
            interseccion = 0.5*(punto + intersecciones_superior[int(index[0])])
            print("Intersección en: ", interseccion)
    
else: 
    for punto in intersecciones_superior:    
        distancia = np.abs(punto - intersecciones_inferior)
        index =np.where(distancia < 1.8)[0]
        if(index != None):
            interseccion = 0.5*(punto + intersecciones_inferior[int(index[0])])
            print("Intersección en: ", interseccion)
    
    
    
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
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a]$", fontsize=30)
ax.set_title(label="Dominio óptico", fontsize=30)
ax.legend(loc="best")
#ax.set_xlim([lim_inf_,lim_sup_])
#ax.set_ylim([0,1])

# Graficando la envolvente
ax = plt.subplot(2,2,3)
ax.plot(lambda_envolvente_superior,envolvente_superior,
           label="Envolvente superior", linewidth = 1.5) 
#ax.set_label("Envolvente")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a]$", fontsize=30)
ax.set_title(label="Dominio óptico", fontsize=30)
ax.legend(loc="best",fontsize=30)

#Creando caja de texto para mostrar los resultados en la imagen

# Concatenando los puntos maximos de la envolvente inferior
text = ""
for interseccion in intersecciones_superior: 
    text += "\nMinimo localizado en: %.3f mm" % interseccion
# Eliminando el espacio en blanco inicial
text = text[1:]

# Creando cadena para caja de texto
textstr = text 

# Estas son propiedades de matplotlib.patch.Patch
props = dict(boxstyle='round', facecolor='teal', alpha=0.5)

graph_text = ax.text(0.05, 0.22, textstr, transform=ax.transAxes, fontsize=35,
        verticalalignment='top', bbox=props)



# Graficando la envolvente
ax = plt.subplot(2,2,4)
ax.plot(lambda_envolvente_inferior,envolvente_inferior,
           label="Envolvente inferior", color="purple", linewidth = 1.5) 
#ax.set_label("Envolvente")
ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
ax.set_ylabel(ylabel=r"$[u.a]$", fontsize=30)
ax.set_title(label="Dominio óptico", fontsize=30)
#ax.set_ylim([-40,-10])
ax.legend(loc="best",fontsize=30)

#Creando caja de texto para mostrar resultados en la imagen

# Concatenando los puntos maximos de la envolvente inferior
text = ""
for interseccion in intersecciones_inferior: 
    text += "\nMaximo localizado en: %.3f mm" % interseccion
# Eliminando el espacio en blanco inicial
text = text[1:]

# Creando cadena para caja de texto
textstr = text 

# Estas son propiedades de matplotlib.patch.Patch
props = dict(boxstyle='round', facecolor='teal', alpha=0.5)

graph_text = ax.text(0.05, 0.22, textstr, transform=ax.transAxes, fontsize=35,
        verticalalignment='top', bbox=props)


# Guardando figura
plt.savefig(carpeta + "-" + nombre_archivo + ".png")
# Mostrando Figura
plt.show()
