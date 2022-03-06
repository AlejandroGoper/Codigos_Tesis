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
from pathlib import Path

"""
==============================================================================
Importando Datos:
    Vamos a dar como input el path hacia una carpeta que contendra todos los
    datos en archivos .dat, nombrados como Espectro (1).dat, etc, y un archivo
    extra denominado Referencia.dat
==============================================================================
"""

# Importando archivos 


ruta_directorio = "../" + "Mediciones_Monse" + "/" + "4xLcav" + "/" 


# Con esta instucción encontramos una lista de todos los archivos txt en el 
# directorio proporcionado
lista = Path(ruta_directorio).glob("**/*.dat")

path = ruta_directorio + "/Referencia.dat"

referencia = np.loadtxt(path, skiprows = 58)
lambda_ref, potencia_dBm_ref = referencia[:,0], referencia[:,1]

# Para cada uno de los archivos vamos a aplicar el programa
for archivo in lista:
    if(archivo.name == "Referencia.dat"):
        continue
    else:
        data = np.loadtxt(archivo, skiprows=58)
        # Separando columnas del archivo en arreglos individuales        
        lambda_, potencia_dBm = data[:,0], data[:,1]
        # Damos por hecho que lambda_ = lambda_ref

        """
        ======================================================================
        Normalizando respecto a la referencia en escala Logaritmica
        ======================================================================
        """    
        potencia_dB = potencia_dBm  - potencia_dBm_ref

        """
        ======================================================================
        Cambiando a escala Lineal
        ======================================================================
        """     
        potencia = 10**(potencia_dB/10)

        """
        ======================================================================
        Definicion de parametros utiles 
        ======================================================================
        """

        lambda_inicial = lambda_[0] # Valor inicial del arreglo
        lambda_final = lambda_[-1] # Valor final del arreglo
        n = len(lambda_) # Cantidad de datos en el arreglo
        # Periodo de muestreo en variable lambda
        T_muestreo_lambda = (lambda_final-lambda_inicial)/n
       

        """
        ======================================================================
        Aplicacion de filtro FIR pasa-bajos de fase lineal:
            Se realiza un cambio de variable y un reescalamiento para que la 
            frecuencia de corte se encuentre en unidades de mm, igual que el 
            OPL y sea mas facil identificar y eliminar las contribuciones.
        ======================================================================
        """
        
        # Al realizar el cambio de variable beta = 1/lambda, tenemos que 
        T_muestreo_beta = (1/lambda_inicial - 1/lambda_final)/n

        # Multiplicamos todo el array por un factor de 2x10**6

        """
        **********************************************************************
        Se multiplica por este factor para que las unidades de T_muestreo 
        esten en milimetros y reflejen el valor del OPL, una explicacion más 
        detallada se encuentra en el archivo FFT_support 
        **********************************************************************
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
        ======================================================================
        Encontrando las envolventes superior e inferior de todo el espectro:
            Se encuentran los puntos maximos y minimos de todo el espectro
        ======================================================================
        """
        # Condicion que asegura que siempre busquemos en un rango valido los
        # puntos maximos
        lim_amplitud = senal_filtrada.min()
        # Buscando picos por encima del valor minimo de la señal
        indices_picos, _ = find_peaks(senal_filtrada, height = lim_amplitud)
        
        # Mapeando los indices de los maximos en el dominio y rango de la señal
        envolvente_superior = senal_filtrada[indices_picos]
        lambda_envolvente_superior = lambda_[indices_picos]
        
        
        # Para encontrar los minimos multiplicamos la senal original por -1
        senal_invertida = - senal_filtrada
        lim_amplitud = senal_invertida.min()
        indices_picos, _ = find_peaks(senal_invertida,height= lim_amplitud)

        envolvente_inferior = senal_filtrada[indices_picos]
        lambda_envolvente_inferior = lambda_[indices_picos]
        

        """
        ==============================================================================
        Empiricamente se ha descubierto que se puede rastrear, al menos, una 
        señal en un rango limitado ya sea del arreglo que contiene los puntos 
        maximos o del de los puntos minimos.
        
        En esta seccion, se limita el rango de una de estas dos señales para
        obtener la señal rastreable.
        ==============================================================================
        """

        lim_inferior = 0.1  # [u.a]
        lim_superior = 0.5  # [u.a]
        # Umbral para dividir la señal en dos (superior e inferior)
        prom = (lim_inferior + lim_superior)/2

        # Encontrando la señal dentro de los limites establecidos
        indices_senal_a_seguir = np.where((envolvente_inferior<lim_superior) & 
                                  (envolvente_inferior >= lim_inferior))
        senal_a_seguir = envolvente_inferior[indices_senal_a_seguir]
        lambda_senal_a_seguir = lambda_envolvente_inferior[indices_senal_a_seguir] 

        # Dividiento la señal en dos: 
        
        # Parte superior de la señal 
        index_env_sup_senal_rastrable = np.where(senal_a_seguir >= prom) 
        env_sup_senal_rastreable = senal_a_seguir[index_env_sup_senal_rastrable]
        lambda_env_sup_senal_rastreable = lambda_senal_a_seguir[index_env_sup_senal_rastrable]
        
        # Parte inferior de la señal
        index_env_inf_senal_rastrable = np.where(senal_a_seguir < prom) 
        env_inf_senal_rastreable = senal_a_seguir[index_env_inf_senal_rastrable]
        lambda_env_inf_senal_rastreable = lambda_senal_a_seguir[index_env_inf_senal_rastrable]


        """
        ======================================================================
        Encontrando los puntos de interseccion:
            Se tomaran los puntos maximos de la envolvente inferior y los
            minimos de la envolvente superior y luego realizamos un promedio 
            de ambos valores
            
        * Para relacionar los maximos con los minimos correspondientes se 
        realiza la comparacion punto a punto entre la envolvente inferior y 
        superior, ambos puntos son interseccion si la distancia entre ellos es 
        menor a una tolerancia definida en nm 
        =======================================================================
        """
        # Encontrando maximos de la envolvente inferior    
        lim_amplitud = env_inf_senal_rastreable.min()

        # Buscando picos por encima del valor minimo de la señal
        indices_picos, _ = find_peaks(env_inf_senal_rastreable, 
                                      height = lim_amplitud)
        
        # Arreglo temporal para almacenar los maximos de la envolvente inferior
        intersecciones_inferior=lambda_env_inf_senal_rastreable[indices_picos]

        # Encontrando minimos de la envolvente superior   
        lim_amplitud = -env_sup_senal_rastreable.max()
        # Buscando picos por encima del vaor minimo
        indices_picos, _ = find_peaks(-env_sup_senal_rastreable, 
                                      height = lim_amplitud)
        # Arreglo temporal para almacenar los minimos de la envolvente superior
        intersecciones_superior = lambda_env_sup_senal_rastreable[indices_picos]
    

        # Lista para almacenar las intersecciones
        intersecciones = []
        
        # Tolerancia en unidades de lambda
        tol = 10 #nm

        # Para cada punto del arreglo de intersecciones superior 
        for punto in intersecciones_superior:    
            # calculamos la distancia a todos los puntos del arreglo de 
            # intersecciones inferior
            distancia = np.abs(punto - intersecciones_inferior)
            # Buscamos los indices de los puntos cuya distancia sea menor que
            # la tolerancia
            index =np.where(distancia < tol)
            # Primero verificamos si el arreglo index no esta vacio 
            if(np.size(index)):
                # Si no esta vacio se realiza el promedio entre los puntos
                # encontrados
                # Inciamos la sumatoria de los puntos
                sumatoria = punto
                # Encontramos los puntos cercanos al punto pivote 
                puntos_cercanos = intersecciones_inferior[index]
                # variable que controla el numero total de puntos
                # El +1 es debido a que se cuenta el punto pivote
                m = len(puntos_cercanos) + 1
                # Realizamos la sumatoria de los puntos
                for p in puntos_cercanos:
                    sumatoria += p
                    # La interseccion es el promedio de los puntos cercanos
                    interseccion = sumatoria/m
                    # Agregamos este valor al arreglo de intersecciones
                    intersecciones.append(interseccion)
                    
        # Convirtiendo la lista de intersecciones a array numpy
        np.array(intersecciones)
    

        """
        ======================================================================
        Graficando resultados y guardando 
        ======================================================================
        """

        # Creando figura
        fig, ax = plt.subplots(figsize=(40,20))
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
        
        graph_text = ax.text(0.05, 0.25, textstr, transform=ax.transAxes, 
                             fontsize=35, verticalalignment='top', 
                             bbox=props, color="black")
        
        # Guardando figura
        
        plt.savefig(ruta_directorio + "-" + archivo.name + ".png")
        # Mostrando Figura
        
        plt.show()
        
        """
        Guardando datos de las envolventes superior e inferior.
        """
        """
        
        x_ = lambda_env_sup_senal_rastreable.reshape(len(lambda_env_sup_senal_rastreable),1)
        xy = np.append(x_, env_sup_senal_rastreable.reshape(len(env_sup_senal_rastreable),1), axis=1)
        
        np.savetxt(ruta_directorio + "/" + "Envolvente_superior_" + archivo.name, xy, fmt="%f" , header="Lambda, Potencia_escala_lineal", delimiter = ",")
        
        x = lambda_env_inf_senal_rastreable.reshape(len(lambda_env_inf_senal_rastreable),1)
        xy = np.append(x, env_inf_senal_rastreable.reshape(len(env_inf_senal_rastreable),1), axis=1)
        
        np.savetxt(ruta_directorio + "/" + "Envolvente_inferior_" + archivo.name, xy, fmt="%f" , header="Lambda, Potencia_escala_lineal", delimiter = ",")
        """    