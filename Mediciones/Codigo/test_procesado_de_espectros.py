#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:28:54 2021

@author: alejandro_goper


Script para pruebas de procesado de espectros obtenidos en el laboratorio


Documentacion para los colores de las graficas:
    - https://matplotlib.org/stable/tutorials/colors/colors.html


"""

from FabryPerot.Filtros_support import Filtro, ventana_de_gauss, ventana_de_hanning, ventana_flattop, ventana_kaiser_bessel
from FabryPerot.FFT_support import encontrar_FFT_dominio_en_OPL
from scipy.signal import find_peaks
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

"""
==============================================================================
Importando Datos:
    Vamos a dar como input el path hacia una carpeta que contendra todos los
    datos en archivos .dat, nombrados como Espectro (1).dat, etc, y un archivo
    extra denominado Referencia.dat
==============================================================================
"""




# Definiendo ruta a la carpeta de las mediciones 
ruta_directorio = "../" + "04-03-2022" +"/" + "Efecto Vernier" +  "/" + "Aire-Glicerina" + "/" + "100um" + "/"


# Con esta instucción encontramos una lista de todos los archivos .dat en el 
# directorio proporcionado
lista = Path(ruta_directorio).glob("**/*.dat")

path = ruta_directorio + "Referencia.dat"

# Cargando datos
referencia = np.loadtxt(path, skiprows = 58)
# Separando datos
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

        print("Procesando: ", archivo.name[:-4])

        """
        ======================================================================
        Normalizando respecto a la referencia en escala Logaritmica 
        ======================================================================
        """    
        potencia_dB = potencia_dBm  - potencia_dBm_ref

        """
        ======================================================================
        Definicion de parametros
        ======================================================================
        """
        # Definiendo limite de busqueda en el espectro de Fourier 
        # (OPL en milimetros)
        lim_inf = 0 # mm 
        lim_sup = 10 # mm
        #Periodo de muestreo = (lambda_[-1] - lambda_[0])/len(lambda_) Approx 0.005 nm
        lambda_inicial = lambda_[0] # Valor inicial del arreglo
        lambda_final = lambda_[-1] # Valor final del arreglo
        n = len(lambda_) # Cantidad de datos en el arreglo
        T_muestreo_lambda = (lambda_final-lambda_inicial)/n
       

        """
        ======================================================================
        Calculando la FFT
        ======================================================================
        """

        opl,amp = encontrar_FFT_dominio_en_OPL(lambda_inicial=lambda_inicial, 
                                               lambda_final=lambda_final, 
                                               senal=potencia_dB)


        """
        ======================================================================
        Estableciendo rango de busqueda en el espacio de Fourier  
        ======================================================================
        """

        # Implementaremos un KNN de Machine Learning para encontrar el vecino 
        # mas cercano al limite de busqueda 
        """
        **********************************************************************
        Se realiza este procedimiento porque, en general, el lim_inf y lim_sup
        no son multiplos enteros de el periodo de muestreo (T_muestreo_lambda) 
        y por lo tanto, al tratar de buscar el indice en el arreglo del OPL, 
        en el dominio de Fourier, no se encontrara el valor exacto y esto 
        devuelve un valor Null
        **********************************************************************
        """

        # Creamos objeto de la clase NearestNeighbors 
        # Como solo ocupamos el vecino mas cercano entonces el parametro n_neighbors=1
        nn = NearestNeighbors(n_neighbors=1)

        # Encontrando el vecino mas cercano a los limites de busqueda en opl
        """
        **********************************************************************
        Dado que el metodo de Machine Learning solo funciona para datos 
        bidimensionales entonces transformamos el array del OPL en una matriz
        **********************************************************************
        """
        nn.fit(opl.reshape((len(opl),1)))

        """
        **********************************************************************
        Identificamos el indice en el arreglo OPL del vecino mas cercano al 
        valor del limite inferior y superior con el uso del metodo kneighbors
        **********************************************************************
        """
        index_lim_inf = nn.kneighbors([[lim_inf]], 1, return_distance=False)[0,0]
        index_lim_sup = nn.kneighbors([[lim_sup]], 1, return_distance=False)[0,0]
        
        """
        **********************************************************************
        Una vez obtenido el indice ya podemos identificar de que valor se 
        trata en el array del OPL y por tanto tambien de AMP
        **********************************************************************
        """
        lim_inf_ = opl[index_lim_inf]
        lim_sup_ = opl[index_lim_sup]

    
        """
        ======================================================================
        Aplicacion de filtro FIR pasa-bajos de fase lineal 
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


        # Creando objeto de la clase Filtro
        filtro = Filtro(_senal=potencia_dB, # senal a filtrar
                        _T_muestreo=T_muestreo_beta_opl, # Periodo de muestreo
                        _frec_corte=lim_sup, # Frecuencia de corte en unidades de T_muestreo
                        _orden=901) # Orden del Filtro

        # Filtrando por el metodo de las ventanas
        senal_filtrada = filtro.filtrar_por_ventana_de_gauss(sigma=0.2)
        """
        ======================================================================
        Aplicando FFT a la señal filtrada
        ======================================================================
        """

        opl_filt,amp_filt = encontrar_FFT_dominio_en_OPL(lambda_inicial=lambda_inicial, 
                                                         lambda_final=lambda_final, 
                                                         senal=senal_filtrada)






        """
        ======================================================================
        Cambiando la señal filtrada a escala Lineal
        ======================================================================
        """
        
        """
        **********************************************************************
         La normalizacion se hace escalando la referencia por un factor que
         hace que el 4% de refleccion obtenido en el espectro, sea, ahora,
         considerado como el 100%, por lo tanto debemos dividir la potencia
         en escala lineal por un factor de 25.
        **********************************************************************
        """
        # Cambiando a escala lineal

        senal_filtrada_esc_lineal = 10**(senal_filtrada/10)
        senal_filtrada_esc_lineal /= 25

        """
        ======================================================================
        Aplicando tecnica WINDOWING:
            Esta tecnica ayuda a mejorar la definicion de los maximos 
            en el espectro de Fourier
        ======================================================================
        """

        # Construyendo una ventana w_n del mismo tamaño que el array de la senal

        # w_n = ventana_de_gauss(orden=len(senal_filtrada_esc_lineal), sigma=0.08)
        w_n = ventana_de_hanning(orden=len(senal_filtrada_esc_lineal))
        # w_n = ventana_flattop(orden=len(senal_filtrada_esc_lineal))

        """
        **********************************************************************
        La ventana de Keiser-Bessel es similar a otras ventanas para distintos 
        valores del parametro beta,  por ejemplo:
            - beta = 0 - Ventana cuadrada
            - beta = 5 - Ventana de Hamming 
            - beta = 6 - Ventana de Hanning
            - beta = 8.6 - Ventana de Blackman - Harris
        **********************************************************************
        """
        # w_n = ventana_kaiser_bessel(orden=len(senal_filtrada_esc_lineal), beta=6)
        # Enventanado de la senal en escala lineal
        senal_enventanada = senal_filtrada_esc_lineal * w_n



        """
        ======================================================================
        Mejoramiento de la resolucion en Fourier post-windowing
        ======================================================================
        """

        # Mejorando la resolucion del espectro añadiendo 0 a los extremos del 
        # array

        # Numero de ceros a agregar en cada extremo
        n_zeros = 10000

        """
        **********************************************************************
        Empiricamente se ha determinado que cuando n_zeros > 10 000 entonces
        el espectro se desplaza por lo que se sugiere usar 0 < n_zeros < 1000
        **********************************************************************
        """
        zeros = list(np.zeros(n_zeros))

        # Agregamos los ceros a cada extremo de la señal
        senal_enventanada = zeros + list(senal_enventanada)
        senal_enventanada = np.array(senal_enventanada + zeros)

        # Agregando las correspondientes longitudes de onda (virtuales) 
        # correspondientes a los ceros añadidos a los extremos 
        lambda_mejorada = np.arange(1510-n_zeros*T_muestreo_lambda, 
                                    1590 + n_zeros*T_muestreo_lambda-0.00001,
                                    T_muestreo_lambda) 

        # Le agregamos un -0.00001 al final del segun parametro para asegurar 
        # que la longitud de lambda_mejorada sea la misma que la de la senal 
        # enventanada dado que si se lo quitamos en ocaciones la longitud 
        # difiere por un valor  

        """
        ======================================================================
        Aplicando FFT a la señal mejorada
        ======================================================================
        """

        # Dado que hemos extendido el dominio en lambda calculamos de nuevo
        lambda_inicial = lambda_mejorada[0]
        lambda_final = lambda_mejorada[-1]

        # Calculando la FFT de la señal enventanada
        opl_env, amp_env = encontrar_FFT_dominio_en_OPL(lambda_inicial=lambda_inicial, 
                                                        lambda_final=lambda_final, 
                                                        senal=senal_enventanada)

        """
        ======================================================================
        Eliminando componente de DC a la señal mejorada
        ======================================================================
        """

        # Eliminando la componenete de DC hasta un margen fijo en el opl
        dc_margen = 0.1 # mm

        # buscamos el indice en el array opl mas cercano a dc_margen
        nn.fit(opl_env.reshape((len(opl_env),1)))
        index_dc_margen = nn.kneighbors([[dc_margen]], 1, return_distance=False)[0,0]
        # eliminamos todas las contribuciones del espectro de fourier hasta dc_margen
        amp_env[:index_dc_margen] = np.zeros(index_dc_margen)


        """
        ======================================================================
        Encontrando los valores maximos
        ======================================================================
        """

        # La funcion de find_peaks de Scipy ya tiene la capacidad de realizar esto
        
        # Encontrando el vecino mas cercano a los limites de busqueda en opl_env

        index_lim_inf = nn.kneighbors([[lim_inf]], 1, return_distance=False)[0,0]
        index_lim_sup = nn.kneighbors([[lim_sup]], 1, return_distance=False)[0,0]

        # Limitamos la busqueda
        amp_env_temp = amp_env[index_lim_inf:index_lim_sup]

        # Necesitamos definir un valor limite en altura en el grafico de la amplitud
        # se buscaran los maximos que superen este valor
        lim_amp = 0.002

        # Buscando maximos en la region limitada
        picos, _ = find_peaks(amp_env_temp, height = lim_amp)

        # Como limitamos la busqueda hay que compensar con los indices anteriores 
        # en el array para obtener el valor verdadero 
        picos += index_lim_inf 

        maximos = opl_env[picos]
    
        # Imprimiendo resultados
        print("==============================================================")
        for maximo in maximos: 
            print("Maximo localizado en: %.3f mm" % maximo)
            
        print("==============================================================")


        """
        ======================================================================
        Graficando resultados
        ======================================================================
        """

        # Creando figura
        fig,ax = plt.subplots(figsize=(40,40))
        fig.set_tight_layout(True)
        # Pone lo mas juntas las graficas posibles
        fig.set_tight_layout(True)
        # Para que no se empalmen los titulos en los ejes
        fig.subplots_adjust(wspace=1.2)
        
        # Cambiando el tamano de la fuente en todos los ejes
        plt.rcParams.update({'font.size': 20})
        
        # Graficando el espectro optico inicial
        ax = plt.subplot(4,2,1)
        espectro_graph, = ax.plot(lambda_,potencia_dB, linewidth=1.5, 
                                  label= "Medición Normalizada")
        ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
        ax.set_ylabel(ylabel=r"$dB$", fontsize=30)
        ax.set_title(label="Dominio óptico", fontsize=30)
        #ax.set_ylim([-40,-10])
        ax.legend(loc="best",fontsize=30)
        
        # Graficando la FFT del espectro inicial
        ax = plt.subplot(4,2,2)
        fft_graph, = ax.plot(opl,amp, linewidth=1.5,color="purple")
        ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
        ax.set_ylabel(ylabel=r"$|dB|$", fontsize=30)
        ax.set_title(label="Dominio de Fourier", fontsize=30)
        ax.set_xlim([lim_inf_,lim_sup_])
        #ax.set_ylim([0,1])
        
        # Graficando el espectro optico de la señal filtrada
        ax = plt.subplot(4,2,3)
        espectro_graph, = ax.plot(lambda_,senal_filtrada, linewidth=1.5, 
                                  label="Señal filtrada")
        ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
        ax.set_ylabel(ylabel=r"$dB$", fontsize=30)
        ax.set_title(label="Dominio óptico", fontsize=30)
        #ax.set_ylim([-40,-10])
        ax.legend(loc="lower left",fontsize=30)
        
        # Graficando la FFT de la señal filtrada
        ax = plt.subplot(4,2,4)
        fft_graph, = ax.plot(opl_filt,amp_filt, linewidth=1.5,color="teal")
        ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
        ax.set_ylabel(ylabel=r"$|dB|$", fontsize=30)
        ax.set_title(label="Dominio de Fourier", fontsize=30)
        ax.set_xlim([lim_inf_,lim_sup_])
        #ax.set_ylim([0,1])
        
        
        # Graficando el espectro optico de la señal tratada 
        ax = plt.subplot(4,2,5)
        espectro_graph, = ax.plot(lambda_, senal_filtrada_esc_lineal, 
                                  linewidth=1.5, label="Señal mejorada")
        ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
        ax.set_ylabel(ylabel=r"$[u.a.]$", fontsize=30)
        ax.set_title(label="Dominio óptico escala lineal", fontsize=30)
        #ax.set_ylim([-40,-10])
        ax.legend(loc="upper left",fontsize=30)
        
        # Graficando FFT de la señal tratada
        ax = plt.subplot(4,2,6)
        fft_graph, = ax.plot(opl_filt,amp_filt, linewidth=1.5,color="navy")
        ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
        ax.set_ylabel(ylabel=r"$|u.a|$", fontsize=30)
        ax.set_title(label="Dominio de Fourier", fontsize=30)
        ax.set_xlim([lim_inf_,lim_sup_])
        #ax.set_ylim([0,2])
    
        # Graficando el espectro optico de la señal tratada 
        ax = plt.subplot(4,2,7)
        espectro_graph, = ax.plot(lambda_mejorada, senal_enventanada, 
                                  linewidth=1.5, label="Señal mejorada")
        ax.set_xlabel(xlabel=r"$\lambda [nm]$", fontsize=30)
        ax.set_ylabel(ylabel=r"$[u.a.]$", fontsize=30)
        ax.set_title(label="Dominio óptico escala lineal", fontsize=30)
        #ax.set_ylim([-40,-10])
        ax.legend(loc="upper left",fontsize=30)
        
        # Graficando FFT de la señal tratada
        ax = plt.subplot(4,2,8)
        fft_graph, = ax.plot(opl_env,amp_env, linewidth=1.5,color="navy")
        ax.set_xlabel(xlabel=r"$OPL [mm]$", fontsize=30)
        ax.set_ylabel(ylabel=r"$|u.a|$", fontsize=30)
        ax.set_title(label="Dominio de Fourier", fontsize=30)
        ax.set_xlim([lim_inf_,lim_sup_])
        #ax.set_ylim([0,2])
        
        
        
        # Guardando figura
        plt.savefig(ruta_directorio + "-" + archivo.name[0:-4] + "_test.png")
        # Mostrando Figura
        plt.show()