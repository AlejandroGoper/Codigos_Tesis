#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:31:19 2021

@author: alejandro_goper


Scripts auxiliares para la correcta interpretacion y visualizacion de la FFT 

    Por I. Alejandro Gómez Pérez.
    
"""


from numpy import floor 
from numpy.fft import fft,fftfreq
"""
 La lógica de esta función es la siguiente:
    Toma unicamente los valores positivos de la fft (indices desde 0 hasta int(n/2)-1 [si n es par] 
    o desde 0 hasta int(n/2) [si n es impar]) 
    
    ** Para construir el vector se uso una propiedad llamada slicing de los arreglos de python.
"""

def recorte_frec_negativas_fft(fft):
    n = len(fft)
    if(n%2 == 0):
        lim = floor(n/2)
    else:
        lim = floor(n/2)+1
        
    vector = fft[0:int(lim)]
    return vector

def calcular_verdadera_amplitud(fft):
    n = len(fft)
    fft_ = recorte_frec_negativas_fft(fft)
    # Basicamente, tomamos solo la parte positiva del espectro de magnitud 
    # duplicando el valor de cada contribución supliendo la parte negativa
    magnitud = 2*abs(fft_)/n
    # Esto por que dado que la frecuencia 0 no tiene una contribucion negativa,
    # por lo que tiene el doble 
    magnitud[0] /= 2 #
    return magnitud

"""
Esta funcion calcula el espectro de amplitud normalizado respecto al numero de datos
y con la contribucion correcta de cada una de las frecuencias positivas de la transformada de Fourier.

Devuelve:
    vfreq_positivas: array con las frecuencias positivas del espectro (eje x)
    magnitud_fft: array con la magnitud de la fft 
"""

def encontrar_FFT(lambda_, Reflectancia):
    # Al realizar el cambio de variable beta = 1/lambda, tenemos que 
    #T_muestreo_beta = T_muestreo_lambda / (lambda_inicial*(lambda_inicial+T_muestreo_lambda))
    T_muestreo_beta = (1/lambda_[0] - 1/lambda_[int(len(lambda_))-1])/len(lambda_)
    # Encontramos la FFT de la reflectancia
    fft_reflectancia = fft(Reflectancia)
    """
    Esta funcion calcula el vector de "frecuencias" de la transformada de fourier, vease
    
    https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
    
    Es el analogo a la funcion vfreq realizada en mi github: 
        
        https://github.com/AlejandroGoper/Fundamento_de_Procesamiento_Digital_de_Senales/blob/main/Tarea5/Codigo/Tarea5.ipynb
        
        Es importante notar el escalamiento de los ejes, dado que en esta nueva variable beta
        el espaciado entre cada frecuencia es del orden de 10**8 nanometros, por lo que debemos
        dividir el vector de frecuencias por un factor de 10**6 para que convierta los nanometros 
        a milimetros, ademas dado que en general tenemos 
        
        x(beta) = cos[2pi * (2OPL) * beta] donde beta es la variable independiente beta = 1/lambda
        
        en el espectro de fourier los picos de "frecuencias" estaran ubicados en +-2OPL 
        
        si queremos que cada pico de frecuencia diga el OPL directo, debemos agregar un factor de
        1/2 adicional al vector de frecuencias
        
        Todo esto podemos realizarlo multiplicando T_muestreo_beta*(2*10**6) 
    """    
    vfreq = fftfreq(len(fft_reflectancia),(T_muestreo_beta)*((2*10**6)))    
    magnitud_fft = calcular_verdadera_amplitud(fft=fft_reflectancia)
    vfreq_positivas = recorte_frec_negativas_fft(fft=vfreq)
    return vfreq_positivas, magnitud_fft
