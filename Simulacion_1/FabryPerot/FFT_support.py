#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:31:19 2021

@author: alejandro_goper

Scripts auxiliares para la correcta interpretacion y visualizacion de la FFT 

    Por I. Alejandro Gómez Pérez.
    
    La lógica de esta función es la siguiente:
    Toma unicamente los valores positivos de la fft (indices desde 0 hasta int(n/2)-1 [si n es par] 
    o desde 0 hasta int(n/2) [si n es impar]) duplica los valores de todo este arreglo (para así
    tomar en cuenta los valores negativos de las frecuencias) y finalmente dividimos entre el numero
    total de datos de la fft para encontrar así el valor de la amplitud correcta.
    
    ** Para construir el vector se uso una propiedad llamada slicing de los arreglos de python.
"""


from numpy import floor

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