#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:57:43 2021

@author: alejandro_goper


Metodos y clases para la aplicacion de filtros FIR mediante ventanas

"""

from numpy import pi, arange, sin, exp, cos, convolve

"""
Definicion de la clase principal
"""
class Filtro():
    """
    =================================
    Definicion de atributos de clase
    =================================
    """
    def __init__(self, _senal, _T_muestreo, _frec_corte, _orden):
        self.senal = _senal
        self.T_muestreo = _T_muestreo
        # Se hace un reescalamiento para ajustar a nuestro modelo
        self.frec_corte = _frec_corte*pi*(2*_T_muestreo)
        self.orden = _orden
    """
    ====================
    Metodos de la clase
    ====================
    """
    
    # Antitransformada de un filtro ideal pasa-bajos
    
    def h_n(self,orden=21):       
        # Parametros del filtro
        N = orden # Orden del filtro
        M = N-1 
        w_c = self.frec_corte  # frecuencia de corte
        # Dominio de la secuencia h[n]
        n = arange(0,N)
        # Calculando los coeficientes b[n] de la respuesta de un filtro pasa bajos ideal
        # Es un seno cardinal
        _h_n = sin(w_c*(n-M/2))/(pi*(n-M/2))
        # Agregando la contribucion central del seno cardinal (dado que no esta definida aun)
        _h_n[int(M/2)] = w_c/pi
        return _h_n

    # Definicion de una ventana gaussiana centrada en (N-1)/2 muestras donde N es el orden del filtro
    def ventana_de_gauss(self, sigma):
        M = self.orden -1
        n = arange(0,M+1)
        w_n = exp(-0.5*((2*n-M)/(sigma*M))**2)
        return w_n
    
    # Definicion de la ventana de hanning centrada en (N-1)/2 muestras
    def ventana_hanning(self):
        M = self.orden -1
        n = arange(0,M+1)
        w_n = 0.5 - 0.5*cos(2*pi*n/M)
        return w_n
    
    # sigma < 0.5
    def filtrar_por_ventana_de_gauss(self,sigma=0.25):
        
        senal = self.senal
        
        w_n = self.ventana_de_gauss(sigma)
        # filtro ideal
        h_n = self.h_n(self.orden)
        # Definicion del filtro, truncamiento de h_n en la ventana de w_n
        s_n = w_n*h_n
        # Normalizando para que la suma sea 1, para que se respete la amplitud
        s_n /= sum(s_n)
        
        # senal filtrada
        senal_filtrada = convolve(s_n,senal, mode="same")
        return senal_filtrada
        
    def filtrar_por_ventana_de_hanning(self):
        senal = self.senal
        
        w_n = self.ventana_de_hanning()
        # filtro ideal
        h_n = self.h_n(self.orden)
        # Definicion del filtro, truncamiento de h_n en la ventana de w_n
        s_n = w_n*h_n
        # Normalizando para que la suma sea 1, para que se respete la amplitud
        s_n /= sum(s_n)
        
        # senal filtrada
        senal_filtrada = convolve(s_n,senal, mode="same")
        return senal_filtrada
    