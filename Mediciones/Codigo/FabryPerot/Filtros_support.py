#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 12:57:43 2021

@author: alejandro_goper


Metodos y clases para la aplicacion de filtros FIR mediante ventanas

Referencias:
    
    [1] http://www.dicis.ugto.mx/profesores/arturogp/documentos/Filtrado%20Digital/Lectura%203_Filtrado_Digital.pdf
    [2] http://profesores.elo.utfsm.cl/~mzanartu/IPD414/Docs/ipd414-c05c.pdf
    [3] https://ipython-books.github.io/102-applying-a-linear-filter-to-a-digital-signal/
    [4] https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
    [5] https://www.youtube.com/watch?v=QWUOBlAABQU

"""

from numpy import pi, arange, sin, exp, cos, convolve

"""

Definicion de la clase principal para la aplicacion de filtros

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
    
    """
    ==========================================================================
    Este metodo es para construir la antitransformada de Fourier de un filtro 
    pasa bajos ideal (una funcion escalon) desplazado para que la secuencia 
    sea causal.
    ==========================================================================
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

    """
    ==========================================================================
    Este metodo construye una ventana gaussiana centrada en (N-1)/2 muestras
    donde N es el orden del filtro (importante que N sea impar)
    
    Requiere:
        - sigma: desviacion estandar de la distribucion gaussiana
            segun la referencia [2] sigma debe ser menor a 0.5.
    ==========================================================================
    """
    def ventana_de_gauss(self, sigma):
        M = self.orden -1
        n = arange(0,M+1)
        w_n = exp(-0.5*((2*n-M)/(sigma*M))**2)
        return w_n
    
    """
    ==========================================================================
    Este metodo construye una ventana de hanning centrada en (N-1)/2 muestras
    donde N es el orden del filtro (importante que N sea impar)
    ==========================================================================
    """
    def ventana_de_hanning(self):
        M = self.orden -1
        n = arange(0,M+1)
        w_n = 0.5 - 0.5*cos(2*pi*n/M)
        return w_n
    
    
    """
    ==========================================================================
    Este metodo aplica un filtro pasabajos por el metodo de las ventanas. 
    Basicamente, usamos la secuencia h_n (la antitransformada del filtro ideal,
    definida de -inf a inf) y la multiplicamos por la ventana gaussiana w_n 
    para truncar la secuencia infinita h_n y tener una secuencia finita s_n.
    
    Luego debemos aplicar el filtro usando la convolucion discreta entre las
    secuencias s_n y la secuencia de la senal original. (el resultado es el
    mismo que si usamos la funcion de scipy -- filtfilt).
    
    Segun la referencia [4] se debe normalizar la ventana respesto a la suma 
    de todas sus contribuciones para preservar la amplitud de la senal original
    ==========================================================================
    """
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
        
    """
    ==========================================================================
    Este metodo aplica un filtro pasabajos por el metodo de las ventanas. 
    Basicamente, usamos la secuencia h_n (la antitransformada del filtro ideal,
    definida de -inf a inf) y la multiplicamos por la ventana gaussiana w_n 
    para truncar la secuencia infinita h_n y tener una secuencia finita s_n.
    
    Luego debemos aplicar el filtro usando la convolucion discreta entre las
    secuencias s_n y la secuencia de la senal original. (el resultado es el
    mismo que si usamos la funcion de scipy -- filtfilt).
    
    Segun la referencia [4] se debe normalizar la ventana respesto a la suma 
    de todas sus contribuciones para preservar la amplitud de la senal original
    ==========================================================================
    """
    
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
    