#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 14:43:39 2021

@author: alejandro_goper
"""

from numpy import pi,sqrt,cos,arange

class FabryPerot_2GAP:
    """
    Clase para la simulacion de Fabry-Perot
    """
    
    """
    Metodo constructor:
        
        Definicion de Atributos de clase
    
    """
    def __init__(self, lambda_inicial, lambda_final, L_medio_1, L_medio_2, eta_medio_1, eta_medio_2, 
                 eta_medio_3, eta_fibra=1.48, alpha_medio_1=0, alpha_medio_2=0, 
                 A_interfaz_1=0, A_interfaz_2=0):
        # Array de longitudes de onda de la simulacion
        self.lambda_ = arange(lambda_inicial, lambda_final, 0.005)
        
        # Indices de refraccion de los medios
        self.eta_1 = eta_fibra 
        self.eta_0 = eta_medio_1
        self.eta_2 = eta_medio_2
        self.eta_3 = eta_medio_3
        
        # Longitudes de las cavidades del primer y segundo medio en mm.
        self.L1 = L_medio_1
        self.L2 = L_medio_2
        
        # Coeficientes de perdidas a traves de los medios
        self.alpha_1 = alpha_medio_1
        self.alpha_2 = alpha_medio_2
        
        # Coeficientes de perdidas en las interfaces
        self.A_1 = A_interfaz_1
        self.A_2 = A_interfaz_2

    """
    Metodo R:
        Evalua la reflectancia
    """
    def R(self):
        
        s1_2 = self.S1_2()
        s2_2 = self.S2_2()
        R1 = self.R1()
        R2 = self.R2()
        R3 = self.R3()
        phi_1 = self.phi_1(self.lambda_)
        phi_2 = self.phi_2(self.lambda_)
        theta_1 = self.theta_1()
        theta_2 = self.theta_2()
        theta_3 = self.theta_3()
        reflectancia = (2*s1_2*sqrt(R1*R2))*cos(2*phi_1-(theta_1-theta_2)) + (2*s1_2*s2_2*sqrt(R2*R3))*cos(2*phi_2-(theta_2-theta_3)) + (2*s1_2*s2_2*sqrt(R1*R3))*cos(2*(phi_1+phi_2)-(theta_1-theta_3)) + (R1 + s1_2*R2 + s1_2*s2_2*R3) 
        return reflectancia 
    
    
    """
    Metodo theta_1:
        Determina el valor de la fase cuando la luz se refleja en la interfaz 1
    """
    def theta_1(self):
        if(self.eta_1>self.eta_0):
            return 0
        else:
            return pi
    
    """
    Metodo theta_2:
        Determina el valor de la fase cuando la luz se refleja en la interfaz 2
    """
    def theta_2(self):
        if(self.eta_0 > self.eta_2):
            return 0
        else:
            return pi
    
    """
    Metodo theta_3:
        Determina el valor de la fase cuando la luz se refleja en la interfaz 3
    """
    def theta_3(self):
        if(self.eta_2 > self.eta_3):
            return 0
        else:
            return pi
    
    """
    Metodo phi_1:
        Determina el desplazamiento de fase cuando la luz atraviesa el medio 1
        El factor de 1x10**6 es para el ajuste de unidades de mm y nm 
    """
    def phi_1(self,lambda_):
        phi = 2*pi*self.eta_0*self.L1*(1*10**6) / lambda_
        return phi
    
    """
    Metodo phi_2:
        Determina el desplazamiento de fase cuando la luz atraviesa el medio 2
    """
    def phi_2(self,lambda_):
        phi = 2*pi*self.eta_2*self.L2*(1*10**6) / lambda_
        return phi
    
    """
    Metodo R1:
        Determina la reflectancia de la primera interfaz
    """
    def R1(self):
        r = (self.eta_0 - self.eta_1)/(self.eta_1+self.eta_0)
        return r*r
    
    """
    Metodo R2:
        Determina la reflectancia de la segunda interfaz
    """
    def R2(self):
        r = (self.eta_2 - self.eta_0)/(self.eta_2+self.eta_0)
        return r*r
    
    """
    Metodo R3:
        Determina la reflectancia de la segunda interfaz
    """
    def R3(self):
        r = (self.eta_3 - self.eta_2)/(self.eta_2+self.eta_3)
        return r*r
    
    """
    Metodo S1_2:
        Calcula uno de los coeficientes auxiliares
    """
    def S1_2(self):
        p = (1-self.A_1)*(1-self.R1())*(1-self.alpha_1)
        return p*p
    
    """
    Metodo S2:
        Calcula uno de los coeficientes auxiliares
    """
    def S2_2(self):
        p = (1-self.A_2)*(1-self.R2())*(1-self.alpha_2)
        return p*p