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
    def __init__(self, lambda_inicial, lambda_final,T_muestreo_lambda, 
                 L_medio_1, L_medio_2, eta_medio_1, eta_medio_2, R_3=0.95, 
                 eta_fibra=1.48, alpha_medio_1=0, alpha_medio_2=0, 
                 A_interfaz_1=0, A_interfaz_2=0):
        
        # Array de longitudes de onda de la simulacion
        self.lambda_ = arange(lambda_inicial, lambda_final+T_muestreo_lambda, T_muestreo_lambda)
        
        # Indices de refraccion de los medios
        self.eta_0 = eta_fibra 
        self.eta_1 = eta_medio_1
        self.eta_2 = eta_medio_2
        
        # Longitudes de las cavidades del primer y segundo medio en mm.
        self.L1 = L_medio_1
        self.L2 = L_medio_2
        
        # Coeficientes de perdidas a traves de los medios
        self.alpha_1 = alpha_medio_1
        self.alpha_2 = alpha_medio_2
        
        # Coeficientes de perdidas en las interfaces
        self.A_1 = A_interfaz_1
        self.A_2 = A_interfaz_2

        # Reflectancia de la ultima interfaz
        self.R3 = R_3

    """
    Metodo Reflectancia:
        Evalua la reflectancia
    """
    def Reflectancia(self):
        
        xi_1 = self.xi(self.A_1, self.alpha_1)
        xi_2 = self.xi(self.A_2, self.alpha_2)

        R1 = self.R(self.eta_0, self.eta_1)
        R2 = self.R(self.eta_1, self.eta_2)
        R3 = self.R3
        
        phi_1 = self.phi(self.eta_1, self.L1,self.lambda_)
        phi_2 = self.phi(self.eta_2, self.L2,self.lambda_)
        
        theta_0 = self.theta(self.eta_0, self.eta_1)
        theta_1 = self.theta(self.eta_1, self.eta_2)
        theta_2 = pi
        
        reflectancia = ((2*xi_1*sqrt(R1*R2))*cos(2*phi_1-(theta_0-theta_1))*(1-R1)**2  + 
                        (2*xi_1*xi_2*sqrt(R2*R3))*cos(2*phi_2-(theta_1-theta_2))*((1-R1)*(1-R2))**2 + 
                        (2*xi_1*xi_2*sqrt(R1*R3))*cos(2*(phi_1+phi_2)-(theta_0-theta_2))*((1-R1)*(1-R2))**2 + 
                        (R1 + R2*xi_1*(1-R1)**2 + R3*xi_1*xi_2*((1-R1)*(1-R2))**2 ))
        return reflectancia 
    
    
    """
    Metodo theta:
        Determina el valor de la fase cuando la luz se refleja en la interfaz
    """
    def theta(self, n_k, n_k1):
        if(n_k>n_k1):
            return 0
        else:
            return pi
    
    """
    Metodo phi:
        Determina el desplazamiento de fase cuando la luz atraviesa el medio 
    """
    def phi(self,eta, L,lambda_):
        phi = 2*pi*eta*L*(1*10**6) / lambda_
        return phi
    
    """
    Metodo R:
        Determina la reflectancia de la interfaz
    """
    def R(self, eta1, eta0):
        r = (eta1 - eta0)/(eta1+eta0)
        return r*r
    
    """
    Metodo xi:
        Calcula uno de los coeficientes auxiliares relativos a las perdidas 
    """
    def xi(self, A, alpha):
        p = (1-A)*(1-alpha)
        return p*p
