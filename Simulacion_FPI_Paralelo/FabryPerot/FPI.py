#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  14 0013:39 2022

@author: alejandro_goper
"""

from numpy import pi,sqrt,cos,arange,exp

"""
==============================================================================
Clase para la simulacion del modelo de un interferometro fabry-perot de dos
cavidades en paralelo.
==============================================================================
"""

class FPI_2GAP_parallel:
    
    """
    ==========================================================================
    Metodo constructor:
        Definicion de Atributos de clase
    ==========================================================================
    """
    def __init__(self, lambda_inicial, lambda_final, T_muestreo_lambda, 
                 L_i1, # Longitud de cavidades del primer interferometro 
                 L_i2, # Longitud de cavidades del segundo interferometro
                 n_i1, # Indices de refraccion del primer interferometro
                 n_i2, # Indices de refraccion del segundo interferometro
                 alpha_i1, # Perdidas del primer interferometro (medio)
                 alpha_i2, # Perdidas del segundo interferometro (medio)
                 A_i1, #  Perdidas del primer interferometro (interfaz)
                 A_i2, # Perdidas del segundo interferometro (interfaz)
                 ):
        
        # Array de longitudes de onda de la simulacion
        self.lambda_ = arange(lambda_inicial, lambda_final, T_muestreo_lambda)
        
        # Indices de refraccion de los medios
        self.n_01, self.n_11, self.n_21, self.n_31= n_i1[:]   # FPI 1 
        self.n_02, self.n_12, self.n_22, self.n_32= n_i2[:]   # FPI 2
        
        # Longitudes de las cavidades del primer y segundo medio en mm.
        self.L_11, self.L_21 = L_i1[:]  # FPI 1
        self.L_12, self.L_22 = L_i2[:]  # FPI 2
        
        # Coeficientes de perdidas a traves de los medios
        self.alpha_11, self.alpha_21 = alpha_i1[:]    # FPI 1
        self.alpha_12, self.alpha_22 = alpha_i2[:]    # FPI 2
         
        # Coeficientes de perdidas en las interfaces
        self.A_11, self.A_21 = A_i1[:]  # FPI 1
        self.A_12, self.A_22 = A_i2[:]  # FPI 2

    """
    ==========================================================================
    Metodo I_out:
        Evalua la intensidad de la luz de salida
    ==========================================================================
    """
    def I_out(self):
        
        # Evaluando las reflectancias del FPI 1 
        R_11 = self.R(n0 = self.n_01, n1 = self.n_11) 
        R_21 = self.R(n0 = self.n_11, n1 = self.n_21)
        R_31 = self.R(n0 = self.n_21, n1 = self.n_31)
        # Evaluando las reflectanciaas del FPI 2
        R_12 = self.R(n0 = self.n_02, n1 = self.n_12) 
        R_22 = self.R(n0 = self.n_12, n1 = self.n_22)
        R_32 = self.R(n0 = self.n_22, n1 = self.n_32)
        
        # Coeficientes de perdidas en los medios del FPI 1 
        a_11 = self.alpha_11
        a_21 = self.alpha_21
        # Coeficientes de perdidas en los medios del FPI 2 
        a_12 = self.alpha_12
        a_22 = self.alpha_22
        
        # Coeficientes de perdidas en las interfaces del FPI 1
        A_11 = self.A_11
        A_21 = self.A_21
        # Coeficientes de perdidas en las interfaces del FPI 2
        A_12 = self.A_12
        A_22 = self.A_22
        
        # Longitud de los medios del FPI 1
        L_11 = self.L_11
        L_21 = self.L_21
        # Longitud de los medios del FPI 2
        L_12 = self.L_12
        L_22 = self.L_22
        
        # Evaluando el desfase theta entre las interfaces del FPI 1 
        # al reflejarse la luz         
        O_01 = self.theta(n0=self.n_01, n1=self.n_11)
        O_11 = self.theta(n0=self.n_11, n1=self.n_21)
        O_21 = self.theta(n0=self.n_21, n1=self.n_31)
        
        # Evauando el desfase theta entre las interfaces del FPI 2 
        # al reflejarse la luz         
        O_02 = self.theta(n0=self.n_02, n1=self.n_12)
        O_12 = self.theta(n0=self.n_12, n1=self.n_22)
        O_22 = self.theta(n0=self.n_22, n1=self.n_32)

        # Evaluando el desplazamiento de fase de la luz al pasar por los medios
        # del FPI 1 
        phi_11 = self.phi(lambda_=self.lambda_, n=self.n_11, L=self.L_11)
        phi_21 = self.phi(lambda_=self.lambda_, n=self.n_21, L=self.L_21)
        # Evaluando el desplazamiento de fase de la luz al pasar por los medios
        # del FPI 2
        phi_12 = self.phi(lambda_=self.lambda_, n=self.n_12, L=self.L_12)
        phi_22 = self.phi(lambda_=self.lambda_, n=self.n_22, L=self.L_22)
        
        
        
        I =0.5*(
                # Contribuciones lineales            
                R_11 + R_12 + R_21*((1-A_11)**2)*((1-R_11)**2)*exp(-4*a_11*L_11) +
                R_22*((1-A_12)**2)*((1-R_12)**2)*exp(-4*(a_12*L_12)) +
                R_31*((1-A_11)**2)*((1-A_21)**2)*((1-R_11)**2)*((1-R_21)**2)*exp(-4*(a_11*L_11 + a_21*L_21)) +
                R_32*((1-A_12)**2)*((1-A_22)**2)*((1-R_12)**2)*((1-R_22)**2)*exp(-4*(a_12*L_12 + a_22*L_22)) +
                2*sqrt(R_11*R_12)*cos(O_01 - O_02) +
                # Contribuciones unicas
                2*sqrt(R_11*R_21)*(1-A_11)*(1-R_11)*exp(-2*a_11*L_11)*cos(2*phi_11-(O_01-O_11)) +
                2*sqrt(R_12*R_21)*(1-A_11)*(1-R_11)*exp(-2*a_11*L_11)*cos(2*phi_11-(O_02-O_11)) +
                2*sqrt(R_11*R_22)*(1-A_12)*(1-R_12)*exp(-2*a_12*L_12)*cos(2*phi_12-(O_01-O_12)) +
                2*sqrt(R_12*R_22)*(1-A_12)*(1-R_12)*exp(-2*a_12*L_12)*cos(2*phi_12-(O_02-O_12)) +
                2*sqrt(R_21*R_31)*((1-A_11)**2)*(1-A_21)*((1-R_11)**2)*(1-R_21)*exp(-2*(2*a_11*L_11+a_21*L_21))*cos(2*phi_21-(O_11-O_21)) +
                2*sqrt(R_22*R_32)*((1-A_12)**2)*(1-A_22)*((1-R_12)**2)*(1-R_22)*exp(-2*(2*a_12*L_12+a_22*L_22))*cos(2*phi_22-(O_12-O_22)) +
                # Contribuciones dobles
                2*sqrt(R_11*R_31)*(1-A_11)*(1-A_21)*(1-R_11)*(1-R_21)*exp(-2*(a_11*L_11+a_21*L_21))*cos(2*(phi_11+phi_21)-(O_01-O_21)) +
                2*sqrt(R_12*R_31)*(1-A_11)*(1-A_21)*(1-R_11)*(1-R_21)*exp(-2*(a_11*L_11+a_21*L_21))*cos(2*(phi_11+phi_21)-(O_02-O_21)) +
                2*sqrt(R_11*R_32)*(1-A_12)*(1-A_22)*(1-R_12)*(1-R_22)*exp(-2*(a_12*L_12+a_22*L_22))*cos(2*(phi_12+phi_22)-(O_01-O_22)) +
                2*sqrt(R_12*R_32)*(1-A_12)*(1-A_22)*(1-R_12)*(1-R_22)*exp(-2*(a_12*L_12+a_22*L_22))*cos(2*(phi_12+phi_22)-(O_02-O_22)) +
                2*sqrt(R_21*R_22)*(1-A_11)*(1-A_12)*(1-R_11)*(1-R_12)*exp(-2*(a_11*L_11+a_12*L_12))*cos(2*(phi_12-phi_11)-(O_11-O_12)) +
                # Contribuciones triples
                2*sqrt(R_22*R_31)*(1-A_11)*(1-A_12)*(1-A_21)*(1-R_11)*(1-R_12)*(1-R_21)*exp(-2*(a_11*L_11 + a_12*L_12 + a_21*L_21))*cos(2*(phi_11+ phi_21 - phi_12) - (O_12-O_21)) +                 
                2*sqrt(R_21*R_32)*(1-A_11)*(1-A_12)*(1-A_22)*(1-R_11)*(1-R_12)*(1-R_22)*exp(-2*(a_11*L_11 + a_12*L_12 + a_22*L_22))*cos(2*(phi_12+ phi_22 - phi_11) - (O_11-O_22)) +
                # Contribuciones cuadruples
                2*sqrt(R_31*R_32)*(1-A_11)*(1-A_12)*(1-A_21)*(1-A_22)*(1-R_11)*(1-R_12)*(1-R_21)*(1-R_22)*exp(-2*(a_11*L_11 + a_12*L_12 + a_21*L_21 + a_22*L_22))*cos(2*( phi_12+phi_22 -phi_11 - phi_21 )-(O_21 - O_22))
                )
        return I 
    
    
    """
    ==========================================================================
    Metodo theta:
        Determina el valor de la fase cuando la luz se refleja en una interfaz
    ==========================================================================
    """
    def theta(self, n0, n1):
        if(n0 >= n1):
            return 0
        else:
            return pi
    
    """
    ==========================================================================
    Metodo phi:
        Determina el desplazamiento de fase cuando la luz con longitud de 
        onda lambda_ atraviesa un medio de determinado indice de refraccion n 
        y longitud L
        
        EL factor de 1*10**6 es para ajustar las unidades a milimetros
    ==========================================================================
    """
    def phi(self, lambda_, n, L):
        f = 2*pi*n*L*(1*10**6)/lambda_
        return f
    
    
    """
    ==========================================================================
    Metodo R:
        Determina la reflectancia de la interfaz entre dos medios con indices
        de refraccion n0 y n1.
    ==========================================================================
    """
    def R(self, n0, n1):
        r = (n1 - n0) / (n1 + n0)
        R_ = r*r
        return R_
    
    
    
"""
==============================================================================
Clase para la simulacion del modelo de un interferometro fabry-perot de una
cavidad en paralelo.

Esta clase HEREDA de la clase anterior para ahorrar la escritura de algunos
metodos
==============================================================================
"""

class FPI_1GAP_parallel(FPI_2GAP_parallel):
    """
    ==========================================================================
    Metodo constructor:
        Definicion de Atributos de clase
    ==========================================================================
    """
    
    def __init__(self, lambda_inicial, lambda_final, T_muestreo_lambda, 
                 L_i1, # Longitudes del primer inteferometro 
                 L_i2, # Longitudes del segundo interferometro
                 n_i1, # Indices de los medios del primer interferometros
                 n_i2, # Indices de los medios del segundo interferometro
                 alpha_i1, # Perdida en los medios del primer interferometro  
                 alpha_i2, # Perdida en los medios del segundo interferometro
                 A_11, # Perdida en las interfaces del primer interferometro
                 A_12 # Perdida en las interfaces del segundo interferometro
                 ):
        
        # Array de longitudes de onda de la simulacion
        self.lambda_ = arange(lambda_inicial, lambda_final, T_muestreo_lambda)
        
        # Indices de refraccion de los medios
        self.n_01, self.n_11, self.n_21 = n_i1[:]   # FPI 1 
        self.n_02, self.n_12, self.n_22 = n_i2[:]   # FPI 2
        
        # Longitudes de las cavidades del primer y segundo medio en mm.
        # El primer medio es la longitud de la fibra desde el acoplador hasta
        # la punta
        self.L_01, self.L_11 = L_i1[:]  # FPI 1
        self.L_02, self.L_12 = L_i2[:]  # FPI 2
        
        # Coeficientes de perdidas a traves de los medios
        self.alpha_01, self.alpha_11 = alpha_i1[:]    # FPI 1
        self.alpha_02, self.alpha_12 = alpha_i2[:]    # FPI 2
         
        # Coeficientes de perdidas en las interfaces
        self.A_11 = A_11  # FPI 1
        self.A_12 = A_12  # FPI 2
    
    """
    ==========================================================================
    Metodo I_out:
        Evalua la intensidad de la luz de salida
    ==========================================================================
    """
    def I_out(self):
        
        # Evaluando las reflectancias del FPI 1 
        R_11 = self.R(n0 = self.n_01, n1 = self.n_11) 
        R_21 = self.R(n0 = self.n_11, n1 = self.n_21)
        # Evaluando las reflectanciaas del FPI 2
        R_12 = self.R(n0 = self.n_02, n1 = self.n_12) 
        R_22 = self.R(n0 = self.n_12, n1 = self.n_22)
    
        # Coeficientes de perdidas en los medios del FPI 1 
        a_01 = self.alpha_01
        a_11 = self.alpha_11
        # Coeficientes de perdidas en los medios del FPI 2 
        a_02 = self.alpha_02
        a_12 = self.alpha_12
        
        # Coeficientes de perdidas en las interfaces del FPI 1
        A_11 = self.A_11
        # Coeficientes de perdidas en las interfaces del FPI 2
        A_12 = self.A_12
        
        # Longitud de los medios del FPI 1
        L_01 = self.L_01
        L_11 = self.L_11
        # Longitud de los medios del FPI 2
        L_02 = self.L_02
        L_12 = self.L_12
        
        # Evaluando el desfase theta entre las interfaces del FPI 1 
        # al reflejarse la luz         
        O_01 = self.theta(n0=self.n_01, n1=self.n_11)
        O_11 = self.theta(n0=self.n_11, n1=self.n_21)
        
        # Evauando el desfase theta entre las interfaces del FPI 2 
        # al reflejarse la luz         
        O_02 = self.theta(n0=self.n_02, n1=self.n_12)
        O_12 = self.theta(n0=self.n_12, n1=self.n_22)

        # Evaluando el desplazamiento de fase de la luz al pasar por los medios
        # del FPI 1 
        phi_01 = self.phi(lambda_=self.lambda_, n=self.n_01, L=self.L_01)
        phi_11 = self.phi(lambda_=self.lambda_, n=self.n_11, L=self.L_11)
        # Evaluando el desplazamiento de fase de la luz al pasar por los medios
        # del FPI 2
        phi_02 = self.phi(lambda_=self.lambda_, n=self.n_02, L=self.L_02)
        phi_12 = self.phi(lambda_=self.lambda_, n=self.n_12, L=self.L_12)
        
        
        
        I =0.25*(
                # Contribuciones lineales 
                R_11*exp(-4*a_01*L_01) + R_12*exp(-4*a_02*L_02) + 
                R_21*((1-A_11)**2)*((1-R_11)**2)*exp(-4*(a_01*L_01 + a_11*L_11))+
                R_22*((1-A_12)**2)*((1-R_12)**2)*exp(-4*(a_02*L_02 + a_12*L_12))+
                # Contribuciones unicas
                2*sqrt(R_11*R_21)*(1-A_11)*(1-R_11)*exp(-2*(2*a_01*L_01 + a_11*L_11))*cos(2*phi_11 + (O_11 - O_01)) + 
                2*sqrt(R_12*R_22)*(1-A_12)*(1-R_12)*exp(-2*(2*a_02*L_02 + a_12*L_12))*cos(2*phi_12 + (O_12 - O_02)) -
                # Contribucions dobles
                2*sqrt(R_11*R_12)*exp(-2*(a_01*L_01 + a_02*L_02))*cos(2*(phi_02-phi_01) + (O_02 - O_01)) - 
                # Contribuciones triples 
                2*sqrt(R_11*R_22)*(1-A_12)*(1-R_12)*exp(-2*(a_01*L_01 + a_02*L_02+ a_12*L_12))*cos(2*(phi_12 + phi_02-phi_01) + (O_12 - O_01)) - 
                2*sqrt(R_12*R_21)*(1-A_11)*(1-R_11)*exp(-2*(a_01*L_01 + a_02*L_02+ a_11*L_11))*cos(2*(phi_11 + phi_01-phi_02) + (O_11 - O_02)) -
                # Contribuciones cuadruples
                2*sqrt(R_21*R_22)*(1-A_11)*(1-A_12)*(1-R_11)*(1-R_12)*exp(-2*(a_01*L_01 + a_02*L_02+ a_11*L_11+ a_12*L_12))*cos(2*(phi_02 - phi_01 + phi_12 - phi_11) + (O_12 - O_11))
                )
        return I
        
        