#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 14:01:59 2021

@author: alejandro_goper


Codigo principal de la primera simulacion 

"""

import numpy as np
import matplotlib.pyplot as plt
from FabryPerot.Clase import FabryPerot_2GAP

"""
==============================================================================

Simulación de la Reflectancia R como funcion de la longitud de onda en un 
Interferometro de Fabry-Perot en serie de 2 cavidades (GAP).

==============================================================================
"""

# Definicion del dominio en longitudes de onda
lambda_ = np.arange(1510,1590+0.005,0.005) #nanometros


obj = FabryPerot_2GAP(lambda_inicial=1510,
                      lambda_final= 1590,
                      T_muestreo_lambda=0.005,
                      L_medio_1 = 0.187, 
                      L_medio_2 = 1, 
                      eta_medio_1 = 1.0, 
                      eta_medio_2 = 1.667)
reflectancia = obj.Reflectancia()


plt.figure()
#plt.plot(lambda_,10*np.log10(reflectancia))
plt.plot(lambda_,reflectancia,linewidth=0.5)
#plt.xlim([1500,1503])
plt.show()