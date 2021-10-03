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
lambda_ = np.arange(1500,1600,0.01) #nanometros


obj = FabryPerot_2GAP(lambda_inicial=1500,lambda_final= 1600,L_medio_1 = 4, L_medio_2=8, eta_medio_1 = 1.0, eta_medio_2 = 1.332, eta_medio_3=1.48)
reflectancia = obj.R()


plt.figure()
#plt.plot(lambda_,10*np.log10(reflectancia))
plt.plot(lambda_,reflectancia)
plt.xlim([1500,1503])
plt.show()