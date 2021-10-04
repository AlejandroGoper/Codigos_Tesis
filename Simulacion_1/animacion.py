#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 19:06:57 2021

@author: alejandro_goper

Codigo para crear animacion de la simulacion FabryPerot

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from FabryPerot.Clase import FabryPerot_2GAP

# Creamos la figura
fig,ax = plt.subplots()
# Para que las graficas no se empalmen
fig.set_tight_layout(True)

lambda_ = np.arange(1500,1600,0.01)

obj = FabryPerot_2GAP(lambda_inicial=1500,lambda_final= 1600,L_medio_1 = .4, L_medio_2=.8, eta_medio_1 = 1.0, eta_medio_2 = 1.332, eta_medio_3=1.48)
reflectancia = obj.R()

graph, = ax.plot(lambda_,10*np.log10(reflectancia))
ax.set_ylabel(r"dB")
ax.set_xlabel(r"$\lambda$")
ax.set_xlim([1500,1550])
ax.set_ylim([-10,5])
ax.set_title("Simulación Fabry-Perot 2 Cavidades en serie.")

def actualizar(i):
    di = 0.02
    L1 = round((i+1)*di,2)
    label = f"Simulación Fabry-Perot 2 Cavidades en serie :: $L_{1}$ = {L1}"
    obj = FabryPerot_2GAP(lambda_inicial=1500,lambda_final= 1600,L_medio_1 = 0.4, L_medio_2=0.8, eta_medio_1 = L1, eta_medio_2 = 1.332, eta_medio_3=1.48)
    reflectancia_db = 10*np.log10(obj.R())
    graph.set_ydata(reflectancia_db)
    ax.set_title(label)
    return graph, ax


anim = FuncAnimation(fig, actualizar, repeat = True, frames= np.arange(0,50), interval = 1000 )

#plt.show()

# Guardaremos la animacion
Writer = writers["ffmpeg"]
writer = Writer(fps=1, metadata={"artist":"IAGP"}, bitrate=1800)
anim.save("Variacion_L1.mp4",writer)