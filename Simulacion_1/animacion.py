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
from FabryPerot.FFT_support import encontrar_FFT

# Creamos la figura
fig,ax = plt.subplots()

#ax1,ax2 = plt.subplots(1,2)
# Para que las graficas no se empalmen
fig.set_tight_layout(True)

lambda_ = np.arange(1500,1600,0.01)

#fig.suptitle("L = ")

obj = FabryPerot_2GAP(lambda_inicial=1500,lambda_final= 1600,L_medio_1 = .4, L_medio_2=.8, eta_medio_1 = 1.0, eta_medio_2 = 1.332, eta_medio_3=1.48)
reflectancia = obj.R()

x_fft, y_fft = encontrar_FFT(lambda_inicial = 1500, T_muestreo_lambda=0.01, Reflectancia=reflectancia)

ax = plt.subplot(1,2,1)

graph, = ax.plot(lambda_,10*np.log10(reflectancia))
ax.set_ylabel(r"R [dB]")
ax.set_xlabel(r"$\lambda$")
ax.set_xlim([1500,1550])
ax.set_ylim([-40,-5])
ax.set_title("Simulación FP - 2 GAP.")

ax = plt.subplot(1,2,2)

graph_2, = ax.plot(x_fft,y_fft, c="green")
#ax.title.set_text("FFT")

ax.set_ylabel(r"R")
ax.set_xlabel(r"OPL[$mm$]")
ax.set_xlim([-0.1,2.5])
#ax.set_ylim([-40,-5])
ax.set_title("FFT")


def actualizar(i):
    di = 0.02
    L1 = round((i+1)*di,2)
    label = f"$L_{1}$ = {L1} mm"
    obj = FabryPerot_2GAP(lambda_inicial=1500,lambda_final= 1600,L_medio_1 = L1, L_medio_2=0.8, eta_medio_1 = 1.0, eta_medio_2 = 1.332, eta_medio_3=1.48)
    ref = obj.R()
    reflectancia_db = 10*np.log10(ref)
        
    x_fft, y_fft = encontrar_FFT(lambda_inicial = 1500, T_muestreo_lambda=0.01, Reflectancia=ref)
    
    fig.suptitle(label)
    
    graph.set_ydata(reflectancia_db)
    #ax.set_title(label)
    graph_2.set_ydata(y_fft)
    return graph, graph_2, ax


anim = FuncAnimation(fig, actualizar, repeat = True, frames= np.arange(0,50), interval = 1000 )

#plt.show()

# Guardaremos la animacion
Writer = writers["ffmpeg"]
writer = Writer(fps=1, metadata={"artist":"IAGP"}, bitrate=1800)
anim.save("Variacion_L1_FFT.mp4",writer)
