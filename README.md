# Codigos para Tesis de licenciatura: Estudio de la respuesta de dos interferómetros Fabry-Pérot extrínsecos de dos cavidades en serie y en paralelo

Los códigos aquí mostrados y construidos por el Ing. Israel Alejandro Gómez Pérez, sirvieron ampliamente en el estudio y análisis de la reflectancia en las variantes de los interferómetros Fabry-Pérot. Brevemente se bosqueja el panorama general del funcionamiento de dichos códigos escritos en el lenguaje de programación Python.

# Objetivos:
- Procesar automáticamente espectros de laboratorio en conjunto
- Aplicar filtros, y técnicas para mejorar la señal base de los espectros del laboratorio
- Simular los modelos discutidos en el documento de tésis
- Realizar una sencilla animación del compartamiento de espectros tomados consecutivamente 

# Requerimientos de software

## Sobre el sistema operativo

Todos los códigos fueron desarrollados usando el sistema operativo Linux Ubuntu 20.04. Sin embargo, el lenguaje de programación Python, es universal y valido en cualquier sistema operativo, siempre y cuando se tenga el compilador de Python pertinente. 

Todos los códidos estan desarrollados en Python 3.9.4 y haciendo uso de Anaconda3 (conda 4.10.3). Se recomienda ampliamente el uso de un ambiente virtual para correr los códigos debido a que algunos programas requieren de paqueterias especiales para correr de manera adecuada. 

# Contenido

Se detalla brevemente el condetido de cada carpeta.

## Mediciones


## Simulacion_FPI_Serie
Contiene la simulacion de un interferometro de Fabry-Perot de dos cavidades para la señal de la reflectancia. Se realiza una animación de la variación de la señal conforme al parámetro $L_{1}$, que es la longitud de la primera cavidad. Además se ha realizado la transformada de fourier de la señal.

## Simulacion_FPI_Paralelo
