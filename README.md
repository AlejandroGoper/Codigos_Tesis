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

Dicha carpeta contiene los códigos para el procesamiento automatico de diversos espectros obtenidos del laboratorio. 

### 08-10-2021
Dicha carpeta contiene dos Jupiter Notebooks para diferentes propositos:

- interact_FFT: Este jupiter notebook permite procesar y modificar en tiempo real, un espectro de Fourier específico localizado en una carpeta determinada y especificada por el usuario, y variar los parametros escenciales tanto visuales como de procesamiento; por ejemplo, se puede cambiar de espectro (siempre y cuando esté localizado en la carpeta), se puede agregar un determinado número de 0 al inicio y final de la señal (post-windowing) para mejorar la resolución en el espacio de Fourier, dc_margin_ es un parámetro que permite eliminar la señal desde 0 mm hasta el valor específicado por el usuario, lim_amp_ es un parámetros que permite localizar los máximos en el espectro de fourier por encima de este valor, beta_ es el parámetro de la ventana Kaisser-Bessel que permite modificar el tipo de ventana según el valor númerico ingresado (de esta manera se pueden probar diferentes ventanas en tiempo real), lim_inf_x y lim_sup_x son parametros que controlar la visualización de la imagen.

- utilidades: Este jupiter notebook contiene fragmentos de código para soluciones AD-HOC, por ejemplo, permite realizar el ajuste lineal de datos, permite realizar gráficos de caja, calcular desviación estandar y promedios, graficar espectros únicos en dominio óptico y de Fourier, filtrar un espectro de laboratorio, filtrado + tecnica windowing, graficar conjuntamente el dominio óptico y de fourier de un solo espectro (con las tencicas de mejora de la senal), graficar la simulación del modelo tanto en serie como en paralelo. 

### 18-10-2021

- utilidades: Este jupiter notebook esta dedicado principalmente a graficar el desplazamiento nominal de las monturas de desplazamiento vs. el desplazamiento medido en el espacio de Fourier y realizar un ajuste lineal, y otro tipo de pruebas no reelevantes para el procesamiento de los espectros en sí mismo.

**Cabe señalar que**: Dentro de la carpeta 08-10-2021 y diversas partes más se encuentra una carpeta llamada **FabryPerot**. Esta carpeta contiene los codigos escenciales para llevar a cabo la serie discreta de Fourier (FFT) así como el filtrado de la señal y la ténica windowing en Python. Dichos codigos están documentados y no es necesario que se explique el functionamiento a detalle en esta sección.

### Mediciones



## Simulacion_FPI_Serie

Contiene la simulacion de un interferometro de Fabry-Perot de dos cavidades para la señal de la reflectancia. Se realiza una animación de la variación de la señal conforme al parámetro $L_{1}$, que es la longitud de la primera cavidad. Además se ha realizado la transformada de fourier de la señal.

- main.py: Contiene una prueba sencilla de como realizar un simulación con los códigos mencionados.
- animación.py: Contiene el codigo necesario para realizar un video en formato .mp4 del comportamiento de la reflectancia (simulada) variando la longitud de una de las cavidades (se puede modificar según las necesidades)
- test_procesado_simulacion.py: Aplica los mismo métodos de mejora de la señal que se han aplicado a los espectros de laboratorio, a la reflectancia simulada.
- FFT.py: Encuentra la serie discreta de Fourier de la simulación para determinadas condiciones.

**Todos los códigos están documentados.**

## Simulacion_FPI_Paralelo

Contiene la simulacion de un interferometro de Fabry-Perot de dos cavidades para la señal de la reflectancia. Se realiza una animación de la variación de la señal conforme al parámetro $L_{1}$, que es la longitud de la primera cavidad. Además se ha realizado la transformada de fourier de la señal.

- animacion.py: Contiene el codigo necesario para realizar un video en formato .mp4 del comportamiento de la reflectancia (simulada) variando la longitud de una de las cavidades (se puede modificar según las necesidades)
- test_clase: Aplica los mismo métodos de mejora de la señal que se han aplicado a los espectros de laboratorio, a la reflectancia simulada.

