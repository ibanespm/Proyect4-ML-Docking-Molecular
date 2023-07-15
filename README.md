# Proyecto 4

## Integrantes

-   Gabriel Loayza
-   Ibañes Perez
-   Edir Vidal
  

## Descripción

El docking molecular es un método computacional que predice cómo se pueden unir dos moléculas,
como un fármaco (ligando) y una proteína (receptor). Sirve para estudiar interacciones moleculares, diseñar
nuevos fármacos y entender cómo los compuestos se unen a sus objetivos biológicos, lo que ayuda en la investigación
y desarrollo de medicamentos.



## Consideraciones

El proyecto consta de varios archivos. El archivo fundamental es ```Red.h``` allí se e
ncuentran todas las definiciones de las funciones usadas para el
entrenamiento y predicción con la red. El archivo de utilidad es ```classificacion_model.cpp```. 
Este archivo permite crear cualquier tipo de MLP pasandole argumentos en su llamada en la terminal
(las opciones se pueden ver en el mismo archivo). Finalmente, tenemos archivos de utilidad .py que 
permitieron cargar el dataset, hacer los tests, y graficar los resultados obtenidos. 

El proyecto consta con una carpeta de date set, donde esta nuestra base datos. Tambien incluye un cuadernillo jupyter(.ipynb)  que debe agregarse el directorio del data set ejecutarlo.
Por otro lado, si solo quiremos ejecutar el test.py solo vamos a probar los datos ya entrenados(model.pth).


## Ver resultados

Ejecutar el test.py, pero instalar los requerimiento de las librerias.
