# Aplicación sobre predicción de energía del Parque Eólico Villonaco

Esta aplicación se basa en la predicción de potencia activa para el Parque Eólico Villonaco (VWF), ubicado en el sur de Ecuador a aproximadamente 2700 m sobre el nivel del mar. Mediante el uso de redes neuronales artificiales, se desarrollaron pruebas experimentales basadas en los modelos de Perceptrón Multicapa (MLP), Memoria de Largo Corto Plazo (LSTM) y Red Neural Convolucional (CNN) para obtener un modelo híbrido que se ajusta mejor características de los modelos individuales. Los datos del sistema SCADA (Supervisory Control and Data Acquisition) de potencia activa para los años 2014 a 2018 se utilizan para entrenar y validar los modelos. El modelo híbrido implementado se presenta como la opción más adecuada por los valores obtenidos, es decir, el error absoluto medio (MAE) y el error cuadrático medio (MSE) que fueron 0.1365 y 0.0974, respectivamente, superando a los demás modelos de previsión de energía eólica.

## Instalación 

La aplicación se ha desarrollado usando el Framework Django versión 3.2.+ con la versión de Python 3.6

Las librerías se instalan usando el archivo requerimientos.txt. Se puede ejecutar la siguiente línea en el terminal

`pip install -r requerimientos.txt`

## Configuración 

Si es necesario se carga la base de datos usando el archivo sql. Luego de haber creado la base de datos llamada _powerprediction_ 

`mysql -u<user> -p powerprediction < pp.sql`

### Información y contacto

Jorge Maldonado: jorgemaldonadoc@unl.edu.ec 
Marlon Viñan: marsantovi@gmail.com 



