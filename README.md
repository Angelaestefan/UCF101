# Dataset UCF101

## Descripción

El dataset UCF101 es una colección de videos de acciones realistas recopilados de YouTube con 101 categorías de acciones. Este proyecto implementa un modelo de reconocimiento de acciones basado en esqueletos que analiza datos de pose humana para clasificar diferentes acciones y actividades.

El modelo utiliza puntos clave de esqueletos 2D extraídos de fotogramas de video para entender los movimientos humanos y predecir clases de acciones como actividades deportivas, tocar instrumentos y acciones cotidianas.

## Arquitectura del Modelo

El modelo de reconocimiento de acciones basado en esqueletos está estructurado de la siguiente manera:

### Componentes Principales:
- **Extractor de Características**: Procesa las secuencias de puntos clave del esqueleto 2D
- **Red Temporal**: Captura las dependencias temporales entre frames usando LSTM/GRU o Transformer
- **Red Espacial**: Analiza las relaciones espaciales entre articulaciones del cuerpo humano
- **Clasificador**: Capa final que predice la clase de acción entre las 101 categorías

### Flujo de Datos:
1. Entrada: Secuencias de coordenadas de esqueletos 2D (17 puntos clave por frame)
2. Procesamiento temporal y espacial de las características
3. Fusión de características espacio-temporales
4. Clasificación final mediante softmax

## Instrucciones de Configuración

### Descargar Datos de Esqueletos

Descarga los datos de esqueletos 2D de UCF101 desde:
https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl
