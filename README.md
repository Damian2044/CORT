# CORT: Clustering Online Con Restricciones De Tamaño

## Introducción

Este repositorio contiene el entorno experimental de **CORT**, un algoritmo de Clustering Online con restricciones exactas de cardinalidad. CORT procesa los datos en una sola pasada y, para cada observación entrante, decide entre asignarla a un clúster activo o fundar un clúster vacío disponible, respetando las cardinalidades objetivo.

## Algoritmos Implementados

### Algoritmos De Referencia

1. **LIB**  
   Algoritmo online basado en apertura probabilística de centros. No impone cardinalidades exactas y puede abrir más centros que el número objetivo de clústeres.

2. **LIBH**  
   Variante heurística de LIB. Usa una estimación inicial más controlada para reducir la apertura excesiva de centros.

3. **COCA**  
   Algoritmo online capacitado. En esta implementación se usa la modalidad base con capacidad uniforme aproximada `ceil(n/k)` por centro abierto. Primero calcula un valor inicial `M` mediante `coupon_collector_non_uniform`, abre centros en esa fase inicial y estima el costo de apertura con esos puntos. Luego procesa el resto del flujo asignando a centros con capacidad disponible o abriendo nuevos centros según la regla probabilística. COCA controla capacidad, pero no impone cardinalidades exactas finales.

4. **COCH**  
   Variante heurística de COCA. Usa una inicialización más pequeña, basada en `k + 1`, y también trabaja con capacidad uniforme aproximada `ceil(n/k)`. Reduce el costo de la fase inicial, pero tampoco garantiza cardinalidades exactas finales.

### Algoritmo Propuesto

5. **CORT**  
   Algoritmo propuesto de Clustering Online con restricciones exactas de cardinalidad. Procesa el flujo en una sola pasada y decide, para cada observación, si conviene asignarla a un clúster ya fundado o fundar uno nuevo. A diferencia de los algoritmos de referencia, CORT fuerza el cumplimiento exacto de las cardinalidades objetivo.

## Estructura Del Repositorio

- `main.py`  
  Runner principal de los experimentos.

- `config.py`  
  Archivo de opciones del experimento: Datasets, algoritmos, métricas, semillas, modalidad de ejecución, workers y salida.

- `catalogo_datasets.py`  
  Define cómo se lee cada dataset: Archivo, columna etiqueta, columnas X, imputación y fuente de embeddings.

- `algoritmos/`  
  Implementaciones de CORT y de los algoritmos de referencia.

- `datos/`  
  Contiene datasets listos para ejecutar y las carpetas base para fuentes de embeddings.

- `preparacion_embeddings/`  
  Generación de embeddings LAION-CLIP para imágenes o texto.

- `metricas.py`  
  Calcula métricas internas, métricas externas, tiempo, centros, cardinalidades finales y uso de RAM.

- `reportes.py`  
  Genera CSV, resumen compacto, imágenes de tabla y `resumen_ejecucion.json`.

- `resultados/`  
  Carpeta local donde se guardan las salidas de cada experimento.

## Datasets

Los datasets se organizan en tres grupos:

- **Sintéticos**: Generados desde el protocolo configurado.
- **Embeddings**: Generados con LAION-CLIP desde imágenes o texto.
- **Tabulares**: Archivos CSV con columna etiqueta y columnas X definidas en el catálogo.

Categorías Por Tamaño:

- **Pequeños**: `n <= 2120`
- **Medianos**: `2120 < n <= 4000`
- **Grandes**: `4001 <= n <= 6500`
- **Extra Grandes**: `n > 6500`

## Instalación

Desde la carpeta `CORT_experimentos`:

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` instala las librerías necesarias para ejecutar los experimentos y generar embeddings con LAION-CLIP en CPU.

## Configuración De Experimentos

La forma recomendada es editar `config.py`.

Opciones Principales:

- `DATASETS_A_CORRER`: Datasets que se ejecutarán.
- `ALGORITMOS_A_CORRER`: Algoritmos comparados.
- `METRICAS_A_CALCULAR`: Métricas guardadas en los reportes.
- `SEMILLA_INICIAL` y `CANTIDAD_SEMILLAS`: Semillas consecutivas del flujo.
- `SEMILLA_SINTETICOS`: Controla la generación de datasets sintéticos.
- `LIMITE_SILUETA`: Límite para calcular SIL con muestra reproducible.
- `MODALIDAD_EJECUCION`: Modalidad de ejecución.
- `WORKERS`: Número de procesos usados en modalidades paralelas. Por defecto se usa `5`.
- `PREFIJO_SALIDA`: Nombre base de la carpeta de resultados.

## Modalidades De Ejecución

- `secuencial`  
  Ejecuta una tarea a la vez. Es la opción más simple y recomendable para datasets pequeños o equipos con pocos recursos.

- `paralelo`  
  Usa `multiprocessing.Pool` y reutiliza workers. Es la opción recomendada para acelerar experimentos grandes.

- `paralelo_aislado`  
  Usa `multiprocessing.Pool(processes=workers, maxtasksperchild=1)`. Cada tarea corre en un proceso limpio. Esto permite una medición de RAM más aislada, aunque puede tardar más.

## Cómo Reproducir Los Experimentos

### Ejecutar Con La Configuración De `config.py`

```bash
python main.py
```

### Ejecutar Un Dataset Específico

```bash
python main.py --datasets iris --semillas 20 --modalidad secuencial
```

### Ejecutar Varios Datasets

```bash
python main.py --datasets Synthetic-1d ecommerce-clip iris --semillas 20
```

### Ejecutar Solo Algunos Algoritmos

```bash
python main.py --datasets iris --algoritmos LIB COCA CORT --semillas 20
```

### Ejecutar En Paralelo

```bash
python main.py --modalidad paralelo --workers 5
```

### Ejecutar Con Procesos Aislados

```bash
python main.py --modalidad paralelo_aislado --workers 5
```

### Cambiar El Prefijo De Salida

```bash
python main.py --datasets iris --semillas 20 --prefijo-salida prueba_iris
```

## Resultados Generados

Cada ejecución crea una carpeta dentro de `resultados/` con marca de tiempo:

```text
resultados/comparacion_YYYYMMDD_HHMMSS/
```

Dentro se guardan:

- `resultados_detalle.csv`: Una fila por dataset, algoritmo y semilla.
- `resultados_resumen.csv`: Promedios y desviaciones por dataset y algoritmo.
- `resultados_resumen_compacto.csv`: Tabla compacta lista para revisar.
- `resultados_resumen_compacto.png`: Imagen de la tabla compacta.
- `resumen_ejecucion.json`: Configuración usada, tiempo total y archivos generados.

## Métricas Reportadas

En los resúmenes compactos, las columnas con formato `m|std` muestran `media|desviación estándar` calculadas sobre las semillas ejecutadas.

- `SIL`: Coeficiente de silueta.
- `ARI`: Adjusted Rand Index.
- `AMI`: Adjusted Mutual Information.
- `NMI`: Normalized Mutual Information.
- `Centros`: Número de centros o clústeres abiertos por el algoritmo.
- `Tiempo (s)`: Tiempo de ejecución medido solo durante el algoritmo.
- `RAM max (MB)`: Mayor RAM observada durante la ejecución del algoritmo.
- `Inc RAM (MB)`: Diferencia entre la RAM al inicio del algoritmo y el pico máximo observado durante ese algoritmo.
- `Cuant. card.`: Cuantiles de las cardinalidades finales de los clústeres.

## Embeddings

Los datasets de embeddings activos son:

- `ecommerce-clip`
- `mechanical-tools-clip`

Si el CSV de embeddings no existe, el cargador lo genera automáticamente desde las fuentes configuradas en `catalogo_datasets.py`.

El modelo usado para generar embeddings es:

```text
hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K
```

Las fuentes de imágenes se colocan en:

```text
datos/fuentes/imagenes/
```

Las fuentes de texto se colocan en:

```text
datos/fuentes/csv_texto/
```

Los embeddings generados se guardan como CSV dentro de:

```text
datos/datasets/
```

## Recomendación Práctica

Para pruebas pequeñas:

```bash
python main.py --datasets iris --semillas 1 --modalidad secuencial
```

Para experimentos completos en una máquina con varios núcleos:

```bash
python main.py --modalidad paralelo --workers 5
```

Para revisar RAM con procesos limpios:

```bash
python main.py --modalidad paralelo_aislado --workers 5
```
