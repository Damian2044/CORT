from __future__ import annotations

# ============================================================================
# CONFIGURACION DEL EXPERIMENTO
# Cambia solo esta sección para preparar una ejecución nueva.
# ============================================================================

# Datasets que se ejecutan. Los nombres deben existir en catalogo_datasets.py.
DATASETS_A_CORRER: tuple[str, ...] = (
    # Sinteticos
    "Synthetic-1d",
    "Synthetic-1d-Over",
    "Synthetic-2d",
    "Synthetic-2d-Over",

    # Embeddings
    "ecommerce-clip",
    "mechanical-tools-clip",
    "bbc-text-clip",

    # Tabulares
    "iris",
    "heart-disease",
    "obesity-levels",
    "glass",
    "breast-cancer-wisconsin",
    "engineering-graduate-salary",
    "water-probability",
    "cure-the-princess",
    "aids-clinical",
    "migration-mexico-usa",
    "bank-loan-approval",
    "wine-quality",
    "cycling-clustering",
    "turkiye-student-evaluation",
    "abalone",
    "Adult",
    "Bank",
    "Diabetes",
)

# Algoritmos comparados. CORT es el algoritmo propuesto.
ALGORITMOS_A_CORRER: tuple[str, ...] = ("LIB", "LIBH", "COCA", "COCH", "CORT")

# Metricas que se calculan y se guardan en resultados.
METRICAS_A_CALCULAR: tuple[str, ...] = (
    "SIL",
    "ARI",
    "AMI",
    "NMI",
    "CENTROS",
    "CARDINALIDAD_CUARTILES",
    "TIEMPO_ALGORITMO",
    "RAM_MAX",
    "INCREMENTO_RAM",
)

# Semillas consecutivas para cambiar el orden de llegada del flujo.
# Con SEMILLA_INICIAL = 0 y CANTIDAD_SEMILLAS = 20 se usan semillas 0..19.
SEMILLA_INICIAL = 0
CANTIDAD_SEMILLAS = 20

# Si es entero, todos los sinteticos usan la misma nube base.
# Si es None, cada semilla genera su propia nube sintetica.
SEMILLA_SINTETICOS: int | None = 123

# Si n supera este limite, SIL se calcula con muestra reproducible.
LIMITE_SILUETA = 50000

# Modalidades disponibles:
# - "secuencial": una ejecucion a la vez, recomendado para bajos recursos.
# - "paralelo": usa multiprocessing.Pool y reutiliza workers para acelerar.
# - "paralelo_aislado": usa multiprocessing.Pool con maxtasksperchild=1.
MODALIDAD_EJECUCION = "paralelo_aislado"
WORKERS = 5

# Carpeta y nombre base de salida.
PREFIJO_SALIDA = "experimento_completo_final_paralelo_aislado_w5"
GUARDAR_RESULTADOS = True

# Muestreo para RAM. Un valor menor mide mejor picos, pero agrega algo de costo.
INTERVALO_RAM_SEGUNDOS = 0.01
