from __future__ import annotations

# ============================================================================
# CATALOGO DE DATASETS
# Este archivo describe como leer cada dataset. No controla la ejecucion.
# Las posiciones son 1-based e inclusivas cuando se usa "rango".
# ============================================================================

DATASETS: dict[str, dict[str, object]] = {
    # Sinteticos
    "Synthetic-1d": {"tipo": "sintetico", "k": 4, "puntos_por_cluster": 1000},
    "Synthetic-1d-Over": {"tipo": "sintetico", "k": 4, "puntos_por_cluster": 1000},
    "Synthetic-2d": {"tipo": "sintetico", "k": 4, "puntos_por_cluster": 1000},
    "Synthetic-2d-Over": {"tipo": "sintetico", "k": 4, "puntos_por_cluster": 1000},

    # Embeddings
    "ecommerce-clip": {
        "tipo": "embedding",
        "archivo": "pequenos/ecommerce-clip.csv",
        "fuente": "imagenes/ecommerce products",
        "etiqueta": "etiqueta",
        "x": {"prefijo": "v_"},
    },
    "mechanical-tools-clip": {
        "tipo": "embedding",
        "archivo": "pequenos/mechanical-tools-clip.csv",
        "fuente": "imagenes/Mechanical Tools Image dataset",
        "etiqueta": "etiqueta",
        "x": {"prefijo": "v_"},
    },
    "bbc-text-clip": {
        "tipo": "embedding_texto",
        "archivo": "medianos/bbc-text-clip.csv",
        "fuente": "csv_texto/bbc_data.csv",
        "texto": "data",
        "etiqueta_fuente": "labels",
        "etiqueta": "etiqueta",
        "x": {"prefijo": "v_"},
    },

    # Tabulares
    "iris": {
        "tipo": "csv",
        "archivo": "pequenos/Iris.csv",
        "etiqueta": "Species",
        "x": {"columnas": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]},
    },
    "heart-disease": {
        "tipo": "csv",
        "archivo": "pequenos/heart.csv",
        "etiqueta": "target",
        "x": {"rango": (1, 13)},
    },
    "obesity-levels": {
        "tipo": "csv",
        "archivo": "pequenos/Obesity.csv",
        "etiqueta": "NObeyesdad",
        "x": {"columnas": ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]},
    },
    "glass": {
        "tipo": "csv",
        "archivo": "pequenos/glass.csv",
        "etiqueta": "Type",
        "x": {"rango": (1, 9)},
    },
    "breast-cancer-wisconsin": {
        "tipo": "csv",
        "archivo": "pequenos/cancer.csv",
        "etiqueta": "diagnosis",
        "x": {"rango": (3, 32)},
    },
    "engineering-graduate-salary": {
        "tipo": "csv",
        "archivo": "medianos/Salary.csv",
        "etiqueta": "CollegeTier",
        "x": {"excluir": [1, 2, 3, 5, 8, 10, 11, 12, 16]},
    },
    "water-probability": {
        "tipo": "csv",
        "archivo": "medianos/Water.csv",
        "etiqueta": "Potability",
        "x": {"rango": (1, 9)},
        "imputar": True,
    },
    "cure-the-princess": {
        "tipo": "csv",
        "archivo": "medianos/Cure.csv",
        "etiqueta": "Cured",
        "x": {"rango": (1, 13)},
    },
    "aids-clinical": {
        "tipo": "csv",
        "archivo": "medianos/AIDS.csv",
        "etiqueta": "label",
        "x": {"rango": (1, 23)},
    },
    "migration-mexico-usa": {
        "tipo": "csv",
        "archivo": "medianos/Migration.csv",
        "etiqueta": "GIM_2000",
        "x": {"excluir": [3, 10]},
    },
    "bank-loan-approval": {
        "tipo": "csv",
        "archivo": "grandes/bank.csv",
        "etiqueta": "Personal.Loan",
        "x": {"excluir": [1, 10]},
    },
    "wine-quality": {
        "tipo": "csv",
        "archivo": "grandes/wine.csv",
        "etiqueta": "type",
        "x": {"rango": (2, 13)},
        "imputar": True,
    },
    "cycling-clustering": {
        "tipo": "csv",
        "archivo": "grandes/bike.csv",
        "etiqueta": "'Cluster'",
        "x": {"rango": (1, 10)},
    },
    "turkiye-student-evaluation": {
        "tipo": "csv",
        "archivo": "grandes/turkiye.csv",
        "etiqueta": "class",
        "x": {"rango": (3, 33)},
    },
    "abalone": {
        "tipo": "csv",
        "archivo": "grandes/abalone.csv",
        "etiqueta": "sex",
        "x": {"rango": (2, 8)},
    },
    "Adult": {
        "tipo": "csv",
        "archivo": "extra_grandes/adult_p.csv",
        "etiqueta": "type",
    },
    "Bank": {
        "tipo": "csv",
        "archivo": "extra_grandes/bank_p.csv",
        "etiqueta": "type",
    },
    "Diabetes": {
        "tipo": "csv",
        "archivo": "extra_grandes/diabetic_p.csv",
        "etiqueta": "type",
    },
}
