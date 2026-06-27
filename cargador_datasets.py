from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle as sklearn_shuffle

from algoritmos.utilidades import DatasetBundle
from catalogo_datasets import DATASETS


RAIZ_PROYECTO = Path(__file__).resolve().parent
RAIZ_DATOS = RAIZ_PROYECTO / "datos"
RAIZ_DATASETS = RAIZ_DATOS / "datasets"
SCRIPT_EMBEDDINGS = RAIZ_PROYECTO / "preparacion_embeddings" / "generar_embeddings.py"
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class DatasetCargado:
    bundle: DatasetBundle
    cardinalidad_objetivo: tuple[int, ...]
    categoria: str


def clasificar_por_tamano(n_filas: int) -> str:
    if n_filas <= 2120:
        return "pequenos"
    if n_filas <= 4000:
        return "medianos"
    if n_filas <= 6500:
        return "grandes"
    return "extra_grandes"


def cargar_dataset(nombre: str, semilla_sinteticos: int | None = 123) -> DatasetCargado:
    """Carga un dataset, aplica su limpieza declarada y devuelve X/y listos."""
    if nombre not in DATASETS:
        raise ValueError(f"Dataset no configurado: {nombre}")

    regla = DATASETS[nombre]
    tipo = str(regla["tipo"])

    if tipo == "sintetico":
        return _cargar_sintetico(nombre, regla, semilla_sinteticos)

    if tipo in {"embedding", "embedding_texto"}:
        _asegurar_embedding(nombre, regla)

    return _cargar_csv(nombre, regla)


def mezclar_dataset(dataset: DatasetCargado, semilla: int) -> DatasetCargado:
    """Aplica el mismo barajado usado en los experimentos unificados previos."""
    x_mezclado, y_mezclado = sklearn_shuffle(
        dataset.bundle.X,
        dataset.bundle.y,
        random_state=semilla,
    )
    bundle = DatasetBundle(
        name=dataset.bundle.name,
        X=x_mezclado,
        y=y_mezclado,
        feature_names=dataset.bundle.feature_names,
        class_names=dataset.bundle.class_names,
    )
    return DatasetCargado(
        bundle=bundle,
        cardinalidad_objetivo=dataset.cardinalidad_objetivo,
        categoria=dataset.categoria,
    )


def _cargar_sintetico(
    nombre: str,
    regla: dict[str, Any],
    semilla_sinteticos: int | None,
) -> DatasetCargado:
    k = int(regla.get("k", 4))
    puntos = regla.get("puntos_por_cluster", 1000)
    if isinstance(puntos, int):
        cardinalidades = tuple([int(puntos)] * k)
    else:
        cardinalidades = tuple(int(v) for v in puntos)
        if len(cardinalidades) != k:
            raise ValueError(f"{nombre}: puntos_por_cluster debe tener largo k.")

    rng = np.random.default_rng(semilla_sinteticos)
    x_partes: list[np.ndarray] = []
    y_partes: list[np.ndarray] = []

    es_2d = "2d" in nombre.lower()
    es_solapado = "over" in nombre.lower()
    separacion = 5.0 if es_solapado else 7.0

    for clase, cantidad in enumerate(cardinalidades):
        if es_2d:
            media = np.array([1.0 + separacion * clase, 3.0], dtype=float)
            x_clase = rng.multivariate_normal(media, np.eye(2), size=cantidad)
        else:
            media = 1.0 + separacion * clase
            x_clase = rng.normal(loc=media, scale=2.0, size=(cantidad, 1))
        x_partes.append(x_clase)
        y_partes.append(np.full(cantidad, clase, dtype=int))

    x_values = np.vstack(x_partes).astype(float)
    y_values = np.concatenate(y_partes).astype(int)
    feature_names = tuple(f"x{i + 1}" for i in range(x_values.shape[1]))
    class_names = tuple(str(i) for i in range(k))
    bundle = DatasetBundle(nombre, x_values, y_values, feature_names, class_names)
    return DatasetCargado(bundle, cardinalidades, clasificar_por_tamano(len(y_values)))


def _asegurar_embedding(nombre: str, regla: dict[str, Any]) -> None:
    ruta = RAIZ_DATASETS / str(regla["archivo"])
    if ruta.exists():
        return

    fuente = str(regla["fuente"])
    ruta_fuente = RAIZ_DATOS / "fuentes" / fuente
    if not ruta_fuente.exists():
        raise FileNotFoundError(
            f"{nombre}: no existe el CSV de embeddings {ruta} y tampoco existe la fuente para generarlo: {ruta_fuente}"
        )
    if str(regla["tipo"]) == "embedding" and not _contiene_imagenes(ruta_fuente):
        raise FileNotFoundError(
            f"{nombre}: no existe el CSV de embeddings {ruta} y la fuente no contiene imagenes validas: {ruta_fuente}"
        )

    print(f"Generando embeddings para {nombre} desde {ruta_fuente}...")
    comando = [
        sys.executable,
        str(SCRIPT_EMBEDDINGS),
        "imagenes" if str(regla["tipo"]) == "embedding" else "texto",
        "--nombre",
        ruta.stem,
        "--fuente",
        fuente,
        "--sobrescribir",
    ]
    if str(regla["tipo"]) == "embedding_texto":
        comando.extend(
            [
                "--columna-texto",
                str(regla["texto"]),
                "--columna-etiqueta",
                str(regla["etiqueta_fuente"]),
            ],
        )

    subprocess.run(comando, cwd=str(RAIZ_PROYECTO), check=True)


def _contiene_imagenes(ruta: Path) -> bool:
    return any(
        item.is_file() and item.suffix.lower() in EXTENSIONES_IMAGEN
        for item in ruta.rglob("*")
    )


def _cargar_csv(nombre: str, regla: dict[str, Any]) -> DatasetCargado:
    ruta = RAIZ_DATASETS / str(regla["archivo"])
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo del dataset: {ruta}")

    opciones = {"index_col": False} if regla.get("index_col_false", False) else {}
    df = pd.read_csv(ruta, sep=str(regla.get("separador", ",")), **opciones)
    df = _limpiar_columnas_vacias(df)

    etiqueta = str(regla["etiqueta"])
    if etiqueta not in df.columns:
        raise ValueError(f"{nombre}: no existe la columna etiqueta '{etiqueta}'.")

    x_df = _seleccionar_x(df, regla)
    if etiqueta in x_df.columns:
        x_df = x_df.drop(columns=[etiqueta])

    y_raw = df[etiqueta]
    x_num = _convertir_x_a_numerico(x_df)
    if bool(regla.get("imputar", False)):
        x_values = SimpleImputer(strategy="median").fit_transform(x_num)
    else:
        x_values = x_num.to_numpy(dtype=float)

    validos = ~np.isnan(x_values).any(axis=1)
    x_values = x_values[validos]
    y_raw = y_raw.iloc[np.flatnonzero(validos)]

    codificador = LabelEncoder()
    y_values = codificador.fit_transform(y_raw.astype(str))
    class_names = tuple(str(v) for v in codificador.classes_)
    feature_names = tuple(str(c) for c in x_num.columns)
    cardinalidad = tuple(int(v) for v in np.bincount(y_values, minlength=len(class_names)))

    bundle = DatasetBundle(nombre, x_values.astype(float), y_values.astype(int), feature_names, class_names)
    return DatasetCargado(bundle, cardinalidad, clasificar_por_tamano(len(y_values)))


def _limpiar_columnas_vacias(df: pd.DataFrame) -> pd.DataFrame:
    columnas = [
        col
        for col in df.columns
        if str(col).strip() and not str(col).strip().lower().startswith("unnamed:")
    ]
    return df.loc[:, columnas].copy()


def _seleccionar_x(df: pd.DataFrame, regla: dict[str, Any]) -> pd.DataFrame:
    spec = dict(regla.get("x", {}))
    if "columnas" in spec:
        return df[list(spec["columnas"])].copy()
    if "prefijo" in spec:
        prefijo = str(spec["prefijo"])
        columnas = [c for c in df.columns if str(c).startswith(prefijo)]
        if not columnas:
            raise ValueError(f"No hay columnas con prefijo {prefijo}.")
        return df[columnas].copy()
    if "rango" in spec:
        inicio, fin = spec["rango"]
        return df.iloc[:, int(inicio) - 1 : int(fin)].copy()
    if "excluir" in spec:
        excluir = set(spec["excluir"])
        columnas = []
        for idx, col in enumerate(df.columns, start=1):
            if idx in excluir or col in excluir:
                continue
            columnas.append(col)
        return df[columnas].copy()
    return df.drop(columns=[str(regla["etiqueta"])]).copy()


def _convertir_x_a_numerico(x_df: pd.DataFrame) -> pd.DataFrame:
    columnas: dict[str, pd.Series] = {}
    for col in x_df.columns:
        serie = x_df[col]
        if pd.api.types.is_numeric_dtype(serie):
            columnas[str(col)] = pd.to_numeric(serie, errors="coerce")
        else:
            valores = serie.astype("string").fillna("<NA>")
            codigos, _ = pd.factorize(valores, sort=True)
            columnas[str(col)] = pd.Series(codigos, index=serie.index, dtype=float)
    x_num = pd.DataFrame(columnas, index=x_df.index)
    validas = [col for col in x_num.columns if not x_num[col].isna().all()]
    if not validas:
        raise ValueError("No quedaron columnas X validas.")
    return x_num[validas]
