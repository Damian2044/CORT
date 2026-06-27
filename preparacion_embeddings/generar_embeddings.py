from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from laion_clip_batch import ExtractorLaionCLIPBatch


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class EntradaImagen:
    ruta: Path
    etiqueta: str
    fuente: str


def clasificar_por_tamano(n_filas: int) -> str:
    if n_filas <= 2120:
        return "pequenos"
    if n_filas <= 4000:
        return "medianos"
    if n_filas <= 6500:
        return "grandes"
    return "extra_grandes"


def raiz_datos() -> Path:
    return Path(__file__).resolve().parent.parent / "datos"


def raiz_fuentes() -> Path:
    return raiz_datos() / "fuentes"


def chunks(items: list, batch_size: int):
    for inicio in range(0, len(items), batch_size):
        yield items[inicio : inicio + batch_size]


def columnas_embedding(dimension: int) -> list[str]:
    return [f"v_{idx}" for idx in range(dimension)]


def resolver_salida(nombre_dataset: str, n_filas: int) -> Path:
    carpeta = raiz_datos() / "datasets" / clasificar_por_tamano(n_filas)
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta / f"{nombre_dataset}.csv"


def ruta_errores(ruta_salida: Path) -> Path:
    return ruta_salida.with_name(f"{ruta_salida.stem}_errores.csv")


def listar_imagenes(origen: Path) -> list[EntradaImagen]:
    entradas: list[EntradaImagen] = []
    for carpeta_clase in sorted(origen.iterdir(), key=lambda path: path.name.lower()):
        if not carpeta_clase.is_dir():
            continue
        etiqueta = carpeta_clase.name
        for ruta in sorted(carpeta_clase.rglob("*"), key=lambda path: str(path).lower()):
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN:
                entradas.append(
                    EntradaImagen(
                        ruta=ruta,
                        etiqueta=etiqueta,
                        fuente=str(ruta.relative_to(raiz_fuentes())),
                    )
                )
    return entradas


def guardar_csv_embeddings(
    ruta_salida: Path,
    filas_base: list[dict[str, str]],
    embeddings,
) -> None:
    if len(filas_base) != len(embeddings):
        raise ValueError("La cantidad de filas y embeddings no coincide.")

    columnas = columnas_embedding(int(embeddings.shape[1]))
    filas = []
    for base, vector in zip(filas_base, embeddings):
        fila = dict(base)
        fila.update({columna: float(valor) for columna, valor in zip(columnas, vector)})
        filas.append(fila)

    pd.DataFrame(filas).to_csv(ruta_salida, index=False)


def generar_desde_imagenes(
    nombre_dataset: str,
    fuente_relativa: str,
    batch_size: int,
    sobrescribir: bool,
) -> Path:
    fuente = raiz_fuentes() / fuente_relativa
    entradas = listar_imagenes(fuente)
    if not entradas:
        raise ValueError(f"No se encontraron imagenes en {fuente}.")

    ruta_salida = resolver_salida(nombre_dataset, len(entradas))
    if ruta_salida.exists() and not sobrescribir:
        return ruta_salida

    extractor = ExtractorLaionCLIPBatch()
    filas_ok: list[dict[str, str]] = []
    vectores = []
    errores: list[dict[str, str]] = []

    total_lotes = math.ceil(len(entradas) / batch_size)
    for lote in tqdm(chunks(entradas, batch_size), total=total_lotes, desc=f"Generando embeddings {nombre_dataset}", unit="lote"):
        lote_ok: list[EntradaImagen] = []
        for entrada in lote:
            try:
                # Abre antes del batch para registrar errores por archivo sin perder el lote completo.
                extractor._to_pil(entrada.ruta)
                lote_ok.append(entrada)
            except Exception as exc:
                errores.append(
                    {
                        "fuente": entrada.fuente,
                        "etiqueta": entrada.etiqueta,
                        "estado": "error",
                        "error": str(exc),
                    }
                )

        if not lote_ok:
            continue

        try:
            embeddings = extractor.extraer_embeddings_imagenes([item.ruta for item in lote_ok])
            vectores.append(embeddings)
            filas_ok.extend(
                {
                    "fuente": item.fuente,
                    "etiqueta": item.etiqueta,
                    "estado": "ok",
                }
                for item in lote_ok
            )
        except Exception as exc:
            for item in lote_ok:
                errores.append(
                    {
                        "fuente": item.fuente,
                        "etiqueta": item.etiqueta,
                        "estado": "error",
                        "error": str(exc),
                    }
                )

    if not vectores:
        raise RuntimeError(f"No se pudo generar ningun embedding para {nombre_dataset}.")

    import numpy as np

    guardar_csv_embeddings(ruta_salida, filas_ok, np.vstack(vectores))
    if errores:
        pd.DataFrame(errores).to_csv(ruta_errores(ruta_salida), index=False)
    return ruta_salida


def generar_desde_csv_texto(
    nombre_dataset: str,
    fuente_relativa: str,
    columna_texto: str,
    columna_etiqueta: str,
    separador: str,
    batch_size: int,
    sobrescribir: bool,
) -> Path:
    fuente = raiz_fuentes() / fuente_relativa
    df = pd.read_csv(fuente, sep=separador)
    if columna_texto not in df.columns:
        raise ValueError(f"No existe la columna de texto: {columna_texto}")
    if columna_etiqueta not in df.columns:
        raise ValueError(f"No existe la columna de etiqueta: {columna_etiqueta}")

    ruta_salida = resolver_salida(nombre_dataset, len(df))
    if ruta_salida.exists() and not sobrescribir:
        return ruta_salida

    extractor = ExtractorLaionCLIPBatch()
    filas_ok: list[dict[str, str]] = []
    vectores = []
    errores: list[dict[str, str]] = []

    registros = list(df[[columna_texto, columna_etiqueta]].itertuples(index=True, name=None))
    total_lotes = math.ceil(len(registros) / batch_size)
    for lote in tqdm(chunks(registros, batch_size), total=total_lotes, desc=f"Generando embeddings {nombre_dataset}", unit="lote"):
        lote_ok = []
        for indice, texto, etiqueta in lote:
            texto_limpio = "" if pd.isna(texto) else str(texto).strip()
            if not texto_limpio:
                errores.append(
                    {
                        "fuente": f"fila_{indice}",
                        "etiqueta": str(etiqueta),
                        "estado": "error",
                        "error": "texto vacio",
                    }
                )
            else:
                lote_ok.append((indice, texto_limpio, etiqueta))

        if not lote_ok:
            continue

        try:
            embeddings = extractor.extraer_embeddings_textos([item[1] for item in lote_ok])
            vectores.append(embeddings)
            filas_ok.extend(
                {
                    "fuente": f"fila_{indice}",
                    "etiqueta": str(etiqueta),
                    "estado": "ok",
                }
                for indice, _, etiqueta in lote_ok
            )
        except Exception as exc:
            for indice, _, etiqueta in lote_ok:
                errores.append(
                    {
                        "fuente": f"fila_{indice}",
                        "etiqueta": str(etiqueta),
                        "estado": "error",
                        "error": str(exc),
                    }
                )

    if not vectores:
        raise RuntimeError(f"No se pudo generar ningun embedding para {nombre_dataset}.")

    import numpy as np

    guardar_csv_embeddings(ruta_salida, filas_ok, np.vstack(vectores))
    if errores:
        pd.DataFrame(errores).to_csv(ruta_errores(ruta_salida), index=False)
    return ruta_salida


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera embeddings LAION-CLIP por lotes.")
    subparsers = parser.add_subparsers(dest="comando", required=True)

    imagenes = subparsers.add_parser("imagenes")
    imagenes.add_argument("--nombre", required=True)
    imagenes.add_argument("--fuente", required=True)
    imagenes.add_argument("--batch-size", type=int, default=64)
    imagenes.add_argument("--sobrescribir", action="store_true")

    texto = subparsers.add_parser("texto")
    texto.add_argument("--nombre", required=True)
    texto.add_argument("--fuente", required=True)
    texto.add_argument("--columna-texto", required=True)
    texto.add_argument("--columna-etiqueta", required=True)
    texto.add_argument("--separador", default=",")
    texto.add_argument("--batch-size", type=int, default=64)
    texto.add_argument("--sobrescribir", action="store_true")

    return parser


def main() -> None:
    args = construir_parser().parse_args()
    if args.comando == "imagenes":
        salida = generar_desde_imagenes(
            nombre_dataset=args.nombre,
            fuente_relativa=args.fuente,
            batch_size=args.batch_size,
            sobrescribir=args.sobrescribir,
        )
    else:
        salida = generar_desde_csv_texto(
            nombre_dataset=args.nombre,
            fuente_relativa=args.fuente,
            columna_texto=args.columna_texto,
            columna_etiqueta=args.columna_etiqueta,
            separador=args.separador,
            batch_size=args.batch_size,
            sobrescribir=args.sobrescribir,
        )
    print(salida)


if __name__ == "__main__":
    main()
