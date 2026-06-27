from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import open_clip


class ExtractorLaionCLIPBatch:
    """Extractor LAION-CLIP con soporte por lotes para imagenes y texto."""

    def __init__(
        self,
        usar_gpu: bool = True,
        modelo: str = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        normalizar: bool = True,
    ) -> None:
        self.modelo_nombre = modelo
        self.normalizar = bool(normalizar)
        self.dispositivo = (
            torch.device("cuda")
            if usar_gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.modelo, _, self.preprocesamiento = open_clip.create_model_and_transforms(
            self.modelo_nombre
        )
        self.modelo = self.modelo.to(self.dispositivo).eval()
        self.tokenizer = open_clip.get_tokenizer(self.modelo_nombre)

    def _to_pil(self, imagen) -> Image.Image:
        if isinstance(imagen, Image.Image):
            return imagen.convert("RGB")

        if isinstance(imagen, (str, Path)):
            return Image.open(imagen).convert("RGB")

        if isinstance(imagen, bytes):
            return Image.open(BytesIO(imagen)).convert("RGB")

        if isinstance(imagen, np.ndarray):
            if imagen.ndim not in (2, 3):
                raise ValueError("Array de imagen invalido.")

            if imagen.ndim == 3:
                if imagen.shape[2] == 4:
                    imagen = imagen[:, :, :3]
                elif imagen.shape[2] != 3:
                    raise ValueError("La imagen debe tener 3 o 4 canales.")

            if np.issubdtype(imagen.dtype, np.floating):
                minimo = float(imagen.min()) if imagen.size > 0 else 0.0
                maximo = float(imagen.max()) if imagen.size > 0 else 0.0
                if 0.0 <= minimo and maximo <= 1.0:
                    imagen = (imagen * 255).clip(0, 255).astype(np.uint8)
                else:
                    imagen = imagen.clip(0, 255).astype(np.uint8)
            elif imagen.dtype != np.uint8:
                imagen = imagen.clip(0, 255).astype(np.uint8)

            return Image.fromarray(imagen).convert("RGB")

        raise ValueError(f"Tipo de imagen no soportado: {type(imagen)}")

    def _procesar_embedding(self, embedding: torch.Tensor) -> np.ndarray:
        if self.normalizar:
            embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return embedding.detach().cpu().float().numpy().astype(np.float32)

    def extraer_embeddings_imagenes(self, imagenes: Iterable[Path]) -> np.ndarray:
        tensores = []
        for imagen in imagenes:
            imagen_pil = self._to_pil(imagen)
            tensores.append(self.preprocesamiento(imagen_pil))

        if not tensores:
            return np.empty((0, 0), dtype=np.float32)

        batch = torch.stack(tensores).to(self.dispositivo)
        with torch.inference_mode():
            embedding = self.modelo.encode_image(batch)
        return self._procesar_embedding(embedding)

    def extraer_embeddings_textos(self, textos: Iterable[str]) -> np.ndarray:
        textos_limpios = [str(texto).strip() for texto in textos]
        if not textos_limpios:
            return np.empty((0, 0), dtype=np.float32)

        tokens = self.tokenizer(textos_limpios).to(self.dispositivo)
        with torch.inference_mode():
            embedding = self.modelo.encode_text(tokens)
        return self._procesar_embedding(embedding)

