"""Carga en caché y ajuste proporcional de imágenes de presentación."""

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

ImageLoader = Callable[[str], np.ndarray | None]
ImageResizer = Callable[..., np.ndarray]


class ImageOverlayError(ValueError):
    """Indica que una imagen no puede prepararse para la interfaz."""


class ImageCache:
    """Carga cada imagen de presentación una sola vez y conserva su versión ajustada."""

    def __init__(
        self,
        *,
        project_root: Path,
        target_width: int,
        target_height: int,
        image_loader: ImageLoader = cv2.imread,
        image_resizer: ImageResizer = cv2.resize,
        logger: logging.Logger = LOGGER,
    ) -> None:
        if type(target_width) is not int or target_width < 1:
            raise ImageOverlayError("target_width debe ser un entero positivo.")
        if type(target_height) is not int or target_height < 1:
            raise ImageOverlayError("target_height debe ser un entero positivo.")

        self._project_root = project_root.resolve()
        self._target_width = target_width
        self._target_height = target_height
        self._image_loader = image_loader
        self._image_resizer = image_resizer
        self._logger = logger
        self._images: dict[str, np.ndarray | None] = {}

    def get(self, relative_path: str | None) -> np.ndarray | None:
        """Devuelve la imagen ajustada o ``None`` sin repetir lecturas fallidas."""
        if relative_path is None:
            return None
        if not isinstance(relative_path, str) or not relative_path.strip():
            self._logger.warning("La ruta de imagen asociada está vacía o no es texto.")
            return None

        cache_key = relative_path.strip().replace("\\", "/")
        if cache_key in self._images:
            return self._images[cache_key]

        resolved_path = self._resolve_path(cache_key)
        if resolved_path is None or not resolved_path.is_file():
            self._logger.warning(
                "No se encontró la imagen de presentación configurada: %s",
                cache_key,
            )
            self._images[cache_key] = None
            return None

        try:
            image = self._image_loader(str(resolved_path))
            prepared = resize_to_fit(
                image,
                target_width=self._target_width,
                target_height=self._target_height,
                image_resizer=self._image_resizer,
            )
        except (
            ImageOverlayError,
            OSError,
            RuntimeError,
            ValueError,
            cv2.error,
        ) as exc:
            self._logger.warning(
                "No se pudo cargar la imagen de presentación %s: %s",
                cache_key,
                exc,
            )
            prepared = None

        self._images[cache_key] = prepared
        return prepared

    def clear(self) -> None:
        """Elimina las imágenes y fallos almacenados en memoria."""
        self._images.clear()

    def _resolve_path(self, cache_key: str) -> Path | None:
        path = Path(cache_key)
        if path.is_absolute():
            self._logger.warning(
                "La imagen de presentación debe usar una ruta relativa: %s",
                cache_key,
            )
            return None
        resolved = (self._project_root / path).resolve()
        if not resolved.is_relative_to(self._project_root):
            self._logger.warning(
                "La imagen de presentación sale de la raíz del proyecto: %s",
                cache_key,
            )
            return None
        return resolved


def resize_to_fit(
    image: np.ndarray | None,
    *,
    target_width: int,
    target_height: int,
    image_resizer: ImageResizer = cv2.resize,
) -> np.ndarray:
    """Ajusta una imagen dentro del área indicada sin deformar su proporción."""
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
    ):
        raise ImageOverlayError("La imagen asociada debe tener forma (alto, ancho, 3).")
    if type(target_width) is not int or target_width < 1:
        raise ImageOverlayError("target_width debe ser un entero positivo.")
    if type(target_height) is not int or target_height < 1:
        raise ImageOverlayError("target_height debe ser un entero positivo.")

    source_height, source_width = image.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = image_resizer(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    if not isinstance(resized, np.ndarray) or resized.shape != (
        resized_height,
        resized_width,
        3,
    ):
        raise ImageOverlayError("OpenCV devolvió una imagen redimensionada inválida.")

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = resized
    canvas.setflags(write=False)
    return canvas
