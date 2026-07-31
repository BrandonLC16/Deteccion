"""Normalización geométrica de landmarks de una mano."""

from collections.abc import Sequence

import numpy as np

HAND_LANDMARK_COUNT = 21
COORDINATE_COUNT = 3
EXPECTED_LANDMARK_SHAPE = (HAND_LANDMARK_COUNT, COORDINATE_COUNT)
WRIST_LANDMARK_INDEX = 0
DEFAULT_MINIMUM_SCALE = 1e-8


class LandmarkNormalizationError(ValueError):
    """Indica que los landmarks no pueden normalizarse de forma segura."""


class LandmarkNormalizer:
    """Centra, escala y canonicaliza la lateralidad de una mano.

    La muñeca se utiliza como origen y la mayor distancia euclidiana entre la
    muñeca y otro landmark se utiliza como escala. Cuando mirror_left_hand está
    activo, las manos izquierdas se reflejan en el eje X para hacerlas
    comparables con manos derechas de la misma geometría.
    """

    def __init__(
        self,
        mirror_left_hand: bool = True,
        minimum_scale: float = DEFAULT_MINIMUM_SCALE,
    ) -> None:
        if type(mirror_left_hand) is not bool:
            raise LandmarkNormalizationError(
                "mirror_left_hand debe ser un valor booleano."
            )
        if (
            isinstance(minimum_scale, bool)
            or not isinstance(minimum_scale, (int, float))
            or not np.isfinite(minimum_scale)
            or minimum_scale <= 0
        ):
            raise LandmarkNormalizationError(
                "minimum_scale debe ser un número finito mayor que cero."
            )

        self._mirror_left_hand = mirror_left_hand
        self._minimum_scale = float(minimum_scale)

    def normalize(
        self,
        landmarks: np.ndarray | Sequence[Sequence[float]],
        handedness: str | None = None,
    ) -> np.ndarray:
        """Devuelve landmarks (21, 3) invariantes a traslación y escala.

        Args:
            landmarks: Coordenadas X, Y, Z de los 21 landmarks de MediaPipe.
            handedness: Lateralidad Left o Right informada por MediaPipe.

        Raises:
            LandmarkNormalizationError: Si la forma, lateralidad, escala o los
                valores de entrada no permiten una normalización segura.
        """
        values = self._as_valid_array(landmarks)
        normalized_handedness = self._normalize_handedness(handedness)

        centered = values - values[WRIST_LANDMARK_INDEX]
        distances = np.linalg.norm(centered[1:], axis=1)
        scale = float(np.max(distances))
        if not np.isfinite(scale) or scale <= self._minimum_scale:
            raise LandmarkNormalizationError(
                "No se pueden normalizar landmarks con escala nula o demasiado pequeña."
            )

        normalized = centered / scale
        if self._mirror_left_hand and normalized_handedness == "left":
            normalized[:, 0] *= -1.0

        result = np.ascontiguousarray(normalized, dtype=np.float32)
        if not np.isfinite(result).all():
            raise LandmarkNormalizationError(
                "La normalización produjo valores NaN o infinitos."
            )
        result.setflags(write=False)
        return result

    @staticmethod
    def _as_valid_array(
        landmarks: np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        try:
            values = np.asarray(landmarks, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise LandmarkNormalizationError(
                "Los landmarks deben contener únicamente valores numéricos."
            ) from exc

        if values.shape != EXPECTED_LANDMARK_SHAPE:
            raise LandmarkNormalizationError(
                "Los landmarks deben tener forma "
                f"{EXPECTED_LANDMARK_SHAPE}; se recibió {values.shape}."
            )
        if not np.isfinite(values).all():
            raise LandmarkNormalizationError(
                "Los landmarks no pueden contener valores NaN o infinitos."
            )
        return values

    @staticmethod
    def _normalize_handedness(handedness: str | None) -> str | None:
        if handedness is None:
            return None
        if not isinstance(handedness, str):
            raise LandmarkNormalizationError(
                "La lateralidad debe ser Left, Right o None."
            )

        normalized = handedness.strip().casefold()
        if normalized not in {"left", "right"}:
            raise LandmarkNormalizationError(
                f"Lateralidad no válida: {handedness!r}. Usa Left, Right o None."
            )
        return normalized
