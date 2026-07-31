"""Extracción de vectores de características geométricas de una mano."""

from collections.abc import Sequence

import numpy as np

from gesture_matcher.recognition.landmark_normalizer import (
    COORDINATE_COUNT,
    HAND_LANDMARK_COUNT,
    LandmarkNormalizationError,
    LandmarkNormalizer,
)

FEATURE_VECTOR_SIZE = HAND_LANDMARK_COUNT * COORDINATE_COUNT


class FeatureExtractionError(ValueError):
    """Indica que no pudo producirse un vector de características válido."""


class FeatureExtractor:
    """Normaliza una mano y aplana sus coordenadas en un vector consistente."""

    def __init__(self, normalizer: LandmarkNormalizer | None = None) -> None:
        self._normalizer = normalizer or LandmarkNormalizer()

    def extract(
        self,
        landmarks: np.ndarray | Sequence[Sequence[float]],
        handedness: str | None = None,
    ) -> np.ndarray:
        """Devuelve un vector (63,) listo para comparación posterior.

        El orden conserva los 21 índices definidos por MediaPipe y, para cada
        landmark, almacena X, Y y Z. El resultado es float32, contiguo e
        inmutable para evitar cambios accidentales después de extraerlo.

        Raises:
            FeatureExtractionError: Si los landmarks no pueden normalizarse o el
                vector resultante no tiene la dimensión esperada.
        """
        try:
            normalized = self._normalizer.normalize(landmarks, handedness)
        except LandmarkNormalizationError as exc:
            raise FeatureExtractionError(
                f"No se pudo extraer el vector de características: {exc}"
            ) from exc

        features = np.ascontiguousarray(normalized.reshape(-1), dtype=np.float32)
        if features.shape != (FEATURE_VECTOR_SIZE,):
            raise FeatureExtractionError(
                "El vector de características debe tener forma "
                f"({FEATURE_VECTOR_SIZE},); se recibió {features.shape}."
            )
        if not np.isfinite(features).all():
            raise FeatureExtractionError(
                "El vector de características contiene valores NaN o infinitos."
            )

        features.setflags(write=False)
        return features
