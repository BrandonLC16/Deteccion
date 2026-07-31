"""Extracción de vectores de características geométricas de una mano."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gesture_matcher.recognition.landmark_normalizer import (
    COORDINATE_COUNT,
    HAND_LANDMARK_COUNT,
    LandmarkNormalizationError,
    LandmarkNormalizer,
)
from gesture_matcher.vision.hand_detector import HandObservation

FEATURE_VECTOR_SIZE = HAND_LANDMARK_COUNT * COORDINATE_COUNT
CANONICAL_HAND_ORDER = ("Left", "Right")
TWO_HAND_RELATIVE_FEATURE_SIZE = COORDINATE_COUNT


class FeatureExtractionError(ValueError):
    """Indica que no pudo producirse un vector de características válido."""


@dataclass(frozen=True)
class HandFeatureVector:
    """Vector combinado y esquema canónico de una o dos manos."""

    vector: np.ndarray
    hand_count: int
    handedness: tuple[str, ...]


class FeatureExtractor:
    """Extrae vectores consistentes de una o dos manos."""

    def __init__(self, normalizer: LandmarkNormalizer | None = None) -> None:
        self._normalizer = (
            normalizer if normalizer is not None else LandmarkNormalizer()
        )

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

    def extract_hands(
        self,
        hands: Sequence[HandObservation],
    ) -> HandFeatureVector:
        """Combina una o dos manos usando el orden canónico Left seguido de Right.

        Para dos manos se agregan tres características con el desplazamiento de
        la muñeca derecha respecto de la izquierda. El desplazamiento se divide
        por la escala media de ambas manos para mantener invariancia a traslación
        y escala uniforme.

        Raises:
            FeatureExtractionError: Si no hay una o dos manos válidas, falta su
                lateralidad o dos manos no forman el par Left/Right.
        """
        ordered_hands, handedness = self._canonicalize_hands(hands)
        vectors = [
            self.extract(hand.landmarks, label)
            for hand, label in zip(ordered_hands, handedness, strict=True)
        ]

        if len(ordered_hands) == 2:
            vectors.append(self._relative_wrist_features(ordered_hands))

        combined = np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32)
        expected_size = FEATURE_VECTOR_SIZE * len(ordered_hands)
        if len(ordered_hands) == 2:
            expected_size += TWO_HAND_RELATIVE_FEATURE_SIZE
        if combined.shape != (expected_size,) or not np.isfinite(combined).all():
            raise FeatureExtractionError(
                "El vector combinado de manos tiene valores o dimensiones inválidos."
            )

        combined.setflags(write=False)
        return HandFeatureVector(
            vector=combined,
            hand_count=len(ordered_hands),
            handedness=handedness,
        )

    @staticmethod
    def _canonicalize_hands(
        hands: Sequence[HandObservation],
    ) -> tuple[tuple[HandObservation, ...], tuple[str, ...]]:
        observations = tuple(hands)
        if len(observations) not in {1, 2}:
            raise FeatureExtractionError(
                "La extracción requiere exactamente una o dos manos."
            )

        normalized_labels: list[str] = []
        for hand in observations:
            label = hand.handedness
            if not isinstance(label, str):
                raise FeatureExtractionError(
                    "Cada mano debe incluir lateralidad Left o Right."
                )
            normalized = label.strip().casefold()
            if normalized not in {"left", "right"}:
                raise FeatureExtractionError(
                    f"Lateralidad no válida para extracción: {label!r}."
                )
            normalized_labels.append(normalized.title())

        if len(observations) == 2 and set(normalized_labels) != set(
            CANONICAL_HAND_ORDER
        ):
            raise FeatureExtractionError(
                "Una seña de dos manos requiere exactamente una mano Left y una Right."
            )

        order = {label: index for index, label in enumerate(CANONICAL_HAND_ORDER)}
        paired = sorted(
            zip(observations, normalized_labels, strict=True),
            key=lambda item: order[item[1]],
        )
        return (
            tuple(item[0] for item in paired),
            tuple(item[1] for item in paired),
        )

    @staticmethod
    def _relative_wrist_features(
        hands: tuple[HandObservation, HandObservation],
    ) -> np.ndarray:
        scales: list[float] = []
        wrist_coordinates: list[np.ndarray] = []
        for hand in hands:
            landmarks = np.asarray(hand.landmarks, dtype=np.float64)
            centered = landmarks - landmarks[0]
            scale = float(np.max(np.linalg.norm(centered[1:], axis=1)))
            if not np.isfinite(scale) or scale <= 0:
                raise FeatureExtractionError(
                    "No se pudo calcular la escala relativa entre las dos manos."
                )
            scales.append(scale)
            wrist_coordinates.append(landmarks[0])

        shared_scale = float(np.mean(scales))
        relative = (wrist_coordinates[1] - wrist_coordinates[0]) / shared_scale
        result = np.ascontiguousarray(relative, dtype=np.float32)
        if (
            result.shape != (TWO_HAND_RELATIVE_FEATURE_SIZE,)
            or not np.isfinite(result).all()
        ):
            raise FeatureExtractionError(
                "El desplazamiento relativo de muñecas no es válido."
            )
        result.setflags(write=False)
        return result
