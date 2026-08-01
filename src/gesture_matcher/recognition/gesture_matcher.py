"""Motor de reconocimiento de señas mediante similitud coseno."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gesture_matcher.recognition.template_repository import (
    FEATURE_DIMENSIONS_BY_HAND_COUNT,
    GestureTemplate,
    TemplateRepository,
)

LOGGER = logging.getLogger(__name__)
UNKNOWN_GESTURE_LABEL = "Pose desconocida"


@dataclass(frozen=True)
class MatchResult:
    """Resultado aceptado o desconocido con su mejor puntuación."""

    gesture_id: str | None
    label: str
    similarity: float
    display_image_path: str | None
    accepted: bool


@dataclass(frozen=True)
class GestureScore:
    """Puntuación ordenable de la mejor muestra de una seña."""

    gesture_id: str
    label: str
    similarity: float
    display_image_path: str | None
    threshold: float
    best_sample_index: int
    hand_count: int


class GestureMatcher:
    """Compara un vector en vivo contra todas las muestras compatibles."""

    def __init__(
        self,
        repository: TemplateRepository,
        *,
        similarity_threshold: float | None = None,
        gesture_thresholds: Mapping[str, float] | None = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._repository = repository
        self._similarity_threshold = _validate_threshold(
            repository.default_similarity_threshold
            if similarity_threshold is None
            else similarity_threshold,
            "similarity_threshold",
        )
        self._gesture_thresholds = {
            gesture_id: _validate_threshold(
                threshold,
                f"gesture_thresholds.{gesture_id}",
            )
            for gesture_id, threshold in (gesture_thresholds or {}).items()
        }
        if any(
            not isinstance(gesture_id, str) or not gesture_id.strip()
            for gesture_id in self._gesture_thresholds
        ):
            raise ValueError(
                "Cada identificador de gesture_thresholds debe ser texto no vacío."
            )
        self._logger = logger

    def match(self, features: np.ndarray) -> MatchResult:
        """Devuelve la mejor coincidencia o un resultado desconocido seguro."""
        ranked = self.rank(features)
        if not ranked:
            return self._unknown_result(0.0)

        best = ranked[0]
        if best.similarity < best.threshold:
            return self._unknown_result(best.similarity)
        return MatchResult(
            gesture_id=best.gesture_id,
            label=best.label,
            similarity=best.similarity,
            display_image_path=best.display_image_path,
            accepted=True,
        )

    def rank(self, features: np.ndarray) -> tuple[GestureScore, ...]:
        """Puntúa y ordena todas las señas con la misma cantidad de manos."""
        prepared = self._prepare_features(features)
        if prepared is None:
            return ()
        vector, vector_norm, hand_count = prepared

        scores: list[GestureScore] = []
        for template in self._repository.for_hand_count(hand_count):
            if template.feature_dimension != vector.size:
                continue
            similarities = self._cosine_similarities(
                vector,
                vector_norm,
                template,
            )
            best_sample_index = int(np.argmax(similarities))
            best_similarity = float(similarities[best_sample_index])
            scores.append(
                GestureScore(
                    gesture_id=template.gesture_id,
                    label=template.label,
                    similarity=best_similarity,
                    display_image_path=template.display_image_path,
                    threshold=self._threshold_for(template),
                    best_sample_index=best_sample_index,
                    hand_count=template.hand_count,
                )
            )

        return tuple(
            sorted(
                scores,
                key=lambda score: (-score.similarity, score.gesture_id),
            )
        )

    def _cosine_similarities(
        self,
        vector: np.ndarray,
        vector_norm: float,
        template: GestureTemplate,
    ) -> np.ndarray:
        template_norms = np.linalg.norm(template.feature_vectors, axis=1)
        similarities = (template.feature_vectors @ vector) / (
            template_norms * vector_norm
        )
        similarities = np.clip(similarities, -1.0, 1.0)
        for sample_index, similarity in enumerate(similarities):
            self._logger.debug(
                "Puntuación gesture_id=%s sample=%d hand_count=%d similarity=%.6f",
                template.gesture_id,
                sample_index,
                template.hand_count,
                similarity,
            )
        return similarities

    def _threshold_for(self, template: GestureTemplate) -> float:
        configured = self._gesture_thresholds.get(template.gesture_id)
        if configured is not None:
            return configured
        if template.similarity_threshold is not None:
            return template.similarity_threshold
        return self._similarity_threshold

    def _prepare_features(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, float, int] | None:
        try:
            vector = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            self._logger.debug("Vector en vivo no numérico; se devuelve desconocido.")
            return None

        if vector.ndim != 1 or vector.size == 0:
            self._logger.debug(
                "Vector en vivo vacío o no unidimensional; se devuelve desconocido."
            )
            return None
        if not np.isfinite(vector).all():
            self._logger.debug(
                "Vector en vivo con NaN o infinitos; se devuelve desconocido."
            )
            return None

        hand_count = _hand_count_for_dimension(int(vector.size))
        if hand_count is None:
            self._logger.debug(
                "Dimensión de vector no compatible: %d.",
                vector.size,
            )
            return None

        vector_norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector_norm) or vector_norm <= np.finfo(np.float64).eps:
            self._logger.debug(
                "Vector en vivo con norma nula o inválida; se devuelve desconocido."
            )
            return None
        return vector, vector_norm, hand_count

    @staticmethod
    def _unknown_result(similarity: float) -> MatchResult:
        return MatchResult(
            gesture_id=None,
            label=UNKNOWN_GESTURE_LABEL,
            similarity=float(similarity),
            display_image_path=None,
            accepted=False,
        )


def _hand_count_for_dimension(feature_dimension: int) -> int | None:
    for hand_count, expected_dimension in FEATURE_DIMENSIONS_BY_HAND_COUNT.items():
        if feature_dimension == expected_dimension:
            return hand_count
    return None


def _validate_threshold(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} debe ser un número entre 0 y 1.")
    converted = float(value)
    if not np.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{field_name} debe estar entre 0 y 1.")
    return converted
