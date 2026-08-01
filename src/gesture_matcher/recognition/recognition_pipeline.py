"""Coordinación de extracción, comparación y estabilización por fotograma."""

import logging
from collections.abc import Sequence
from typing import Self

from gesture_matcher.recognition.feature_extractor import (
    FeatureExtractionError,
    FeatureExtractor,
)
from gesture_matcher.recognition.gesture_matcher import (
    UNKNOWN_GESTURE_LABEL,
    GestureMatcher,
    MatchResult,
)
from gesture_matcher.recognition.landmark_normalizer import LandmarkNormalizer
from gesture_matcher.recognition.template_repository import TemplateRepository
from gesture_matcher.recognition.temporal_filter import TemporalFilter
from gesture_matcher.utils.config_loader import AppConfig
from gesture_matcher.vision.hand_detector import HandObservation

LOGGER = logging.getLogger(__name__)


class RecognitionPipeline:
    """Transforma manos detectadas en un resultado confirmado para la interfaz."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        matcher: GestureMatcher,
        temporal_filter: TemporalFilter,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._feature_extractor = feature_extractor
        self._matcher = matcher
        self._temporal_filter = temporal_filter
        self._logger = logger

    @classmethod
    def from_config(cls, config: AppConfig) -> Self:
        """Carga una vez las plantillas y crea el pipeline configurado."""
        repository = TemplateRepository.load(
            config.resources.gesture_templates,
            config.resources.gesture_metadata,
            project_root=config.project_root,
        )
        normalizer = LandmarkNormalizer(
            mirror_left_hand=config.recognition.mirror_left_hand
        )
        feature_extractor = FeatureExtractor(normalizer)
        matcher = GestureMatcher(
            repository,
            similarity_threshold=config.recognition.similarity_threshold,
            gesture_thresholds=config.recognition.gesture_thresholds,
        )
        return cls(
            feature_extractor,
            matcher,
            TemporalFilter(config.temporal_filter),
        )

    def recognize(self, hands: Sequence[HandObservation]) -> MatchResult:
        """Devuelve una seña estabilizada para cero, una o dos manos."""
        observations = tuple(hands)
        if not observations:
            return self._temporal_filter.update(_unknown_result())

        try:
            features = self._feature_extractor.extract_hands(observations)
        except FeatureExtractionError as exc:
            self._logger.debug(
                "No se extrajeron características del fotograma: %s",
                exc,
            )
            raw_result = _unknown_result()
        else:
            raw_result = self._matcher.match(features.vector)
        return self._temporal_filter.update(raw_result)

    def reset(self) -> None:
        """Reinicia el historial temporal del reconocimiento."""
        self._temporal_filter.reset()


def _unknown_result() -> MatchResult:
    return MatchResult(
        gesture_id=None,
        label=UNKNOWN_GESTURE_LABEL,
        similarity=0.0,
        display_image_path=None,
        accepted=False,
    )
