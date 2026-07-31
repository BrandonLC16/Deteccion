"""Construye plantillas persistentes desde imágenes de referencia."""

import logging
from collections.abc import Callable
from pathlib import Path

from gesture_matcher.recognition.feature_extractor import FeatureExtractor
from gesture_matcher.recognition.landmark_normalizer import LandmarkNormalizer
from gesture_matcher.recognition.template_builder import (
    TemplateBuilder,
    TemplateBuildError,
    TemplateBuildResult,
    format_build_report,
    save_template_artifacts,
)
from gesture_matcher.utils.config_loader import (
    AppConfig,
    ConfigError,
    HandDetectionConfig,
    load_config,
)
from gesture_matcher.utils.logging_config import configure_logging
from gesture_matcher.vision.hand_detector import HandDetectorError
from gesture_matcher.vision.image_hand_detector import ImageHandDetector

LOGGER = logging.getLogger(__name__)

DetectorFactory = Callable[
    [HandDetectionConfig, Path],
    ImageHandDetector,
]


def run_template_build(
    config: AppConfig,
    *,
    detector_factory: DetectorFactory = ImageHandDetector,
) -> TemplateBuildResult:
    """Procesa las referencias configuradas y escribe NPZ más metadatos JSON."""
    normalizer = LandmarkNormalizer(
        mirror_left_hand=config.recognition.mirror_left_hand
    )
    feature_extractor = FeatureExtractor(normalizer)

    with detector_factory(
        config.hand_detection,
        config.resources.hand_model,
    ) as detector:
        result = TemplateBuilder(detector, feature_extractor).build(
            config.resources.reference_images
        )

    LOGGER.info(
        "\n%s",
        format_build_report(result, project_root=config.project_root),
    )
    save_template_artifacts(
        result,
        templates_path=config.resources.gesture_templates,
        metadata_path=config.resources.gesture_metadata,
        project_root=config.project_root,
        display_images_root=config.resources.display_images,
        default_similarity_threshold=config.recognition.similarity_threshold,
        gesture_thresholds=config.recognition.gesture_thresholds,
        mirror_left_hand=config.recognition.mirror_left_hand,
    )
    LOGGER.info(
        "Plantillas guardadas en %s y metadatos en %s.",
        config.resources.gesture_templates,
        config.resources.gesture_metadata,
    )
    return result


def main() -> int:
    """Carga configuración, construye plantillas y reporta errores accionables."""
    try:
        config = load_config()
    except ConfigError as exc:
        logging.basicConfig(level=logging.ERROR)
        LOGGER.error("No se pudo iniciar la construcción de plantillas: %s", exc)
        return 1

    configure_logging(config.logging)
    try:
        run_template_build(config)
    except KeyboardInterrupt:
        LOGGER.info("Construcción cancelada por el usuario.")
        return 1
    except (HandDetectorError, TemplateBuildError) as exc:
        LOGGER.error("No se pudieron construir las plantillas: %s", exc)
        return 1
    except Exception:
        LOGGER.exception("La construcción terminó por un error inesperado.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
