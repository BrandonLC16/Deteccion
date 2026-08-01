"""Punto de entrada y composición de dependencias de la aplicación."""

import logging
from collections.abc import Callable
from pathlib import Path

from gesture_matcher.camera.camera_service import CameraError, CameraService
from gesture_matcher.recognition.recognition_pipeline import RecognitionPipeline
from gesture_matcher.recognition.template_repository import TemplateRepositoryError
from gesture_matcher.ui.image_overlay import ImageCache, ImageOverlayError
from gesture_matcher.ui.opencv_view import OpenCVView, OpenCVViewError
from gesture_matcher.utils.config_loader import (
    AppConfig,
    CameraConfig,
    ConfigError,
    DisplayConfig,
    HandDetectionConfig,
    load_config,
)
from gesture_matcher.utils.logging_config import configure_logging
from gesture_matcher.vision.hand_detector import (
    HandDetector,
    HandDetectorError,
)
from gesture_matcher.vision.landmark_drawer import (
    LandmarkDrawer,
    LandmarkDrawingError,
)

LOGGER = logging.getLogger(__name__)

CameraFactory = Callable[[CameraConfig], CameraService]
DetectorFactory = Callable[[HandDetectionConfig, Path], HandDetector]
ViewFactory = Callable[[DisplayConfig, ImageCache], OpenCVView]
DrawerFactory = Callable[[], LandmarkDrawer]
RecognitionFactory = Callable[[AppConfig], RecognitionPipeline]
ImageCacheFactory = Callable[..., ImageCache]


def run_application(
    config: AppConfig,
    *,
    camera_factory: CameraFactory = CameraService,
    detector_factory: DetectorFactory = HandDetector,
    view_factory: ViewFactory = OpenCVView,
    drawer_factory: DrawerFactory = LandmarkDrawer,
    recognition_factory: RecognitionFactory = RecognitionPipeline.from_config,
    image_cache_factory: ImageCacheFactory = ImageCache,
) -> None:
    """Conecta detección, reconocimiento estable y presentación de video."""
    recognition = recognition_factory(config)
    image_cache = image_cache_factory(
        project_root=config.project_root,
        target_width=config.display.result_image_width,
        target_height=config.display.result_image_height,
    )
    with (
        detector_factory(
            config.hand_detection,
            config.resources.hand_model,
        ) as detector,
        camera_factory(config.camera) as camera,
        view_factory(config.display, image_cache) as view,
    ):
        drawer = drawer_factory()
        while True:
            frame = camera.read()
            detection = detector.detect(frame)
            result = recognition.recognize(detection.hands)
            if config.display.show_landmarks:
                frame = drawer.draw(frame, detection.hands)

            should_continue = view.show(
                frame,
                fps=camera.fps,
                hand_count=len(detection.hands),
                result=result,
            )
            if not should_continue:
                break


def main() -> int:
    """Carga la configuración y ejecuta la detección de manos en video."""
    try:
        config = load_config()
    except ConfigError as exc:
        logging.basicConfig(level=logging.ERROR)
        LOGGER.error("No se pudo iniciar la aplicación: %s", exc)
        return 1

    configure_logging(config.logging)
    try:
        run_application(config)
    except KeyboardInterrupt:
        LOGGER.info("Aplicación cerrada por el usuario.")
    except (
        CameraError,
        HandDetectorError,
        ImageOverlayError,
        LandmarkDrawingError,
        OpenCVViewError,
        TemplateRepositoryError,
    ) as exc:
        LOGGER.error("No se pudo ejecutar la aplicación: %s", exc)
        return 1
    except Exception:
        LOGGER.exception("La aplicación terminó por un error inesperado.")
        return 1

    LOGGER.info("Aplicación cerrada correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
