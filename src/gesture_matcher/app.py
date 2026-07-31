"""Punto de entrada y composición de dependencias de la aplicación."""

import logging
from collections.abc import Callable
from pathlib import Path

from gesture_matcher.camera.camera_service import CameraError, CameraService
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
ViewFactory = Callable[[DisplayConfig], OpenCVView]
DrawerFactory = Callable[[], LandmarkDrawer]


def run_application(
    config: AppConfig,
    *,
    camera_factory: CameraFactory = CameraService,
    detector_factory: DetectorFactory = HandDetector,
    view_factory: ViewFactory = OpenCVView,
    drawer_factory: DrawerFactory = LandmarkDrawer,
) -> None:
    """Conecta cámara, detector y vista durante el ciclo de video."""
    with (
        detector_factory(
            config.hand_detection,
            config.resources.hand_model,
        ) as detector,
        camera_factory(config.camera) as camera,
        view_factory(config.display) as view,
    ):
        drawer = drawer_factory()
        while True:
            frame = camera.read()
            detection = detector.detect(frame)
            if config.display.show_landmarks:
                frame = drawer.draw(frame, detection.hands)

            should_continue = view.show(
                frame,
                fps=camera.fps,
                hand_count=len(detection.hands),
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
        LandmarkDrawingError,
        OpenCVViewError,
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
