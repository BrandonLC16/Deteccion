"""Apertura, lectura y liberación segura de cámaras mediante OpenCV."""

import logging
from collections.abc import Callable
from time import perf_counter
from typing import Protocol, Self

import cv2
import numpy as np

from gesture_matcher.utils.config_loader import CameraConfig

LOGGER = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Indica que la cámara no pudo utilizarse de forma segura."""


class CameraOpenError(CameraError):
    """Indica que OpenCV no pudo abrir o configurar la cámara."""


class CameraReadError(CameraError):
    """Indica que no se pudo obtener un fotograma válido."""


class VideoCapture(Protocol):
    """Operaciones de ``cv2.VideoCapture`` utilizadas por el servicio."""

    def isOpened(self) -> bool:  # noqa: N802 - nombre definido por OpenCV
        """Informa si el dispositivo de captura está disponible."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Obtiene el siguiente fotograma de la cámara."""

    def set(self, property_id: int, value: float) -> bool:
        """Solicita un valor para una propiedad de captura."""

    def release(self) -> None:
        """Libera el dispositivo de captura."""


CaptureFactory = Callable[[int], VideoCapture]
Clock = Callable[[], float]


class CameraService:
    """Administra una cámara configurable sin incorporar lógica de reconocimiento."""

    def __init__(
        self,
        config: CameraConfig,
        *,
        capture_factory: CaptureFactory | None = None,
        clock: Clock = perf_counter,
    ) -> None:
        self._config = config
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._clock = clock
        self._capture: VideoCapture | None = None
        self._last_frame_time: float | None = None
        self._fps = 0.0

    @property
    def is_open(self) -> bool:
        """Indica si existe un dispositivo abierto actualmente."""
        return self._capture is not None and bool(self._capture.isOpened())

    @property
    def fps(self) -> float:
        """Devuelve los FPS instantáneos medidos entre las últimas dos lecturas."""
        return self._fps

    def open(self) -> None:
        """Abre la cámara y solicita la resolución configurada."""
        if self.is_open:
            return
        if self._capture is not None:
            self.release()

        capture: VideoCapture | None = None
        try:
            capture = self._capture_factory(self._config.index)
            if not capture.isOpened():
                raise CameraOpenError(
                    f"No se pudo abrir la cámara con índice {self._config.index}. "
                    "Comprueba que esté conectada y que otra aplicación no la use."
                )

            self._configure_resolution(capture)
        except CameraOpenError:
            self._release_after_open_failure(capture)
            raise
        except Exception as exc:
            self._release_after_open_failure(capture)
            raise CameraOpenError(
                f"No se pudo inicializar la cámara con índice {self._config.index}: "
                f"{exc}"
            ) from exc

        self._capture = capture
        self._reset_metrics()

    def read(self) -> np.ndarray:
        """Lee un fotograma válido y aplica el efecto espejo configurado."""
        capture = self._capture
        if capture is None:
            raise CameraReadError(
                "No se puede leer un fotograma porque la cámara no está abierta."
            )

        try:
            if not capture.isOpened():
                raise CameraReadError(
                    f"La cámara con índice {self._config.index} se cerró "
                    "antes de leer el fotograma."
                )
            success, frame = capture.read()
        except CameraReadError:
            raise
        except Exception as exc:
            raise CameraReadError(
                f"Falló la lectura de la cámara con índice {self._config.index}: {exc}"
            ) from exc

        if not success or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise CameraReadError(
                f"La cámara con índice {self._config.index} no devolvió "
                "un fotograma válido."
            )

        if self._config.mirror:
            try:
                frame = cv2.flip(frame, 1)
            except Exception as exc:
                raise CameraReadError(
                    "No se pudo aplicar el efecto espejo al fotograma."
                ) from exc

        self._update_fps()
        return frame

    def release(self) -> None:
        """Libera la cámara; puede llamarse más de una vez de forma segura."""
        capture = self._capture
        self._capture = None
        self._reset_metrics()
        if capture is None:
            return

        try:
            capture.release()
        except Exception as exc:
            raise CameraError(
                f"No se pudo liberar la cámara con índice {self._config.index}: {exc}"
            ) from exc

    def __enter__(self) -> Self:
        """Abre la cámara para utilizar el servicio como administrador de contexto."""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Libera la cámara al abandonar el administrador de contexto."""
        self.release()

    def _configure_resolution(self, capture: VideoCapture) -> None:
        requested_properties = (
            (cv2.CAP_PROP_FRAME_WIDTH, float(self._config.width), "anchura"),
            (cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.height), "altura"),
        )
        for property_id, value, label in requested_properties:
            if not capture.set(property_id, value):
                LOGGER.warning(
                    "La cámara con índice %s no aceptó la %s solicitada de %s px.",
                    self._config.index,
                    label,
                    int(value),
                )

    def _update_fps(self) -> None:
        current_time = self._clock()
        if self._last_frame_time is None:
            self._fps = 0.0
        else:
            elapsed = current_time - self._last_frame_time
            self._fps = 1.0 / elapsed if elapsed > 0.0 else 0.0
        self._last_frame_time = current_time

    def _reset_metrics(self) -> None:
        self._last_frame_time = None
        self._fps = 0.0

    def _release_after_open_failure(self, capture: VideoCapture | None) -> None:
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            LOGGER.exception(
                "También falló la liberación de la cámara con índice %s.",
                self._config.index,
            )
