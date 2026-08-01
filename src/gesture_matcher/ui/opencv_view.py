"""Ventana de video, reconocimiento e imagen asociada mediante OpenCV."""

import math
from typing import Protocol, Self

import cv2
import numpy as np

from gesture_matcher.recognition.gesture_matcher import (
    UNKNOWN_GESTURE_LABEL,
    MatchResult,
)
from gesture_matcher.ui.image_overlay import ImageCache
from gesture_matcher.utils.config_loader import DisplayConfig

WINDOW_TITLE = "Gesture Matcher - Q o ESC para salir"
EXIT_KEYS = {27, ord("q"), ord("Q")}
TEXT_COLOR = (255, 255, 255)
PANEL_BACKGROUND = (32, 32, 32)
PANEL_SEPARATOR = (96, 96, 96)
PANEL_PADDING = 20
MIN_PANEL_WIDTH = 360


class OpenCVViewError(RuntimeError):
    """Indica que la ventana OpenCV no pudo operar correctamente."""


class OpenCVBackend(Protocol):
    """Operaciones de OpenCV utilizadas por la vista."""

    def putText(self, *args: object, **kwargs: object) -> np.ndarray:  # noqa: N802
        """Dibuja texto informativo sobre el fotograma."""

    def imshow(self, window_name: str, frame: np.ndarray) -> None:
        """Muestra el fotograma en una ventana."""

    def waitKey(self, delay: int) -> int:  # noqa: N802
        """Espera brevemente una tecla."""

    def destroyAllWindows(self) -> None:  # noqa: N802
        """Destruye las ventanas creadas por OpenCV."""


class OpenCVView:
    """Presenta video y el último resultado confirmado sin reconocer señas."""

    def __init__(
        self,
        config: DisplayConfig,
        image_cache: ImageCache,
        *,
        backend: OpenCVBackend = cv2,
        window_title: str = WINDOW_TITLE,
    ) -> None:
        self._show_fps = config.show_fps
        self._result_image_width = config.result_image_width
        self._result_image_height = config.result_image_height
        self._image_cache = image_cache
        self._backend = backend
        self._window_title = window_title
        self._closed = False

    def show(
        self,
        frame: np.ndarray,
        *,
        fps: float,
        hand_count: int,
        result: MatchResult,
    ) -> bool:
        """Muestra un fotograma y devuelve ``False`` al presionar Q o ESC."""
        self._validate_frame(frame)
        if not isinstance(result, MatchResult):
            raise OpenCVViewError("El resultado para mostrar debe ser MatchResult.")
        if self._closed:
            raise OpenCVViewError("La ventana de video ya está cerrada.")

        try:
            display_frame, panel_start, image_visible = self._compose_layout(
                frame,
                result,
            )
            if self._show_fps:
                self._draw_text(display_frame, f"FPS: {fps:.1f}", (10, 30))
            self._draw_text(display_frame, f"Manos: {hand_count}", (10, 60))
            self._draw_result(
                display_frame,
                result,
                panel_start=panel_start,
                image_visible=image_visible,
            )
            self._backend.imshow(self._window_title, display_frame)
            key = self._backend.waitKey(1) & 0xFF
        except Exception as exc:
            raise OpenCVViewError(
                f"No se pudo actualizar la ventana de video: {exc}"
            ) from exc

        return key not in EXIT_KEYS

    def _compose_layout(
        self,
        frame: np.ndarray,
        result: MatchResult,
    ) -> tuple[np.ndarray, int, bool]:
        frame_height, frame_width = frame.shape[:2]
        panel_width = max(
            MIN_PANEL_WIDTH,
            self._result_image_width + PANEL_PADDING * 2,
        )
        display_height = max(
            frame_height,
            self._result_image_height + 140,
        )
        display = np.full(
            (display_height, frame_width + panel_width, 3),
            PANEL_BACKGROUND,
            dtype=np.uint8,
        )
        display[:frame_height, :frame_width] = frame
        display[:, frame_width : frame_width + 2] = PANEL_SEPARATOR

        image = None
        if result.accepted and result.display_image_path is not None:
            image = self._image_cache.get(result.display_image_path)
        if image is None:
            return display, frame_width, False
        if image.shape != (
            self._result_image_height,
            self._result_image_width,
            3,
        ):
            raise OpenCVViewError(
                "La imagen en caché no coincide con las dimensiones configuradas."
            )

        image_x = frame_width + (panel_width - self._result_image_width) // 2
        image_y = PANEL_PADDING
        display[
            image_y : image_y + self._result_image_height,
            image_x : image_x + self._result_image_width,
        ] = image
        return display, frame_width, True

    def _draw_result(
        self,
        frame: np.ndarray,
        result: MatchResult,
        *,
        panel_start: int,
        image_visible: bool,
    ) -> None:
        text_x = panel_start + PANEL_PADDING
        text_y = PANEL_PADDING + self._result_image_height + 35 if image_visible else 50
        label = result.label if result.accepted else UNKNOWN_GESTURE_LABEL
        similarity = float(result.similarity)
        if not math.isfinite(similarity):
            similarity = 0.0

        self._draw_text(frame, label, (text_x, text_y))
        self._draw_text(
            frame,
            f"Similitud: {similarity * 100:.1f} %",
            (text_x, text_y + 35),
        )
        if result.accepted and not image_visible:
            self._draw_text(
                frame,
                "Sin imagen asociada",
                (text_x, text_y + 70),
            )

    def close(self) -> None:
        """Destruye las ventanas OpenCV una sola vez."""
        if self._closed:
            return
        self._closed = True
        try:
            self._backend.destroyAllWindows()
        except Exception as exc:
            raise OpenCVViewError(
                f"No se pudieron cerrar las ventanas OpenCV: {exc}"
            ) from exc

    def __enter__(self) -> Self:
        """Devuelve la vista activa para utilizarla como contexto."""
        if self._closed:
            raise OpenCVViewError("La ventana de video ya está cerrada.")
        return self

    def __exit__(self, *_: object) -> None:
        """Cierra todas las ventanas al abandonar el contexto."""
        self.close()

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
    ) -> None:
        self._backend.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            raise OpenCVViewError(
                "El fotograma para mostrar debe tener forma (alto, ancho, 3)."
            )
