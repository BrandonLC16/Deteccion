"""Ventana mínima de video y estado mediante OpenCV."""

from typing import Protocol, Self

import cv2
import numpy as np

from gesture_matcher.utils.config_loader import DisplayConfig

WINDOW_TITLE = "Gesture Matcher - Q o ESC para salir"
EXIT_KEYS = {27, ord("q"), ord("Q")}
TEXT_COLOR = (255, 255, 255)


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
    """Presenta video, FPS y cantidad de manos sin ejecutar detección."""

    def __init__(
        self,
        config: DisplayConfig,
        *,
        backend: OpenCVBackend = cv2,
        window_title: str = WINDOW_TITLE,
    ) -> None:
        self._show_fps = config.show_fps
        self._backend = backend
        self._window_title = window_title
        self._closed = False

    def show(self, frame: np.ndarray, *, fps: float, hand_count: int) -> bool:
        """Muestra un fotograma y devuelve ``False`` al presionar Q o ESC."""
        self._validate_frame(frame)
        if self._closed:
            raise OpenCVViewError("La ventana de video ya está cerrada.")

        try:
            if self._show_fps:
                self._draw_text(frame, f"FPS: {fps:.1f}", (10, 30))
            self._draw_text(frame, f"Manos: {hand_count}", (10, 60))
            self._backend.imshow(self._window_title, frame)
            key = self._backend.waitKey(1) & 0xFF
        except Exception as exc:
            raise OpenCVViewError(
                f"No se pudo actualizar la ventana de video: {exc}"
            ) from exc

        return key not in EXIT_KEYS

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
