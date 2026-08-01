"""Ventana responsiva de video, reconocimiento e imagen asociada."""

import logging
import math
from dataclasses import dataclass
from typing import Protocol, Self

import cv2
import numpy as np

from gesture_matcher.recognition.gesture_matcher import (
    UNKNOWN_GESTURE_LABEL,
    MatchResult,
)
from gesture_matcher.ui.image_overlay import ImageCache, resize_to_fit
from gesture_matcher.utils.config_loader import DisplayConfig

LOGGER = logging.getLogger(__name__)

WINDOW_TITLE = "Anime_Pose: Detección de señas y gestos"
EXIT_KEYS = {27, ord("q"), ord("Q")}
TEXT_COLOR = (255, 255, 255)
WINDOW_BACKGROUND = (16, 16, 16)
PANEL_BACKGROUND = (32, 32, 32)
SLOT_BACKGROUND = (8, 8, 8)
PANEL_BORDER = (80, 80, 80)
INITIAL_WINDOW_WIDTH = 1024
INITIAL_WINDOW_HEIGHT = 640
MIN_CANVAS_WIDTH = 320
MIN_CANVAS_HEIGHT = 240
MAX_PANEL_WIDTH = 920
MAX_PANEL_HEIGHT = 560
OUTER_MARGIN = 32
MIN_OUTER_MARGIN = 12
INNER_PADDING = 20
SECTION_GAP = 20
RESULT_INFO_HEIGHT = 90
CAMERA_WIDTH_RATIO = 0.50


@dataclass(frozen=True)
class Rect:
    """Rectángulo entero dentro del lienzo de la ventana."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        """Coordenada exclusiva del borde derecho."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Coordenada exclusiva del borde inferior."""
        return self.y + self.height


@dataclass(frozen=True)
class LayoutGeometry:
    """Geometría calculada para un tamaño concreto de ventana."""

    canvas_width: int
    canvas_height: int
    panel: Rect
    camera_slot: Rect
    image_slot: Rect
    info_y: int
    padding: int
    gap: int
    font_scale: float


def calculate_layout(window_width: int, window_height: int) -> LayoutGeometry:
    """Calcula un panel compacto, centrado y proporcional para la ventana."""
    if type(window_width) is not int or type(window_height) is not int:
        raise OpenCVViewError("El tamaño de ventana debe expresarse con enteros.")
    if window_width < 1 or window_height < 1:
        raise OpenCVViewError("El tamaño de ventana debe ser positivo.")

    canvas_width = max(MIN_CANVAS_WIDTH, window_width)
    canvas_height = max(MIN_CANVAS_HEIGHT, window_height)
    margin = min(
        OUTER_MARGIN,
        max(MIN_OUTER_MARGIN, min(canvas_width, canvas_height) // 16),
    )
    panel_width = min(MAX_PANEL_WIDTH, canvas_width - margin * 2)
    panel_height = min(MAX_PANEL_HEIGHT, canvas_height - margin * 2)
    panel = Rect(
        x=(canvas_width - panel_width) // 2,
        y=(canvas_height - panel_height) // 2,
        width=panel_width,
        height=panel_height,
    )

    padding = min(INNER_PADDING, max(10, panel_width // 40))
    gap = min(SECTION_GAP, max(12, panel_width // 45))
    info_height = min(RESULT_INFO_HEIGHT, max(64, panel_height // 5))
    visual_height = max(1, panel_height - padding * 2 - info_height)
    sections_width = max(2, panel_width - padding * 2 - gap)
    camera_width = max(1, int(round(sections_width * CAMERA_WIDTH_RATIO)))
    image_width = max(1, sections_width - camera_width)
    visual_y = panel.y + padding
    camera_slot = Rect(
        x=panel.x + padding,
        y=visual_y,
        width=camera_width,
        height=visual_height,
    )
    image_slot = Rect(
        x=camera_slot.right + gap,
        y=visual_y,
        width=image_width,
        height=visual_height,
    )
    font_scale = max(0.45, min(0.68, 0.68 * panel_width / MAX_PANEL_WIDTH))
    return LayoutGeometry(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        panel=panel,
        camera_slot=camera_slot,
        image_slot=image_slot,
        info_y=visual_y + visual_height + 26,
        padding=padding,
        gap=gap,
        font_scale=font_scale,
    )


class OpenCVViewError(RuntimeError):
    """Indica que la ventana OpenCV no pudo operar correctamente."""


class OpenCVBackend(Protocol):
    """Operaciones de OpenCV utilizadas por la vista."""

    def namedWindow(self, window_name: str, flags: int) -> None:  # noqa: N802
        """Crea una ventana redimensionable."""

    def resizeWindow(  # noqa: N802
        self,
        window_name: str,
        width: int,
        height: int,
    ) -> None:
        """Configura el tamaño inicial de la ventana."""

    def getWindowImageRect(  # noqa: N802
        self,
        window_name: str,
    ) -> tuple[int, int, int, int]:
        """Devuelve la posición y el tamaño útil actual de la ventana."""

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
        self._image_cache = image_cache
        self._backend = backend
        self._window_title = window_title
        self._window_width = INITIAL_WINDOW_WIDTH
        self._window_height = INITIAL_WINDOW_HEIGHT
        self._closed = False
        try:
            window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
            self._backend.namedWindow(self._window_title, window_flags)
            self._backend.resizeWindow(
                self._window_title,
                self._window_width,
                self._window_height,
            )
        except Exception as exc:
            raise OpenCVViewError(
                f"No se pudo crear la ventana de video: {exc}"
            ) from exc

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
            window_width, window_height = self._current_window_size()
            layout = calculate_layout(window_width, window_height)
            display_frame, image_visible = self._compose_layout(
                frame,
                result,
                layout,
            )
            if self._show_fps:
                self._draw_text(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (layout.camera_slot.x + 10, layout.camera_slot.y + 26),
                    font_scale=min(0.58, layout.font_scale),
                )
            self._draw_text(
                display_frame,
                f"Manos: {hand_count}",
                (layout.camera_slot.x + 10, layout.camera_slot.y + 52),
                font_scale=min(0.58, layout.font_scale),
            )
            self._draw_result(
                display_frame,
                result,
                layout=layout,
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
        layout: LayoutGeometry,
    ) -> tuple[np.ndarray, bool]:
        display = np.full(
            (layout.canvas_height, layout.canvas_width, 3),
            WINDOW_BACKGROUND,
            dtype=np.uint8,
        )
        self._fill_rect(display, layout.panel, PANEL_BACKGROUND)
        self._fill_rect(display, layout.camera_slot, SLOT_BACKGROUND)
        self._fill_rect(display, layout.image_slot, SLOT_BACKGROUND)

        camera_image = resize_to_fit(
            frame,
            target_width=layout.camera_slot.width,
            target_height=layout.camera_slot.height,
        )
        display[
            layout.camera_slot.y : layout.camera_slot.bottom,
            layout.camera_slot.x : layout.camera_slot.right,
        ] = camera_image

        image = None
        if result.accepted and result.display_image_path is not None:
            image = self._image_cache.get(result.display_image_path)
        if image is None:
            self._draw_border(display, layout.camera_slot)
            self._draw_border(display, layout.image_slot)
            return display, False

        presentation_image = resize_to_fit(
            image,
            target_width=layout.image_slot.width,
            target_height=layout.image_slot.height,
        )
        display[
            layout.image_slot.y : layout.image_slot.bottom,
            layout.image_slot.x : layout.image_slot.right,
        ] = presentation_image
        self._draw_border(display, layout.camera_slot)
        self._draw_border(display, layout.image_slot)
        return display, True

    def _draw_result(
        self,
        frame: np.ndarray,
        result: MatchResult,
        *,
        layout: LayoutGeometry,
        image_visible: bool,
    ) -> None:
        text_x = layout.panel.x + layout.padding
        text_y = layout.info_y
        line_step = max(24, int(round(30 * layout.font_scale / 0.68)))
        label = result.label if result.accepted else UNKNOWN_GESTURE_LABEL
        similarity = float(result.similarity)
        if not math.isfinite(similarity):
            similarity = 0.0

        self._draw_text(
            frame,
            f"Pose detectada: {label}",
            (text_x, text_y),
            font_scale=layout.font_scale,
        )
        self._draw_text(
            frame,
            f"Similitud: {similarity * 100:.1f} %",
            (text_x, text_y + line_step),
            font_scale=layout.font_scale,
        )
        if result.accepted and not image_visible:
            self._draw_text(
                frame,
                "Sin imagen asociada",
                (text_x, text_y + line_step * 2),
                font_scale=layout.font_scale,
            )

    def _current_window_size(self) -> tuple[int, int]:
        try:
            rect = self._backend.getWindowImageRect(self._window_title)
        except Exception as exc:
            LOGGER.debug(
                "No se pudo consultar el tamaño actual de la ventana: %s",
                exc,
            )
            return self._window_width, self._window_height

        if (
            isinstance(rect, tuple)
            and len(rect) == 4
            and type(rect[2]) is int
            and type(rect[3]) is int
            and rect[2] > 0
            and rect[3] > 0
        ):
            self._window_width = rect[2]
            self._window_height = rect[3]
        return self._window_width, self._window_height

    @staticmethod
    def _fill_rect(
        frame: np.ndarray,
        rect: Rect,
        color: tuple[int, int, int],
    ) -> None:
        frame[rect.y : rect.bottom, rect.x : rect.right] = color

    @staticmethod
    def _draw_border(frame: np.ndarray, rect: Rect) -> None:
        thickness = min(2, rect.width, rect.height)
        frame[
            rect.y : rect.y + thickness,
            rect.x : rect.right,
        ] = PANEL_BORDER
        frame[
            rect.bottom - thickness : rect.bottom,
            rect.x : rect.right,
        ] = PANEL_BORDER
        frame[
            rect.y : rect.bottom,
            rect.x : rect.x + thickness,
        ] = PANEL_BORDER
        frame[
            rect.y : rect.bottom,
            rect.right - thickness : rect.right,
        ] = PANEL_BORDER

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
        *,
        font_scale: float = 0.7,
    ) -> None:
        self._backend.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            raise OpenCVViewError(
                "El fotograma para mostrar debe ser uint8 y tener forma "
                "(alto, ancho, 3)."
            )
