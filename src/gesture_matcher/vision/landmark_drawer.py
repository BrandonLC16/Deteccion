"""Representación de landmarks de manos sobre fotogramas BGR."""

from collections.abc import Sequence
from typing import Protocol

import cv2
import mediapipe as mp
import numpy as np

from gesture_matcher.vision.hand_detector import (
    HAND_LANDMARK_COUNT,
    HandObservation,
)

HAND_CONNECTIONS = tuple(
    (connection.start, connection.end)
    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
)
LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 128, 0)
LABEL_COLOR = (255, 255, 255)


class LandmarkDrawingError(RuntimeError):
    """Indica que los landmarks no pudieron dibujarse de forma segura."""


class DrawingBackend(Protocol):
    """Operaciones de OpenCV utilizadas para dibujar landmarks."""

    def line(self, *args: object, **kwargs: object) -> np.ndarray:
        """Dibuja una conexión entre dos landmarks."""

    def circle(self, *args: object, **kwargs: object) -> np.ndarray:
        """Dibuja un landmark."""

    def putText(self, *args: object, **kwargs: object) -> np.ndarray:  # noqa: N802
        """Dibuja la lateralidad de una mano."""


class LandmarkDrawer:
    """Dibuja puntos, conexiones y lateralidad sin realizar detección."""

    def __init__(self, *, backend: DrawingBackend = cv2) -> None:
        self._backend = backend

    def draw(
        self,
        frame: np.ndarray,
        hands: Sequence[HandObservation],
    ) -> np.ndarray:
        """Anota un fotograma BGR y devuelve la misma instancia."""
        self._validate_frame(frame)

        try:
            for hand in hands:
                points = self._pixel_points(hand.landmarks, frame.shape)
                for start, end in HAND_CONNECTIONS:
                    self._backend.line(
                        frame,
                        points[start],
                        points[end],
                        CONNECTION_COLOR,
                        2,
                        cv2.LINE_AA,
                    )
                for point in points:
                    self._backend.circle(
                        frame,
                        point,
                        3,
                        LANDMARK_COLOR,
                        -1,
                        cv2.LINE_AA,
                    )
                self._draw_handedness(frame, hand, points[0])
        except LandmarkDrawingError:
            raise
        except Exception as exc:
            raise LandmarkDrawingError(
                f"No se pudieron dibujar los landmarks de manos: {exc}"
            ) from exc

        return frame

    def _draw_handedness(
        self,
        frame: np.ndarray,
        hand: HandObservation,
        wrist: tuple[int, int],
    ) -> None:
        if hand.handedness is None:
            return
        label = hand.handedness
        if hand.handedness_score is not None:
            label = f"{label} {hand.handedness_score:.2f}"
        position = (wrist[0], max(20, wrist[1] - 10))
        self._backend.putText(
            frame,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            LABEL_COLOR,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _pixel_points(
        landmarks: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> tuple[tuple[int, int], ...]:
        if landmarks.shape != (HAND_LANDMARK_COUNT, 3):
            raise LandmarkDrawingError(
                f"Se esperaban landmarks con forma ({HAND_LANDMARK_COUNT}, 3)."
            )
        if not np.isfinite(landmarks).all():
            raise LandmarkDrawingError("Los landmarks contienen NaN o infinito.")

        height, width = frame_shape[:2]
        return tuple(
            (
                int(round(float(np.clip(landmark[0], 0.0, 1.0)) * (width - 1))),
                int(round(float(np.clip(landmark[1], 0.0, 1.0)) * (height - 1))),
            )
            for landmark in landmarks
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.size == 0
        ):
            raise LandmarkDrawingError(
                "El fotograma para dibujar debe tener forma (alto, ancho, 3)."
            )
