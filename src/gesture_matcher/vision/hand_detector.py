"""Detección de manos con MediaPipe Tasks Hand Landmarker."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from typing import Any, Protocol, Self

import cv2
import mediapipe as mp
import numpy as np

from gesture_matcher.utils.config_loader import HandDetectionConfig

HAND_LANDMARK_COUNT = 21


class HandDetectorError(RuntimeError):
    """Indica que el detector de manos no pudo operar de forma segura."""


class HandDetectorInitializationError(HandDetectorError):
    """Indica que MediaPipe Hand Landmarker no pudo inicializarse."""


class HandDetectionError(HandDetectorError):
    """Indica que falló la detección en un fotograma."""


@dataclass(frozen=True)
class HandObservation:
    """Landmarks y lateralidad de una mano detectada en un fotograma."""

    landmarks: np.ndarray
    world_landmarks: np.ndarray
    handedness: str | None
    handedness_score: float | None


@dataclass(frozen=True)
class HandDetectionResult:
    """Resultado inmutable de detección para un fotograma de video."""

    hands: tuple[HandObservation, ...]
    timestamp_ms: int


class HandLandmarker(Protocol):
    """Operaciones de Hand Landmarker utilizadas por el adaptador."""

    def detect_for_video(self, image: mp.Image, timestamp_ms: int) -> Any:
        """Detecta landmarks en un fotograma con timestamp creciente."""

    def close(self) -> None:
        """Libera los recursos nativos de MediaPipe."""


LandmarkerFactory = Callable[
    [mp.tasks.vision.HandLandmarkerOptions],
    HandLandmarker,
]
Clock = Callable[[], int]


class HandDetector:
    """Convierte fotogramas BGR en observaciones de una o dos manos."""

    def __init__(
        self,
        config: HandDetectionConfig,
        model_path: Path,
        *,
        landmarker_factory: LandmarkerFactory | None = None,
        clock_ns: Clock = monotonic_ns,
    ) -> None:
        resolved_model_path = model_path.resolve()
        if not resolved_model_path.is_file():
            raise HandDetectorInitializationError(
                f"No se encontró el modelo de manos: {resolved_model_path}. "
                "Agrega hand_landmarker.task en la ruta configurada."
            )

        factory = (
            landmarker_factory or mp.tasks.vision.HandLandmarker.create_from_options
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(resolved_model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=config.max_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

        try:
            self._landmarker: HandLandmarker | None = factory(options)
        except Exception as exc:
            raise HandDetectorInitializationError(
                f"No se pudo cargar MediaPipe Hand Landmarker desde "
                f"{resolved_model_path}: {exc}"
            ) from exc

        self._max_hands = config.max_hands
        self._clock_ns = clock_ns
        self._last_timestamp_ms: int | None = None

    def detect(
        self,
        frame: np.ndarray,
        *,
        timestamp_ms: int | None = None,
    ) -> HandDetectionResult:
        """Detecta manos en un fotograma BGR usando el modo VIDEO de MediaPipe."""
        landmarker = self._landmarker
        if landmarker is None:
            raise HandDetectionError("El detector de manos ya está cerrado.")

        self._validate_frame(frame)
        current_timestamp = self._next_timestamp_ms(timestamp_ms)

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            media_pipe_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_frame),
            )
            raw_result = landmarker.detect_for_video(
                media_pipe_image,
                current_timestamp,
            )
            return self._convert_result(raw_result, current_timestamp)
        except HandDetectionError:
            raise
        except Exception as exc:
            raise HandDetectionError(
                f"Falló MediaPipe Hand Landmarker en el timestamp "
                f"{current_timestamp} ms: {exc}"
            ) from exc

    def close(self) -> None:
        """Cierra Hand Landmarker; puede llamarse varias veces de forma segura."""
        landmarker = self._landmarker
        self._landmarker = None
        self._last_timestamp_ms = None
        if landmarker is None:
            return

        try:
            landmarker.close()
        except Exception as exc:
            raise HandDetectorError(
                f"No se pudo cerrar MediaPipe Hand Landmarker: {exc}"
            ) from exc

    def __enter__(self) -> Self:
        """Devuelve el detector activo para utilizarlo como contexto."""
        if self._landmarker is None:
            raise HandDetectorError("El detector de manos ya está cerrado.")
        return self

    def __exit__(self, *_: object) -> None:
        """Libera MediaPipe al salir del administrador de contexto."""
        self.close()

    def _next_timestamp_ms(self, timestamp_ms: int | None) -> int:
        if timestamp_ms is None:
            timestamp_ms = self._clock_ns() // 1_000_000
            if (
                self._last_timestamp_ms is not None
                and timestamp_ms <= self._last_timestamp_ms
            ):
                timestamp_ms = self._last_timestamp_ms + 1
        elif type(timestamp_ms) is not int or timestamp_ms < 0:
            raise HandDetectionError(
                "El timestamp del fotograma debe ser un entero no negativo."
            )

        if (
            self._last_timestamp_ms is not None
            and timestamp_ms <= self._last_timestamp_ms
        ):
            raise HandDetectionError(
                "Los timestamps de video deben ser estrictamente crecientes."
            )

        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _convert_result(
        self,
        raw_result: Any,
        timestamp_ms: int,
    ) -> HandDetectionResult:
        normalized_hands = raw_result.hand_landmarks
        world_hands = raw_result.hand_world_landmarks
        handedness = raw_result.handedness

        if len(normalized_hands) != len(world_hands):
            raise HandDetectionError(
                "MediaPipe devolvió cantidades distintas de landmarks normalizados "
                "y mundiales."
            )
        if len(normalized_hands) > self._max_hands:
            raise HandDetectionError(
                "MediaPipe devolvió más manos que el máximo configurado."
            )

        observations: list[HandObservation] = []
        for index, normalized_landmarks in enumerate(normalized_hands):
            hand_label, hand_score = self._read_handedness(handedness, index)
            observations.append(
                HandObservation(
                    landmarks=self._landmarks_to_array(
                        normalized_landmarks,
                        "normalizados",
                    ),
                    world_landmarks=self._landmarks_to_array(
                        world_hands[index],
                        "mundiales",
                    ),
                    handedness=hand_label,
                    handedness_score=hand_score,
                )
            )

        return HandDetectionResult(
            hands=tuple(observations),
            timestamp_ms=timestamp_ms,
        )

    @staticmethod
    def _landmarks_to_array(
        landmarks: Sequence[Any],
        coordinate_system: str,
    ) -> np.ndarray:
        if len(landmarks) != HAND_LANDMARK_COUNT:
            raise HandDetectionError(
                f"MediaPipe devolvió {len(landmarks)} landmarks {coordinate_system}; "
                f"se esperaban {HAND_LANDMARK_COUNT}."
            )

        values: list[tuple[float, float, float]] = []
        for landmark in landmarks:
            coordinates = (landmark.x, landmark.y, landmark.z)
            if any(value is None for value in coordinates):
                raise HandDetectionError(
                    f"MediaPipe devolvió landmarks {coordinate_system} incompletos."
                )
            values.append(tuple(float(value) for value in coordinates))

        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise HandDetectionError(
                f"MediaPipe devolvió landmarks {coordinate_system} con NaN o infinito."
            )
        array.setflags(write=False)
        return array

    @staticmethod
    def _read_handedness(
        handedness: Sequence[Sequence[Any]],
        hand_index: int,
    ) -> tuple[str | None, float | None]:
        if hand_index >= len(handedness) or not handedness[hand_index]:
            return None, None

        category = handedness[hand_index][0]
        label = category.category_name or category.display_name
        score = float(category.score) if category.score is not None else None
        return label, score

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise HandDetectionError("El fotograma debe ser un arreglo NumPy.")
        if frame.dtype != np.uint8:
            raise HandDetectionError("El fotograma debe utilizar valores uint8.")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise HandDetectionError(
                "El fotograma debe tener forma (alto, ancho, 3) y no estar vacío."
            )
