"""Detección offline de manos en imágenes de referencia."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, Self

import cv2
import mediapipe as mp
import numpy as np

from gesture_matcher.utils.config_loader import HandDetectionConfig
from gesture_matcher.vision.hand_detector import (
    HAND_LANDMARK_COUNT,
    HandDetectionError,
    HandDetectorError,
    HandDetectorInitializationError,
    HandObservation,
)


class ImageHandLandmarker(Protocol):
    """Operaciones de Hand Landmarker usadas para imágenes independientes."""

    def detect(self, image: mp.Image) -> Any:
        """Detecta landmarks en una imagen."""

    def close(self) -> None:
        """Libera los recursos nativos de MediaPipe."""


ImageLandmarkerFactory = Callable[
    [mp.tasks.vision.HandLandmarkerOptions],
    ImageHandLandmarker,
]


class ImageHandDetector:
    """Convierte una imagen BGR en observaciones de hasta dos manos."""

    def __init__(
        self,
        config: HandDetectionConfig,
        model_path: Path,
        *,
        landmarker_factory: ImageLandmarkerFactory | None = None,
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
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=config.max_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

        try:
            self._landmarker: ImageHandLandmarker | None = factory(options)
        except Exception as exc:
            raise HandDetectorInitializationError(
                "No se pudo cargar MediaPipe Hand Landmarker para imágenes desde "
                f"{resolved_model_path}: {exc}"
            ) from exc

        self._max_hands = config.max_hands

    def detect(self, image: np.ndarray) -> tuple[HandObservation, ...]:
        """Detecta manos en una imagen BGR sin conservar estado entre muestras."""
        landmarker = self._landmarker
        if landmarker is None:
            raise HandDetectionError("El detector de imágenes ya está cerrado.")

        self._validate_image(image)
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            media_pipe_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_image),
            )
            raw_result = landmarker.detect(media_pipe_image)
            return self._convert_result(raw_result)
        except HandDetectionError:
            raise
        except Exception as exc:
            raise HandDetectionError(
                f"Falló MediaPipe al procesar una imagen de referencia: {exc}"
            ) from exc

    def close(self) -> None:
        """Cierra Hand Landmarker; puede llamarse varias veces de forma segura."""
        landmarker = self._landmarker
        self._landmarker = None
        if landmarker is None:
            return

        try:
            landmarker.close()
        except Exception as exc:
            raise HandDetectorError(
                f"No se pudo cerrar el detector de imágenes: {exc}"
            ) from exc

    def __enter__(self) -> Self:
        """Devuelve el detector activo como administrador de contexto."""
        if self._landmarker is None:
            raise HandDetectorError("El detector de imágenes ya está cerrado.")
        return self

    def __exit__(self, *_: object) -> None:
        """Libera MediaPipe al salir del administrador de contexto."""
        self.close()

    def _convert_result(self, raw_result: Any) -> tuple[HandObservation, ...]:
        normalized_hands = raw_result.hand_landmarks
        world_hands = raw_result.hand_world_landmarks
        handedness = raw_result.handedness

        if len(normalized_hands) != len(world_hands):
            raise HandDetectionError(
                "MediaPipe devolvió cantidades distintas de landmarks normalizados "
                "y mundiales para una imagen."
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
        return tuple(observations)

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
        if score is not None and not np.isfinite(score):
            raise HandDetectionError(
                "MediaPipe devolvió una confianza de lateralidad no finita."
            )
        return label, score

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise HandDetectionError("La imagen debe ser un arreglo NumPy.")
        if image.dtype != np.uint8:
            raise HandDetectionError("La imagen debe utilizar valores uint8.")
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise HandDetectionError(
                "La imagen debe tener forma (alto, ancho, 3) y no estar vacía."
            )
