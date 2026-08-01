"""Carga y validación estricta de la configuración YAML."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from gesture_matcher.utils.resource_path import (
    ResourcePathError,
    get_project_root,
    resolve_project_path,
)


class ConfigError(ValueError):
    """Indica que la configuración no puede utilizarse de forma segura."""


@dataclass(frozen=True)
class CameraConfig:
    """Parámetros de captura configurables para OpenCV."""

    index: int
    width: int
    height: int
    mirror: bool


@dataclass(frozen=True)
class HandDetectionConfig:
    """Parámetros de MediaPipe Hand Landmarker."""

    max_hands: int
    min_detection_confidence: float
    min_presence_confidence: float
    min_tracking_confidence: float


@dataclass(frozen=True)
class RecognitionConfig:
    """Parámetros de extracción y comparación de señas."""

    similarity_method: str
    similarity_threshold: float
    gesture_thresholds: Mapping[str, float]
    mirror_left_hand: bool


@dataclass(frozen=True)
class TemporalFilterConfig:
    """Parámetros para estabilizar resultados entre fotogramas."""

    window_size: int
    stable_frames: int
    min_consecutive_frames: int
    hold_frames: int
    hysteresis_frames: int


@dataclass(frozen=True)
class DisplayConfig:
    """Opciones de presentación de OpenCV."""

    show_landmarks: bool
    show_fps: bool
    result_image_width: int
    result_image_height: int


@dataclass(frozen=True)
class ResourceConfig:
    """Rutas de recursos resueltas desde la raíz del proyecto."""

    hand_model: Path
    gesture_templates: Path
    gesture_metadata: Path
    reference_images: Path
    display_images: Path


@dataclass(frozen=True)
class LoggingConfig:
    """Nivel del registro técnico."""

    level: str


@dataclass(frozen=True)
class AppConfig:
    """Configuración validada e inmutable de la aplicación."""

    project_root: Path
    camera: CameraConfig
    hand_detection: HandDetectionConfig
    recognition: RecognitionConfig
    temporal_filter: TemporalFilterConfig
    display: DisplayConfig
    resources: ResourceConfig
    logging: LoggingConfig


def load_config(
    config_path: str | Path | None = None,
    *,
    project_root: Path | None = None,
) -> AppConfig:
    """Carga ``config.yaml`` y devuelve valores tipados y validados."""
    try:
        root = get_project_root(project_root)
    except ResourcePathError as exc:
        raise ConfigError(str(exc)) from exc

    path = Path(config_path) if config_path is not None else Path("config/config.yaml")
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"No se encontró el archivo de configuración: {path}")

    try:
        with path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"No se pudo leer la configuración {path}: {exc}") from exc

    config = _as_mapping(raw_config, "configuración")
    camera_data = _section(config, "camera")
    hand_data = _section(config, "hand_detection")
    recognition_data = _section(config, "recognition")
    temporal_data = _section(config, "temporal_filter")
    display_data = _section(config, "display")
    resource_data = _section(config, "resources")
    logging_data = _section(config, "logging")

    camera = CameraConfig(
        index=_integer(camera_data, "index", minimum=0),
        width=_integer(camera_data, "width", minimum=1),
        height=_integer(camera_data, "height", minimum=1),
        mirror=_boolean(camera_data, "mirror"),
    )
    hand_detection = HandDetectionConfig(
        max_hands=_integer(hand_data, "max_hands", minimum=1, maximum=2),
        min_detection_confidence=_probability(hand_data, "min_detection_confidence"),
        min_presence_confidence=_probability(hand_data, "min_presence_confidence"),
        min_tracking_confidence=_probability(hand_data, "min_tracking_confidence"),
    )

    similarity_method = _string(recognition_data, "similarity_method").lower()
    if similarity_method != "cosine":
        raise ConfigError(
            "recognition.similarity_method debe ser 'cosine' durante el MVP."
        )
    recognition = RecognitionConfig(
        similarity_method=similarity_method,
        similarity_threshold=_probability(recognition_data, "similarity_threshold"),
        gesture_thresholds=_gesture_thresholds(recognition_data),
        mirror_left_hand=_boolean(recognition_data, "mirror_left_hand"),
    )

    window_size = _integer(temporal_data, "window_size", minimum=1)
    stable_frames = _integer(temporal_data, "stable_frames", minimum=1)
    min_consecutive_frames = _integer(
        temporal_data,
        "min_consecutive_frames",
        minimum=1,
    )
    hold_frames = _integer(temporal_data, "hold_frames", minimum=0)
    hysteresis_frames = _integer(
        temporal_data,
        "hysteresis_frames",
        minimum=0,
    )
    if stable_frames > window_size:
        raise ConfigError(
            "temporal_filter.stable_frames no puede superar "
            "temporal_filter.window_size."
        )
    if min_consecutive_frames > window_size:
        raise ConfigError(
            "temporal_filter.min_consecutive_frames no puede superar "
            "temporal_filter.window_size."
        )
    if hysteresis_frames >= stable_frames:
        raise ConfigError(
            "temporal_filter.hysteresis_frames debe ser menor que "
            "temporal_filter.stable_frames."
        )
    temporal_filter = TemporalFilterConfig(
        window_size=window_size,
        stable_frames=stable_frames,
        min_consecutive_frames=min_consecutive_frames,
        hold_frames=hold_frames,
        hysteresis_frames=hysteresis_frames,
    )

    display = DisplayConfig(
        show_landmarks=_boolean(display_data, "show_landmarks"),
        show_fps=_boolean(display_data, "show_fps"),
        result_image_width=_integer(display_data, "result_image_width", minimum=1),
        result_image_height=_integer(display_data, "result_image_height", minimum=1),
    )
    resources = _resource_config(resource_data, root)

    log_level = _string(logging_data, "level").upper()
    if log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise ConfigError(
            "logging.level debe ser CRITICAL, ERROR, WARNING, INFO o DEBUG."
        )

    return AppConfig(
        project_root=root,
        camera=camera,
        hand_detection=hand_detection,
        recognition=recognition,
        temporal_filter=temporal_filter,
        display=display,
        resources=resources,
        logging=LoggingConfig(level=log_level),
    )


def _resource_config(data: Mapping[str, Any], root: Path) -> ResourceConfig:
    try:
        return ResourceConfig(
            hand_model=resolve_project_path(
                _string(data, "hand_model"), project_root=root
            ),
            gesture_templates=resolve_project_path(
                _string(data, "gesture_templates"), project_root=root
            ),
            gesture_metadata=resolve_project_path(
                _string(data, "gesture_metadata"), project_root=root
            ),
            reference_images=resolve_project_path(
                _string(data, "reference_images"), project_root=root
            ),
            display_images=resolve_project_path(
                _string(data, "display_images"), project_root=root
            ),
        )
    except ResourcePathError as exc:
        raise ConfigError(str(exc)) from exc


def _gesture_thresholds(data: Mapping[str, Any]) -> Mapping[str, float]:
    raw_thresholds = _as_mapping(
        _required(data, "gesture_thresholds"),
        "recognition.gesture_thresholds",
    )
    thresholds: dict[str, float] = {}
    for gesture_id, raw_value in raw_thresholds.items():
        if not isinstance(gesture_id, str) or not gesture_id.strip():
            raise ConfigError(
                "Cada clave de recognition.gesture_thresholds debe ser texto no vacío."
            )
        thresholds[gesture_id] = _probability_value(
            raw_value, f"recognition.gesture_thresholds.{gesture_id}"
        )
    return MappingProxyType(thresholds)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _as_mapping(_required(config, name), name)


def _required(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ConfigError(f"Falta el campo obligatorio: {key}")
    return data[key]


def _as_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} debe ser un objeto de configuración.")
    return value


def _integer(
    data: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = _required(data, key)
    if type(value) is not int:
        raise ConfigError(f"{key} debe ser un entero.")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key} debe ser mayor o igual que {minimum}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{key} debe ser menor o igual que {maximum}.")
    return value


def _probability(data: Mapping[str, Any], key: str) -> float:
    return _probability_value(_required(data, key), key)


def _probability_value(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} debe ser un número entre 0 y 1.")
    converted = float(value)
    if not 0.0 <= converted <= 1.0:
        raise ConfigError(f"{field_name} debe estar entre 0 y 1.")
    return converted


def _boolean(data: Mapping[str, Any], key: str) -> bool:
    value = _required(data, key)
    if type(value) is not bool:
        raise ConfigError(f"{key} debe ser true o false.")
    return value


def _string(data: Mapping[str, Any], key: str) -> str:
    value = _required(data, key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} debe ser texto no vacío.")
    return value.strip()
