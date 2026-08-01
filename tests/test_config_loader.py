from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from gesture_matcher.utils.config_loader import ConfigError, load_config


@pytest.fixture
def valid_config() -> dict[str, object]:
    return {
        "camera": {"index": 0, "width": 1280, "height": 720, "mirror": True},
        "hand_detection": {
            "max_hands": 2,
            "min_detection_confidence": 0.5,
            "min_presence_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
        "recognition": {
            "similarity_method": "cosine",
            "similarity_threshold": 0.85,
            "gesture_thresholds": {"victory": 0.9},
            "mirror_left_hand": True,
        },
        "temporal_filter": {
            "window_size": 7,
            "stable_frames": 5,
            "min_consecutive_frames": 3,
            "hold_frames": 3,
            "hysteresis_frames": 1,
        },
        "display": {
            "show_landmarks": True,
            "show_fps": True,
            "result_image_width": 320,
            "result_image_height": 320,
        },
        "resources": {
            "hand_model": "models/hand_landmarker.task",
            "gesture_templates": "data/gesture_templates.npz",
            "gesture_metadata": "data/gestures.json",
            "reference_images": "assets/reference_images",
            "display_images": "assets/display_images",
        },
        "logging": {"level": "INFO"},
    }


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return tmp_path


def write_config(
    project_root: Path,
    config: dict[str, object],
    filename: str = "config.yaml",
) -> Path:
    path = project_root / filename
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_loads_valid_configuration(
    project_root: Path, valid_config: dict[str, object]
) -> None:
    config_path = write_config(project_root, valid_config)

    config = load_config(config_path, project_root=project_root)

    assert config.camera.width == 1280
    assert config.hand_detection.max_hands == 2
    assert config.recognition.gesture_thresholds["victory"] == pytest.approx(0.9)
    assert config.temporal_filter.min_consecutive_frames == 3
    assert config.temporal_filter.hold_frames == 3
    assert config.temporal_filter.hysteresis_frames == 1
    assert (
        config.resources.hand_model
        == (project_root / "models/hand_landmarker.task").resolve()
    )


def test_rejects_missing_configuration(project_root: Path) -> None:
    missing_path = project_root / "missing.yaml"

    with pytest.raises(ConfigError, match="No se encontró"):
        load_config(missing_path, project_root=project_root)


def test_rejects_missing_required_field(
    project_root: Path, valid_config: dict[str, object]
) -> None:
    invalid = deepcopy(valid_config)
    del invalid["camera"]["width"]  # type: ignore[index]
    config_path = write_config(project_root, invalid)

    with pytest.raises(ConfigError, match="width"):
        load_config(config_path, project_root=project_root)


def test_rejects_wrong_field_type(
    project_root: Path, valid_config: dict[str, object]
) -> None:
    invalid = deepcopy(valid_config)
    invalid["camera"]["index"] = "0"  # type: ignore[index]
    config_path = write_config(project_root, invalid)

    with pytest.raises(ConfigError, match="index debe ser un entero"):
        load_config(config_path, project_root=project_root)


@pytest.mark.parametrize("invalid_dimension", [0, -1])
def test_rejects_invalid_resolution(
    project_root: Path,
    valid_config: dict[str, object],
    invalid_dimension: int,
) -> None:
    invalid = deepcopy(valid_config)
    invalid["camera"]["width"] = invalid_dimension  # type: ignore[index]
    config_path = write_config(project_root, invalid)

    with pytest.raises(ConfigError, match="width"):
        load_config(config_path, project_root=project_root)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, "high"])
def test_rejects_invalid_similarity_threshold(
    project_root: Path,
    valid_config: dict[str, object],
    threshold: object,
) -> None:
    invalid = deepcopy(valid_config)
    invalid["recognition"]["similarity_threshold"] = threshold  # type: ignore[index]
    config_path = write_config(project_root, invalid)

    with pytest.raises(ConfigError, match="similarity_threshold"):
        load_config(config_path, project_root=project_root)


def test_rejects_resource_path_outside_project(
    project_root: Path, valid_config: dict[str, object]
) -> None:
    invalid = deepcopy(valid_config)
    invalid["resources"]["hand_model"] = "../hand_landmarker.task"  # type: ignore[index]
    config_path = write_config(project_root, invalid)

    with pytest.raises(ConfigError, match="sale de la raíz"):
        load_config(config_path, project_root=project_root)


def test_rejects_more_stable_frames_than_window(
    project_root: Path, valid_config: dict[str, object]
) -> None:
    invalid = deepcopy(valid_config)
    invalid["temporal_filter"]["stable_frames"] = 8  # type: ignore[index]
    config_path = write_config(project_root, invalid)

    with pytest.raises(ConfigError, match="stable_frames"):
        load_config(config_path, project_root=project_root)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("min_consecutive_frames", 0),
        ("min_consecutive_frames", 8),
        ("hold_frames", -1),
        ("hold_frames", True),
        ("hysteresis_frames", -1),
        ("hysteresis_frames", 5),
    ],
)
def test_rejects_invalid_temporal_filter_parameters(
    project_root: Path,
    valid_config: dict[str, object],
    field_name: str,
    invalid_value: object,
) -> None:
    invalid = deepcopy(valid_config)
    invalid["temporal_filter"][field_name] = invalid_value  # type: ignore[index]
    config_path = write_config(project_root, invalid)

    with pytest.raises(ConfigError, match=field_name):
        load_config(config_path, project_root=project_root)
