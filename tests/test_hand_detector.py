from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mediapipe as mp
import numpy as np
import pytest

from gesture_matcher.utils.config_loader import HandDetectionConfig
from gesture_matcher.vision.hand_detector import (
    HandDetectionError,
    HandDetector,
    HandDetectorInitializationError,
)


@pytest.fixture
def detection_config() -> HandDetectionConfig:
    return HandDetectionConfig(
        max_hands=2,
        min_detection_confidence=0.6,
        min_presence_confidence=0.7,
        min_tracking_confidence=0.8,
    )


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "hand_landmarker.task"
    path.write_bytes(b"test model")
    return path


def make_landmarks(offset: float = 0.0, count: int = 21) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=offset + index / 100.0,
            y=offset + index / 200.0,
            z=-index / 300.0,
        )
        for index in range(count)
    ]


def make_result(labels: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        hand_landmarks=[make_landmarks(index / 10.0) for index, _ in enumerate(labels)],
        hand_world_landmarks=[
            make_landmarks(index / 20.0) for index, _ in enumerate(labels)
        ],
        handedness=[
            [
                SimpleNamespace(
                    category_name=label,
                    display_name=None,
                    score=0.9 - index / 10.0,
                )
            ]
            for index, label in enumerate(labels)
        ],
    )


def build_detector(
    detection_config: HandDetectionConfig,
    model_path: Path,
    landmarker: Mock,
    *,
    clock_ns: Mock | None = None,
) -> tuple[HandDetector, Mock]:
    factory = Mock(return_value=landmarker)
    detector = HandDetector(
        detection_config,
        model_path,
        landmarker_factory=factory,
        clock_ns=clock_ns or Mock(return_value=1_000_000_000),
    )
    return detector, factory


def test_builds_current_media_pipe_video_options(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    landmarker = Mock()

    _, factory = build_detector(detection_config, model_path, landmarker)

    options = factory.call_args.args[0]
    assert options.base_options.model_asset_path == str(model_path.resolve())
    assert options.running_mode is mp.tasks.vision.RunningMode.VIDEO
    assert options.num_hands == 2
    assert options.min_hand_detection_confidence == pytest.approx(0.6)
    assert options.min_hand_presence_confidence == pytest.approx(0.7)
    assert options.min_tracking_confidence == pytest.approx(0.8)


def test_rejects_missing_model(
    detection_config: HandDetectionConfig,
    tmp_path: Path,
) -> None:
    factory = Mock()
    missing_model = tmp_path / "missing.task"

    with pytest.raises(HandDetectorInitializationError, match="No se encontró"):
        HandDetector(
            detection_config,
            missing_model,
            landmarker_factory=factory,
        )

    factory.assert_not_called()


@pytest.mark.parametrize("labels", [["Left"], ["Left", "Right"]])
def test_detects_one_or_two_hands_and_converts_bgr_to_rgb(
    detection_config: HandDetectionConfig,
    model_path: Path,
    labels: list[str],
) -> None:
    landmarker = Mock()
    landmarker.detect_for_video.return_value = make_result(labels)
    detector, _ = build_detector(detection_config, model_path, landmarker)
    frame = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    result = detector.detect(frame, timestamp_ms=25)

    media_pipe_image, timestamp_ms = landmarker.detect_for_video.call_args.args
    assert timestamp_ms == 25
    assert media_pipe_image.image_format == mp.ImageFormat.SRGB
    assert media_pipe_image.numpy_view()[0, 0].tolist() == [30, 20, 10]
    assert len(result.hands) == len(labels)
    assert [hand.handedness for hand in result.hands] == labels
    assert result.hands[0].landmarks.shape == (21, 3)
    assert result.hands[0].world_landmarks.shape == (21, 3)
    assert not result.hands[0].landmarks.flags.writeable


def test_returns_empty_result_when_no_hands_are_detected(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    landmarker = Mock()
    landmarker.detect_for_video.return_value = make_result([])
    detector, _ = build_detector(detection_config, model_path, landmarker)

    result = detector.detect(np.zeros((2, 2, 3), dtype=np.uint8), timestamp_ms=1)

    assert result.hands == ()


def test_generates_strictly_increasing_timestamps(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    landmarker = Mock()
    landmarker.detect_for_video.return_value = make_result([])
    clock = Mock(return_value=1_000_000_000)
    detector, _ = build_detector(
        detection_config,
        model_path,
        landmarker,
        clock_ns=clock,
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    first = detector.detect(frame)
    second = detector.detect(frame)

    assert (first.timestamp_ms, second.timestamp_ms) == (1000, 1001)
    timestamps = [call.args[1] for call in landmarker.detect_for_video.call_args_list]
    assert timestamps == [1000, 1001]


def test_rejects_non_increasing_explicit_timestamp(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    landmarker = Mock()
    landmarker.detect_for_video.return_value = make_result([])
    detector, _ = build_detector(detection_config, model_path, landmarker)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    detector.detect(frame, timestamp_ms=10)

    with pytest.raises(HandDetectionError, match="estrictamente crecientes"):
        detector.detect(frame, timestamp_ms=10)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.array([], dtype=np.uint8),
    ],
)
def test_rejects_invalid_frames(
    detection_config: HandDetectionConfig,
    model_path: Path,
    frame: np.ndarray,
) -> None:
    detector, _ = build_detector(detection_config, model_path, Mock())

    with pytest.raises(HandDetectionError, match="fotograma"):
        detector.detect(frame)


def test_rejects_malformed_media_pipe_landmarks(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    landmarker = Mock()
    malformed = make_result(["Left"])
    malformed.hand_landmarks[0] = make_landmarks(count=20)
    landmarker.detect_for_video.return_value = malformed
    detector, _ = build_detector(detection_config, model_path, landmarker)

    with pytest.raises(HandDetectionError, match="se esperaban 21"):
        detector.detect(np.zeros((2, 2, 3), dtype=np.uint8))


def test_wraps_media_pipe_detection_errors(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    landmarker = Mock()
    landmarker.detect_for_video.side_effect = RuntimeError("inferencia fallida")
    detector, _ = build_detector(detection_config, model_path, landmarker)

    with pytest.raises(HandDetectionError, match="inferencia fallida"):
        detector.detect(np.zeros((2, 2, 3), dtype=np.uint8))


def test_close_is_idempotent_and_prevents_new_detections(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    landmarker = Mock()
    detector, _ = build_detector(detection_config, model_path, landmarker)

    detector.close()
    detector.close()

    landmarker.close.assert_called_once_with()
    with pytest.raises(HandDetectionError, match="cerrado"):
        detector.detect(np.zeros((2, 2, 3), dtype=np.uint8))
