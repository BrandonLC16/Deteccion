from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mediapipe as mp
import numpy as np
import pytest

from gesture_matcher.utils.config_loader import HandDetectionConfig
from gesture_matcher.vision.hand_detector import HandDetectionError
from gesture_matcher.vision.image_hand_detector import ImageHandDetector


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


def make_landmarks(count: int = 21) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=index / 100.0,
            y=index / 200.0,
            z=-index / 300.0,
        )
        for index in range(count)
    ]


def make_result(labels: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        hand_landmarks=[make_landmarks() for _ in labels],
        hand_world_landmarks=[make_landmarks() for _ in labels],
        handedness=[
            [
                SimpleNamespace(
                    category_name=label,
                    display_name=None,
                    score=0.9,
                )
            ]
            for label in labels
        ],
    )


def build_detector(
    config: HandDetectionConfig,
    model_path: Path,
    landmarker: Mock,
) -> tuple[ImageHandDetector, Mock]:
    factory = Mock(return_value=landmarker)
    return (
        ImageHandDetector(
            config,
            model_path,
            landmarker_factory=factory,
        ),
        factory,
    )


def test_builds_media_pipe_image_options(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    _, factory = build_detector(detection_config, model_path, Mock())

    options = factory.call_args.args[0]
    assert options.running_mode is mp.tasks.vision.RunningMode.IMAGE
    assert options.num_hands == 2
    assert options.min_hand_detection_confidence == pytest.approx(0.6)
    assert options.min_hand_presence_confidence == pytest.approx(0.7)


def test_detects_hands_and_converts_bgr_to_rgb(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    landmarker = Mock()
    landmarker.detect.return_value = make_result(["Right", "Left"])
    detector, _ = build_detector(detection_config, model_path, landmarker)
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    hands = detector.detect(image)

    media_pipe_image = landmarker.detect.call_args.args[0]
    assert media_pipe_image.image_format == mp.ImageFormat.SRGB
    assert media_pipe_image.numpy_view()[0, 0].tolist() == [30, 20, 10]
    assert [hand.handedness for hand in hands] == ["Right", "Left"]
    assert all(hand.landmarks.shape == (21, 3) for hand in hands)
    assert all(not hand.landmarks.flags.writeable for hand in hands)


def test_returns_empty_tuple_when_no_hand_is_detected(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    landmarker = Mock()
    landmarker.detect.return_value = make_result([])
    detector, _ = build_detector(detection_config, model_path, landmarker)

    result = detector.detect(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result == ()


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.array([], dtype=np.uint8),
    ],
)
def test_rejects_invalid_images(
    detection_config: HandDetectionConfig,
    model_path: Path,
    image: np.ndarray,
) -> None:
    detector, _ = build_detector(detection_config, model_path, Mock())

    with pytest.raises(HandDetectionError, match="imagen"):
        detector.detect(image)


def test_rejects_malformed_landmarks(
    detection_config: HandDetectionConfig,
    model_path: Path,
) -> None:
    landmarker = Mock()
    malformed = make_result(["Right"])
    malformed.hand_landmarks[0] = make_landmarks(count=20)
    landmarker.detect.return_value = malformed
    detector, _ = build_detector(detection_config, model_path, landmarker)

    with pytest.raises(HandDetectionError, match="se esperaban 21"):
        detector.detect(np.zeros((2, 2, 3), dtype=np.uint8))


def test_close_is_idempotent(
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
