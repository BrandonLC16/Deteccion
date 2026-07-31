from dataclasses import replace
from unittest.mock import Mock, call

import cv2
import numpy as np
import pytest

from gesture_matcher.camera.camera_service import (
    CameraOpenError,
    CameraReadError,
    CameraService,
)
from gesture_matcher.utils.config_loader import CameraConfig


@pytest.fixture
def camera_config() -> CameraConfig:
    return CameraConfig(index=2, width=640, height=480, mirror=True)


@pytest.fixture
def opened_capture() -> Mock:
    capture = Mock()
    capture.isOpened.return_value = True
    capture.set.return_value = True
    return capture


def test_opens_camera_and_requests_configured_resolution(
    camera_config: CameraConfig,
    opened_capture: Mock,
) -> None:
    capture_factory = Mock(return_value=opened_capture)
    camera = CameraService(camera_config, capture_factory=capture_factory)

    camera.open()

    capture_factory.assert_called_once_with(2)
    assert opened_capture.set.call_args_list == [
        call(cv2.CAP_PROP_FRAME_WIDTH, 640.0),
        call(cv2.CAP_PROP_FRAME_HEIGHT, 480.0),
    ]
    assert camera.is_open


def test_releases_capture_when_camera_cannot_be_opened(
    camera_config: CameraConfig,
) -> None:
    capture = Mock()
    capture.isOpened.return_value = False
    camera = CameraService(camera_config, capture_factory=Mock(return_value=capture))

    with pytest.raises(CameraOpenError, match="índice 2"):
        camera.open()

    capture.release.assert_called_once_with()
    assert not camera.is_open


def test_releases_capture_when_configuration_raises(
    camera_config: CameraConfig,
    opened_capture: Mock,
) -> None:
    opened_capture.set.side_effect = RuntimeError("propiedad no disponible")
    camera = CameraService(
        camera_config,
        capture_factory=Mock(return_value=opened_capture),
    )

    with pytest.raises(CameraOpenError, match="propiedad no disponible"):
        camera.open()

    opened_capture.release.assert_called_once_with()


def test_reads_mirrored_frames_and_updates_fps(
    camera_config: CameraConfig,
    opened_capture: Mock,
) -> None:
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    opened_capture.read.side_effect = [(True, frame), (True, frame)]
    clock = Mock(side_effect=[10.0, 10.5])
    camera = CameraService(
        camera_config,
        capture_factory=Mock(return_value=opened_capture),
        clock=clock,
    )
    camera.open()

    first_frame = camera.read()
    assert np.array_equal(first_frame, frame[:, ::-1])
    assert camera.fps == 0.0

    camera.read()
    assert camera.fps == pytest.approx(2.0)


def test_preserves_frame_when_mirror_is_disabled(
    camera_config: CameraConfig,
    opened_capture: Mock,
) -> None:
    frame = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    opened_capture.read.return_value = (True, frame)
    camera = CameraService(
        replace(camera_config, mirror=False),
        capture_factory=Mock(return_value=opened_capture),
    )
    camera.open()

    assert camera.read() is frame


def test_rejects_read_before_open(camera_config: CameraConfig) -> None:
    camera = CameraService(camera_config)

    with pytest.raises(CameraReadError, match="no está abierta"):
        camera.read()


@pytest.mark.parametrize(
    ("success", "frame"),
    [(False, None), (True, None), (True, np.array([], dtype=np.uint8))],
)
def test_rejects_invalid_frames(
    camera_config: CameraConfig,
    opened_capture: Mock,
    success: bool,
    frame: np.ndarray | None,
) -> None:
    opened_capture.read.return_value = (success, frame)
    camera = CameraService(
        camera_config,
        capture_factory=Mock(return_value=opened_capture),
    )
    camera.open()

    with pytest.raises(CameraReadError, match="fotograma válido"):
        camera.read()


def test_release_is_idempotent_and_resets_metrics(
    camera_config: CameraConfig,
    opened_capture: Mock,
) -> None:
    opened_capture.read.return_value = (
        True,
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    camera = CameraService(
        camera_config,
        capture_factory=Mock(return_value=opened_capture),
    )
    camera.open()
    camera.read()

    camera.release()
    camera.release()

    opened_capture.release.assert_called_once_with()
    assert not camera.is_open
    assert camera.fps == 0.0


def test_context_manager_releases_camera_after_error(
    camera_config: CameraConfig,
    opened_capture: Mock,
) -> None:
    camera = CameraService(
        camera_config,
        capture_factory=Mock(return_value=opened_capture),
    )

    with pytest.raises(RuntimeError, match="fallo en el ciclo"):
        with camera:
            raise RuntimeError("fallo en el ciclo")

    opened_capture.release.assert_called_once_with()
    assert not camera.is_open
