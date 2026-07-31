from dataclasses import replace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import gesture_matcher.app as app_module
from gesture_matcher.utils.config_loader import AppConfig, load_config
from gesture_matcher.vision.hand_detector import (
    HandDetectionError,
    HandDetectionResult,
)


@pytest.fixture
def app_config() -> AppConfig:
    return load_config()


def context_resource() -> MagicMock:
    resource = MagicMock()
    resource.__enter__.return_value = resource
    resource.__exit__.return_value = False
    return resource


def test_run_application_connects_video_detection_drawing_and_view(
    app_config: AppConfig,
) -> None:
    camera = context_resource()
    detector = context_resource()
    view = context_resource()
    drawer = Mock()
    first_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    second_frame = np.ones((2, 2, 3), dtype=np.uint8)
    camera.read.side_effect = [first_frame, second_frame]
    camera.fps = 30.0
    detector.detect.side_effect = [
        HandDetectionResult(hands=(), timestamp_ms=1),
        HandDetectionResult(hands=(), timestamp_ms=2),
    ]
    drawer.draw.side_effect = lambda frame, _hands: frame
    view.show.side_effect = [True, False]
    camera_factory = Mock(return_value=camera)
    detector_factory = Mock(return_value=detector)
    view_factory = Mock(return_value=view)
    drawer_factory = Mock(return_value=drawer)

    app_module.run_application(
        app_config,
        camera_factory=camera_factory,
        detector_factory=detector_factory,
        view_factory=view_factory,
        drawer_factory=drawer_factory,
    )

    camera_factory.assert_called_once_with(app_config.camera)
    detector_factory.assert_called_once_with(
        app_config.hand_detection,
        app_config.resources.hand_model,
    )
    view_factory.assert_called_once_with(app_config.display)
    assert camera.read.call_count == 2
    assert detector.detect.call_count == 2
    assert drawer.draw.call_count == 2
    assert view.show.call_count == 2
    assert all(call.kwargs["hand_count"] == 0 for call in view.show.call_args_list)
    detector.__exit__.assert_called_once()
    camera.__exit__.assert_called_once()
    view.__exit__.assert_called_once()


def test_run_application_skips_drawing_when_landmarks_are_disabled(
    app_config: AppConfig,
) -> None:
    config = replace(
        app_config,
        display=replace(app_config.display, show_landmarks=False),
    )
    camera = context_resource()
    detector = context_resource()
    view = context_resource()
    drawer = Mock()
    camera.read.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
    camera.fps = 0.0
    detector.detect.return_value = HandDetectionResult(hands=(), timestamp_ms=1)
    view.show.return_value = False

    app_module.run_application(
        config,
        camera_factory=Mock(return_value=camera),
        detector_factory=Mock(return_value=detector),
        view_factory=Mock(return_value=view),
        drawer_factory=Mock(return_value=drawer),
    )

    drawer.draw.assert_not_called()


def test_run_application_releases_every_resource_after_detection_error(
    app_config: AppConfig,
) -> None:
    camera = context_resource()
    detector = context_resource()
    view = context_resource()
    camera.read.return_value = np.zeros((2, 2, 3), dtype=np.uint8)
    detector.detect.side_effect = HandDetectionError("inferencia fallida")

    with pytest.raises(HandDetectionError, match="inferencia fallida"):
        app_module.run_application(
            app_config,
            camera_factory=Mock(return_value=camera),
            detector_factory=Mock(return_value=detector),
            view_factory=Mock(return_value=view),
            drawer_factory=Mock(),
        )

    detector.__exit__.assert_called_once()
    camera.__exit__.assert_called_once()
    view.__exit__.assert_called_once()


def test_main_returns_success_after_normal_exit(
    monkeypatch: pytest.MonkeyPatch,
    app_config: AppConfig,
) -> None:
    run_application = Mock()
    monkeypatch.setattr(app_module, "load_config", Mock(return_value=app_config))
    monkeypatch.setattr(app_module, "configure_logging", Mock())
    monkeypatch.setattr(app_module, "run_application", run_application)

    assert app_module.main() == 0
    run_application.assert_called_once_with(app_config)


def test_main_reports_known_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    app_config: AppConfig,
) -> None:
    monkeypatch.setattr(app_module, "load_config", Mock(return_value=app_config))
    monkeypatch.setattr(app_module, "configure_logging", Mock())
    monkeypatch.setattr(
        app_module,
        "run_application",
        Mock(side_effect=HandDetectionError("inferencia fallida")),
    )

    assert app_module.main() == 1
