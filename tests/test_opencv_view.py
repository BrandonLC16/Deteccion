from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from gesture_matcher.ui.opencv_view import WINDOW_TITLE, OpenCVView, OpenCVViewError
from gesture_matcher.utils.config_loader import DisplayConfig


@pytest.fixture
def display_config() -> DisplayConfig:
    return DisplayConfig(
        show_landmarks=True,
        show_fps=True,
        result_image_width=320,
        result_image_height=320,
    )


def test_shows_video_fps_and_hand_count(display_config: DisplayConfig) -> None:
    backend = Mock()
    backend.waitKey.return_value = -1
    view = OpenCVView(display_config, backend=backend)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    should_continue = view.show(frame, fps=29.75, hand_count=2)

    assert should_continue
    assert backend.putText.call_count == 2
    drawn_text = [call.args[1] for call in backend.putText.call_args_list]
    assert drawn_text == ["FPS: 29.8", "Manos: 2"]
    backend.imshow.assert_called_once_with(WINDOW_TITLE, frame)
    backend.waitKey.assert_called_once_with(1)


def test_respects_disabled_fps(display_config: DisplayConfig) -> None:
    backend = Mock()
    backend.waitKey.return_value = -1
    view = OpenCVView(replace(display_config, show_fps=False), backend=backend)

    view.show(np.zeros((2, 2, 3), dtype=np.uint8), fps=30.0, hand_count=0)

    backend.putText.assert_called_once()
    assert backend.putText.call_args.args[1] == "Manos: 0"


@pytest.mark.parametrize("key", [27, ord("q"), ord("Q")])
def test_stops_for_exit_keys(display_config: DisplayConfig, key: int) -> None:
    backend = Mock()
    backend.waitKey.return_value = key
    view = OpenCVView(display_config, backend=backend)

    assert not view.show(
        np.zeros((2, 2, 3), dtype=np.uint8),
        fps=0.0,
        hand_count=0,
    )


def test_close_is_idempotent(display_config: DisplayConfig) -> None:
    backend = Mock()
    view = OpenCVView(display_config, backend=backend)

    view.close()
    view.close()

    backend.destroyAllWindows.assert_called_once_with()


def test_context_manager_closes_windows_after_error(
    display_config: DisplayConfig,
) -> None:
    backend = Mock()
    view = OpenCVView(display_config, backend=backend)

    with pytest.raises(RuntimeError, match="fallo del ciclo"):
        with view:
            raise RuntimeError("fallo del ciclo")

    backend.destroyAllWindows.assert_called_once_with()


def test_wraps_window_errors(display_config: DisplayConfig) -> None:
    backend = Mock()
    backend.imshow.side_effect = RuntimeError("ventana no disponible")
    view = OpenCVView(display_config, backend=backend)

    with pytest.raises(OpenCVViewError, match="ventana no disponible"):
        view.show(
            np.zeros((2, 2, 3), dtype=np.uint8),
            fps=0.0,
            hand_count=0,
        )


def test_rejects_invalid_frame(display_config: DisplayConfig) -> None:
    view = OpenCVView(display_config, backend=Mock())

    with pytest.raises(OpenCVViewError, match="forma"):
        view.show(np.zeros((2, 2), dtype=np.uint8), fps=0.0, hand_count=0)
