from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from gesture_matcher.recognition.gesture_matcher import (
    UNKNOWN_GESTURE_LABEL,
    MatchResult,
)
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


@pytest.fixture
def image_cache() -> Mock:
    return Mock()


def unknown_result(*, display_image_path: str | None = None) -> MatchResult:
    return MatchResult(
        gesture_id=None,
        label=UNKNOWN_GESTURE_LABEL,
        similarity=0.25,
        display_image_path=display_image_path,
        accepted=False,
    )


def known_result(*, display_image_path: str | None = None) -> MatchResult:
    return MatchResult(
        gesture_id="victory",
        label="Victory",
        similarity=0.93,
        display_image_path=display_image_path,
        accepted=True,
    )


def test_shows_video_fps_hand_count_and_unknown_result(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    backend = Mock()
    backend.waitKey.return_value = -1
    view = OpenCVView(display_config, image_cache, backend=backend)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    should_continue = view.show(
        frame,
        fps=29.75,
        hand_count=2,
        result=unknown_result(),
    )

    assert should_continue
    assert backend.putText.call_count == 4
    drawn_text = [call.args[1] for call in backend.putText.call_args_list]
    assert drawn_text == [
        "FPS: 29.8",
        "Manos: 2",
        UNKNOWN_GESTURE_LABEL,
        "Similitud: 25.0 %",
    ]
    displayed = backend.imshow.call_args.args[1]
    assert backend.imshow.call_args.args[0] == WINDOW_TITLE
    assert displayed.shape == (460, 380, 3)
    assert np.array_equal(displayed[:20, :20], frame)
    backend.waitKey.assert_called_once_with(1)
    image_cache.get.assert_not_called()


def test_respects_disabled_fps(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    backend = Mock()
    backend.waitKey.return_value = -1
    view = OpenCVView(
        replace(display_config, show_fps=False),
        image_cache,
        backend=backend,
    )

    view.show(
        np.zeros((2, 2, 3), dtype=np.uint8),
        fps=30.0,
        hand_count=0,
        result=unknown_result(),
    )

    drawn_text = [call.args[1] for call in backend.putText.call_args_list]
    assert drawn_text == ["Manos: 0", UNKNOWN_GESTURE_LABEL, "Similitud: 25.0 %"]


@pytest.mark.parametrize("key", [27, ord("q"), ord("Q")])
def test_stops_for_exit_keys(
    display_config: DisplayConfig,
    image_cache: Mock,
    key: int,
) -> None:
    backend = Mock()
    backend.waitKey.return_value = key
    view = OpenCVView(display_config, image_cache, backend=backend)

    assert not view.show(
        np.zeros((2, 2, 3), dtype=np.uint8),
        fps=0.0,
        hand_count=0,
        result=unknown_result(),
    )


def test_close_is_idempotent(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    backend = Mock()
    view = OpenCVView(display_config, image_cache, backend=backend)

    view.close()
    view.close()

    backend.destroyAllWindows.assert_called_once_with()


def test_context_manager_closes_windows_after_error(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    backend = Mock()
    view = OpenCVView(display_config, image_cache, backend=backend)

    with pytest.raises(RuntimeError, match="fallo del ciclo"):
        with view:
            raise RuntimeError("fallo del ciclo")

    backend.destroyAllWindows.assert_called_once_with()


def test_wraps_window_errors(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    backend = Mock()
    backend.imshow.side_effect = RuntimeError("ventana no disponible")
    view = OpenCVView(display_config, image_cache, backend=backend)

    with pytest.raises(OpenCVViewError, match="ventana no disponible"):
        view.show(
            np.zeros((2, 2, 3), dtype=np.uint8),
            fps=0.0,
            hand_count=0,
            result=unknown_result(),
        )


def test_rejects_invalid_frame(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    view = OpenCVView(display_config, image_cache, backend=Mock())

    with pytest.raises(OpenCVViewError, match="forma"):
        view.show(
            np.zeros((2, 2), dtype=np.uint8),
            fps=0.0,
            hand_count=0,
            result=unknown_result(),
        )


def test_image_is_loaded_only_for_accepted_result(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    backend = Mock()
    backend.waitKey.return_value = -1
    image_cache.get.return_value = np.full((320, 320, 3), 200, dtype=np.uint8)
    view = OpenCVView(display_config, image_cache, backend=backend)
    image_path = "assets/display_images/victory.jpg"

    view.show(
        np.zeros((20, 20, 3), dtype=np.uint8),
        fps=30.0,
        hand_count=1,
        result=unknown_result(display_image_path=image_path),
    )
    image_cache.get.assert_not_called()

    view.show(
        np.zeros((20, 20, 3), dtype=np.uint8),
        fps=30.0,
        hand_count=1,
        result=known_result(display_image_path=image_path),
    )

    image_cache.get.assert_called_once_with(image_path)
    displayed = backend.imshow.call_args.args[1]
    assert np.any(displayed[:, 20:] == 200)


def test_known_result_without_image_reports_missing_association(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    backend = Mock()
    backend.waitKey.return_value = -1
    view = OpenCVView(display_config, image_cache, backend=backend)

    view.show(
        np.zeros((20, 20, 3), dtype=np.uint8),
        fps=30.0,
        hand_count=1,
        result=known_result(),
    )

    drawn_text = [call.args[1] for call in backend.putText.call_args_list]
    assert "Victory" in drawn_text
    assert "Similitud: 93.0 %" in drawn_text
    assert "Sin imagen asociada" in drawn_text
