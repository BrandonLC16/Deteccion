from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from gesture_matcher.recognition.gesture_matcher import (
    UNKNOWN_GESTURE_LABEL,
    MatchResult,
)
from gesture_matcher.ui.opencv_view import (
    CAMERA_WIDTH_RATIO,
    INITIAL_WINDOW_HEIGHT,
    INITIAL_WINDOW_WIDTH,
    MAX_PANEL_HEIGHT,
    MAX_PANEL_WIDTH,
    WINDOW_TITLE,
    OpenCVView,
    OpenCVViewError,
    calculate_layout,
)
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
    frame = np.full((20, 20, 3), 25, dtype=np.uint8)

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
        f"Pose detectada: {UNKNOWN_GESTURE_LABEL}",
        "Similitud: 25.0 %",
    ]
    displayed = backend.imshow.call_args.args[1]
    layout = calculate_layout(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT)
    assert backend.imshow.call_args.args[0] == WINDOW_TITLE
    assert displayed.shape == (INITIAL_WINDOW_HEIGHT, INITIAL_WINDOW_WIDTH, 3)
    camera_region = displayed[
        layout.camera_slot.y : layout.camera_slot.bottom,
        layout.camera_slot.x : layout.camera_slot.right,
    ]
    assert np.any(camera_region == 25)
    backend.namedWindow.assert_called_once_with(
        WINDOW_TITLE,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
    )
    backend.resizeWindow.assert_called_once_with(
        WINDOW_TITLE,
        INITIAL_WINDOW_WIDTH,
        INITIAL_WINDOW_HEIGHT,
    )
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
    assert drawn_text == [
        "Manos: 0",
        f"Pose detectada: {UNKNOWN_GESTURE_LABEL}",
        "Similitud: 25.0 %",
    ]


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
    layout = calculate_layout(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT)
    image_region = displayed[
        layout.image_slot.y : layout.image_slot.bottom,
        layout.image_slot.x : layout.image_slot.right,
    ]
    assert np.any(image_region == 200)


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
    assert "Pose detectada: Victory" in drawn_text
    assert "Similitud: 93.0 %" in drawn_text
    assert "Sin imagen asociada" in drawn_text


def test_layout_is_compact_centered_and_uses_requested_proportions() -> None:
    layout = calculate_layout(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT)

    assert layout.panel.width == MAX_PANEL_WIDTH
    assert layout.panel.height == MAX_PANEL_HEIGHT
    assert abs(layout.panel.x - (layout.canvas_width - layout.panel.right)) <= 1
    assert abs(layout.panel.y - (layout.canvas_height - layout.panel.bottom)) <= 1
    assert layout.image_slot.height == layout.camera_slot.height
    sections_width = layout.camera_slot.width + layout.image_slot.width
    assert layout.camera_slot.width / sections_width == pytest.approx(
        CAMERA_WIDTH_RATIO,
        abs=0.01,
    )
    assert layout.image_slot.width / sections_width == pytest.approx(
        1 - CAMERA_WIDTH_RATIO,
        abs=0.01,
    )
    assert layout.image_slot.width > layout.camera_slot.width
    assert layout.image_slot.x - layout.camera_slot.right == layout.gap


@pytest.mark.parametrize(
    ("window_width", "window_height"),
    [(800, 500), (1400, 900), (500, 300)],
)
def test_layout_adapts_to_window_size_and_remains_centered(
    window_width: int,
    window_height: int,
) -> None:
    layout = calculate_layout(window_width, window_height)

    assert layout.panel.width <= MAX_PANEL_WIDTH
    assert layout.panel.height <= MAX_PANEL_HEIGHT
    assert layout.panel.x > 0
    assert layout.panel.y > 0
    assert abs(layout.panel.x - (layout.canvas_width - layout.panel.right)) <= 1
    assert abs(layout.panel.y - (layout.canvas_height - layout.panel.bottom)) <= 1
    assert layout.camera_slot.right < layout.image_slot.right <= layout.panel.right
    assert layout.image_slot.height == layout.camera_slot.height
    sections_width = layout.camera_slot.width + layout.image_slot.width
    assert layout.camera_slot.width / sections_width == pytest.approx(
        CAMERA_WIDTH_RATIO,
        abs=0.01,
    )
    assert layout.image_slot.width / sections_width == pytest.approx(
        1 - CAMERA_WIDTH_RATIO,
        abs=0.01,
    )


@pytest.mark.parametrize(
    ("window_width", "window_height"),
    [(0, 500), (500, 0), (-1, 500), (500.0, 300)],
)
def test_layout_rejects_invalid_window_dimensions(
    window_width: object,
    window_height: object,
) -> None:
    with pytest.raises(OpenCVViewError, match="ventana"):
        calculate_layout(window_width, window_height)  # type: ignore[arg-type]


def test_view_recomposes_canvas_after_window_resize(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    backend = Mock()
    backend.waitKey.return_value = -1
    backend.getWindowImageRect.side_effect = [
        (0, 0, 800, 500),
        (0, 0, 1200, 800),
    ]
    view = OpenCVView(display_config, image_cache, backend=backend)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    view.show(frame, fps=30.0, hand_count=1, result=unknown_result())
    first_display = backend.imshow.call_args.args[1]
    view.show(frame, fps=30.0, hand_count=1, result=unknown_result())
    second_display = backend.imshow.call_args.args[1]

    assert first_display.shape == (500, 800, 3)
    assert second_display.shape == (800, 1200, 3)


def test_camera_keeps_its_aspect_ratio_inside_smaller_section(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    backend = Mock()
    backend.waitKey.return_value = -1
    view = OpenCVView(display_config, image_cache, backend=backend)
    frame = np.full((90, 160, 3), 255, dtype=np.uint8)

    view.show(frame, fps=30.0, hand_count=1, result=unknown_result())

    displayed = backend.imshow.call_args.args[1]
    layout = calculate_layout(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT)
    camera_region = displayed[
        layout.camera_slot.y : layout.camera_slot.bottom,
        layout.camera_slot.x : layout.camera_slot.right,
    ]
    white_pixels = np.all(camera_region == 255, axis=2)
    rows, columns = np.where(white_pixels)
    rendered_height = rows.max() - rows.min() + 1
    rendered_width = columns.max() - columns.min() + 1

    assert rendered_width / rendered_height == pytest.approx(16 / 9, rel=0.04)
    assert layout.image_slot.width > layout.camera_slot.width
    assert layout.image_slot.height == layout.camera_slot.height


def test_camera_and_presentation_use_38_62_sections(
    display_config: DisplayConfig,
    image_cache: Mock,
) -> None:
    backend = Mock()
    backend.waitKey.return_value = -1
    image_cache.get.return_value = np.full((100, 100, 3), 200, dtype=np.uint8)
    view = OpenCVView(display_config, image_cache, backend=backend)

    view.show(
        np.full((90, 160, 3), 100, dtype=np.uint8),
        fps=30.0,
        hand_count=1,
        result=known_result(display_image_path="assets/display_images/victory.jpg"),
    )

    displayed = backend.imshow.call_args.args[1]
    layout = calculate_layout(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT)
    camera_region = displayed[
        layout.camera_slot.y : layout.camera_slot.bottom,
        layout.camera_slot.x : layout.camera_slot.right,
    ]
    image_region = displayed[
        layout.image_slot.y : layout.image_slot.bottom,
        layout.image_slot.x : layout.image_slot.right,
    ]

    sections_width = camera_region.shape[1] + image_region.shape[1]
    assert camera_region.shape[0] == image_region.shape[0]
    assert camera_region.shape[1] / sections_width == pytest.approx(
        CAMERA_WIDTH_RATIO,
        abs=0.01,
    )
    assert image_region.shape[1] / sections_width == pytest.approx(
        1 - CAMERA_WIDTH_RATIO,
        abs=0.01,
    )
    assert np.any(camera_region == 100)
    assert np.any(image_region == 200)
