from unittest.mock import Mock

import numpy as np
import pytest

from gesture_matcher.vision.hand_detector import HandObservation
from gesture_matcher.vision.landmark_drawer import (
    HAND_CONNECTIONS,
    LandmarkDrawer,
    LandmarkDrawingError,
)


def make_hand(
    *,
    handedness: str | None = "Left",
    landmark_count: int = 21,
) -> HandObservation:
    landmarks = np.zeros((landmark_count, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(-0.1, 1.1, landmark_count)
    landmarks[:, 1] = np.linspace(0.0, 1.0, landmark_count)
    return HandObservation(
        landmarks=landmarks,
        world_landmarks=np.zeros((landmark_count, 3), dtype=np.float32),
        handedness=handedness,
        handedness_score=0.95 if handedness else None,
    )


def test_draws_landmarks_connections_and_labels_for_two_hands() -> None:
    backend = Mock()
    drawer = LandmarkDrawer(backend=backend)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    returned = drawer.draw(frame, [make_hand(), make_hand(handedness="Right")])

    assert returned is frame
    assert backend.line.call_count == len(HAND_CONNECTIONS) * 2
    assert backend.circle.call_count == 42
    assert backend.putText.call_count == 2


def test_clamps_normalized_coordinates_to_frame_bounds() -> None:
    backend = Mock()
    drawer = LandmarkDrawer(backend=backend)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    drawer.draw(frame, [make_hand()])

    points = [call.args[1] for call in backend.circle.call_args_list]
    assert all(0 <= x < 200 and 0 <= y < 100 for x, y in points)
    assert points[0][0] == 0
    assert points[-1][0] == 199


def test_skips_handedness_label_when_it_is_unavailable() -> None:
    backend = Mock()
    drawer = LandmarkDrawer(backend=backend)

    drawer.draw(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [make_hand(handedness=None)],
    )

    backend.putText.assert_not_called()


def test_rejects_incorrect_landmark_shape() -> None:
    drawer = LandmarkDrawer(backend=Mock())

    with pytest.raises(LandmarkDrawingError, match=r"forma \(21, 3\)"):
        drawer.draw(
            np.zeros((20, 20, 3), dtype=np.uint8),
            [make_hand(landmark_count=20)],
        )


def test_wraps_backend_errors() -> None:
    backend = Mock()
    backend.line.side_effect = RuntimeError("dibujo fallido")
    drawer = LandmarkDrawer(backend=backend)

    with pytest.raises(LandmarkDrawingError, match="dibujo fallido"):
        drawer.draw(np.zeros((20, 20, 3), dtype=np.uint8), [make_hand()])
