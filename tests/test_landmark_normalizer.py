import numpy as np
import pytest

from gesture_matcher.recognition.landmark_normalizer import (
    LandmarkNormalizationError,
    LandmarkNormalizer,
)


@pytest.fixture
def landmarks() -> np.ndarray:
    values = np.zeros((21, 3), dtype=np.float64)
    values[0] = [0.25, 0.40, -0.10]
    for index in range(1, 21):
        values[index] = values[0] + [
            index * 0.012,
            (index % 5 + 1) * 0.018,
            -(index % 4) * 0.007,
        ]
    return values


def test_is_invariant_to_translation(landmarks: np.ndarray) -> None:
    normalizer = LandmarkNormalizer()

    original = normalizer.normalize(landmarks, "Right")
    translated = normalizer.normalize(
        landmarks + np.array([15.0, -8.0, 3.5]),
        "Right",
    )

    np.testing.assert_allclose(translated, original, atol=1e-6)
    np.testing.assert_array_equal(original[0], np.zeros(3, dtype=np.float32))


def test_is_invariant_to_uniform_scale(landmarks: np.ndarray) -> None:
    normalizer = LandmarkNormalizer()

    original = normalizer.normalize(landmarks, "Right")
    scaled = normalizer.normalize(landmarks * 4.75, "Right")

    np.testing.assert_allclose(scaled, original, atol=1e-6)
    assert np.max(np.linalg.norm(original, axis=1)) == pytest.approx(1.0)


def test_mirrors_left_hand_to_right_hand_geometry(landmarks: np.ndarray) -> None:
    left_landmarks = landmarks.copy()
    wrist_x = left_landmarks[0, 0]
    left_landmarks[:, 0] = 2 * wrist_x - left_landmarks[:, 0]
    normalizer = LandmarkNormalizer(mirror_left_hand=True)

    right = normalizer.normalize(landmarks, "Right")
    left = normalizer.normalize(left_landmarks, "Left")

    np.testing.assert_allclose(left, right, atol=1e-6)


def test_can_preserve_lateral_geometry(landmarks: np.ndarray) -> None:
    left_landmarks = landmarks.copy()
    wrist_x = left_landmarks[0, 0]
    left_landmarks[:, 0] = 2 * wrist_x - left_landmarks[:, 0]
    normalizer = LandmarkNormalizer(mirror_left_hand=False)

    right = normalizer.normalize(landmarks, "Right")
    left = normalizer.normalize(left_landmarks, "Left")

    np.testing.assert_allclose(left[:, 0], -right[:, 0], atol=1e-6)
    np.testing.assert_allclose(left[:, 1:], right[:, 1:], atol=1e-6)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_values(
    landmarks: np.ndarray,
    invalid_value: float,
) -> None:
    invalid_landmarks = landmarks.copy()
    invalid_landmarks[5, 1] = invalid_value

    with pytest.raises(LandmarkNormalizationError, match="NaN o infinitos"):
        LandmarkNormalizer().normalize(invalid_landmarks)


def test_rejects_landmarks_without_scale() -> None:
    collapsed = np.ones((21, 3), dtype=np.float32)

    with pytest.raises(LandmarkNormalizationError, match="escala nula"):
        LandmarkNormalizer().normalize(collapsed)


@pytest.mark.parametrize(
    "invalid_landmarks",
    [
        np.empty((0, 3)),
        np.zeros((20, 3)),
        np.zeros((21, 2)),
        np.zeros(63),
        np.zeros((1, 21, 3)),
    ],
)
def test_rejects_incorrect_dimensions(invalid_landmarks: np.ndarray) -> None:
    with pytest.raises(LandmarkNormalizationError, match=r"forma \(21, 3\)"):
        LandmarkNormalizer().normalize(invalid_landmarks)


@pytest.mark.parametrize("handedness", ["unknown", "", 1])
def test_rejects_invalid_handedness(
    landmarks: np.ndarray,
    handedness: object,
) -> None:
    with pytest.raises(LandmarkNormalizationError, match="[Ll]ateralidad"):
        LandmarkNormalizer().normalize(landmarks, handedness)  # type: ignore[arg-type]


def test_returns_float32_contiguous_read_only_array(
    landmarks: np.ndarray,
) -> None:
    result = LandmarkNormalizer().normalize(landmarks)

    assert result.shape == (21, 3)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert not result.flags.writeable
