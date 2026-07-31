import numpy as np
import pytest

from gesture_matcher.recognition.feature_extractor import (
    FEATURE_VECTOR_SIZE,
    FeatureExtractionError,
    FeatureExtractor,
)
from gesture_matcher.recognition.landmark_normalizer import LandmarkNormalizer


@pytest.fixture
def landmarks() -> np.ndarray:
    values = np.zeros((21, 3), dtype=np.float32)
    for index in range(1, 21):
        values[index] = [
            index * 0.02,
            (index % 4 + 1) * 0.03,
            -(index % 3) * 0.01,
        ]
    return values


def test_extracts_flat_normalized_feature_vector(landmarks: np.ndarray) -> None:
    normalizer = LandmarkNormalizer()
    extractor = FeatureExtractor(normalizer)

    result = extractor.extract(landmarks, "Right")

    expected = normalizer.normalize(landmarks, "Right").reshape(-1)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (FEATURE_VECTOR_SIZE,)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert not result.flags.writeable


def test_features_preserve_translation_and_scale_invariance(
    landmarks: np.ndarray,
) -> None:
    extractor = FeatureExtractor()

    original = extractor.extract(landmarks, "Right")
    transformed = extractor.extract(
        landmarks * 3.25 + np.array([7.0, -2.0, 4.0]),
        "Right",
    )

    np.testing.assert_allclose(transformed, original, atol=1e-6)


def test_features_canonicalize_left_handedness(landmarks: np.ndarray) -> None:
    left_landmarks = landmarks.copy()
    left_landmarks[:, 0] *= -1
    extractor = FeatureExtractor()

    right = extractor.extract(landmarks, "Right")
    left = extractor.extract(left_landmarks, "Left")

    np.testing.assert_allclose(left, right, atol=1e-6)


@pytest.mark.parametrize(
    "invalid_landmarks",
    [
        np.zeros((20, 3)),
        np.zeros((21, 2)),
        np.zeros(63),
    ],
)
def test_rejects_incorrect_landmark_dimensions(
    invalid_landmarks: np.ndarray,
) -> None:
    with pytest.raises(FeatureExtractionError, match=r"forma \(21, 3\)"):
        FeatureExtractor().extract(invalid_landmarks)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_rejects_invalid_landmark_values(
    landmarks: np.ndarray,
    invalid_value: float,
) -> None:
    invalid_landmarks = landmarks.copy()
    invalid_landmarks[10, 2] = invalid_value

    with pytest.raises(FeatureExtractionError, match="NaN o infinitos"):
        FeatureExtractor().extract(invalid_landmarks)
