import numpy as np
import pytest

from gesture_matcher.recognition.feature_extractor import (
    CANONICAL_HAND_ORDER,
    FEATURE_VECTOR_SIZE,
    TWO_HAND_RELATIVE_FEATURE_SIZE,
    FeatureExtractionError,
    FeatureExtractor,
)
from gesture_matcher.recognition.landmark_normalizer import LandmarkNormalizer
from gesture_matcher.vision.hand_detector import HandObservation


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


def make_hand(
    landmarks: np.ndarray,
    handedness: str | None,
    *,
    wrist_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> HandObservation:
    shifted = landmarks + np.asarray(wrist_offset, dtype=np.float32)
    return HandObservation(
        landmarks=shifted,
        world_landmarks=shifted.copy(),
        handedness=handedness,
        handedness_score=0.95 if handedness is not None else None,
    )


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


def test_orders_two_hands_left_then_right_independently_of_input(
    landmarks: np.ndarray,
) -> None:
    left = make_hand(landmarks, "Left", wrist_offset=(0.1, 0.2, 0.0))
    right = make_hand(landmarks, "Right", wrist_offset=(0.7, 0.3, 0.0))
    extractor = FeatureExtractor()

    ordered = extractor.extract_hands([left, right])
    reversed_input = extractor.extract_hands([right, left])

    assert ordered.handedness == CANONICAL_HAND_ORDER
    assert ordered.hand_count == 2
    assert ordered.vector.shape == (
        FEATURE_VECTOR_SIZE * 2 + TWO_HAND_RELATIVE_FEATURE_SIZE,
    )
    np.testing.assert_array_equal(reversed_input.vector, ordered.vector)


def test_two_hand_relative_features_are_translation_and_scale_invariant(
    landmarks: np.ndarray,
) -> None:
    left = make_hand(landmarks, "Left", wrist_offset=(0.1, 0.2, 0.0))
    right = make_hand(landmarks, "Right", wrist_offset=(0.7, 0.3, 0.0))
    extractor = FeatureExtractor()
    original = extractor.extract_hands([right, left])

    transformed_hands = [
        HandObservation(
            landmarks=hand.landmarks * 3.0 + np.array([5.0, -2.0, 1.0]),
            world_landmarks=hand.world_landmarks,
            handedness=hand.handedness,
            handedness_score=hand.handedness_score,
        )
        for hand in (right, left)
    ]
    transformed = extractor.extract_hands(transformed_hands)

    np.testing.assert_allclose(transformed.vector, original.vector, atol=1e-5)


def test_extracts_single_hand_with_its_canonical_label(
    landmarks: np.ndarray,
) -> None:
    result = FeatureExtractor().extract_hands([make_hand(landmarks, "right")])

    assert result.hand_count == 1
    assert result.handedness == ("Right",)
    assert result.vector.shape == (FEATURE_VECTOR_SIZE,)


@pytest.mark.parametrize(
    "handedness",
    [
        ("Left", "Left"),
        ("Right", "Right"),
        ("Left", None),
    ],
)
def test_rejects_ambiguous_two_hand_laterality(
    landmarks: np.ndarray,
    handedness: tuple[str | None, str | None],
) -> None:
    hands = [make_hand(landmarks, label) for label in handedness]

    with pytest.raises(FeatureExtractionError, match="Left|lateralidad"):
        FeatureExtractor().extract_hands(hands)


@pytest.mark.parametrize("hand_count", [0, 3])
def test_rejects_unsupported_hand_count(
    landmarks: np.ndarray,
    hand_count: int,
) -> None:
    hands = [
        make_hand(landmarks, "Left" if index == 0 else "Right")
        for index in range(hand_count)
    ]

    with pytest.raises(FeatureExtractionError, match="una o dos manos"):
        FeatureExtractor().extract_hands(hands)
