from unittest.mock import Mock

import numpy as np

from gesture_matcher.recognition.feature_extractor import (
    FeatureExtractionError,
    HandFeatureVector,
)
from gesture_matcher.recognition.gesture_matcher import MatchResult
from gesture_matcher.recognition.recognition_pipeline import RecognitionPipeline
from gesture_matcher.vision.hand_detector import HandObservation


def observation() -> HandObservation:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    return HandObservation(
        landmarks=landmarks,
        world_landmarks=landmarks,
        handedness="Right",
        handedness_score=0.99,
    )


def known_result() -> MatchResult:
    return MatchResult(
        gesture_id="victory",
        label="Victory",
        similarity=0.95,
        display_image_path="assets/display_images/victory.jpg",
        accepted=True,
    )


def test_extracts_matches_and_stabilizes_detected_hands() -> None:
    extractor = Mock()
    matcher = Mock()
    temporal_filter = Mock()
    features = HandFeatureVector(
        vector=np.ones(63, dtype=np.float32),
        hand_count=1,
        handedness=("Right",),
    )
    stable = known_result()
    extractor.extract_hands.return_value = features
    matcher.match.return_value = known_result()
    temporal_filter.update.return_value = stable
    pipeline = RecognitionPipeline(extractor, matcher, temporal_filter)
    hands = (observation(),)

    result = pipeline.recognize(hands)

    assert result is stable
    extractor.extract_hands.assert_called_once_with(hands)
    matcher.match.assert_called_once_with(features.vector)
    temporal_filter.update.assert_called_once_with(matcher.match.return_value)


def test_no_hands_updates_filter_with_unknown_result() -> None:
    extractor = Mock()
    matcher = Mock()
    temporal_filter = Mock()
    temporal_filter.update.side_effect = lambda result: result
    pipeline = RecognitionPipeline(extractor, matcher, temporal_filter)

    result = pipeline.recognize(())

    assert not result.accepted
    assert result.gesture_id is None
    extractor.extract_hands.assert_not_called()
    matcher.match.assert_not_called()
    temporal_filter.update.assert_called_once()


def test_invalid_hands_become_unknown_without_matching() -> None:
    extractor = Mock()
    extractor.extract_hands.side_effect = FeatureExtractionError("lateralidad ambigua")
    matcher = Mock()
    temporal_filter = Mock()
    temporal_filter.update.side_effect = lambda result: result
    pipeline = RecognitionPipeline(extractor, matcher, temporal_filter)

    result = pipeline.recognize((observation(),))

    assert not result.accepted
    matcher.match.assert_not_called()
    temporal_filter.update.assert_called_once()


def test_reset_delegates_to_temporal_filter() -> None:
    temporal_filter = Mock()
    pipeline = RecognitionPipeline(Mock(), Mock(), temporal_filter)

    pipeline.reset()

    temporal_filter.reset.assert_called_once_with()
