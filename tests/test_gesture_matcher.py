import logging

import numpy as np
import pytest

from gesture_matcher.recognition.gesture_matcher import (
    UNKNOWN_GESTURE_LABEL,
    GestureMatcher,
)
from gesture_matcher.recognition.template_repository import (
    GestureTemplate,
    TemplateRepository,
)


def unit_vector(index: int, dimension: int = 63) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[index] = 1.0
    return vector


def make_template(
    gesture_id: str,
    vectors: list[np.ndarray],
    *,
    hand_count: int = 1,
    threshold: float | None = None,
    display_image_path: str | None = None,
) -> GestureTemplate:
    return GestureTemplate(
        gesture_id=gesture_id,
        label=gesture_id.replace("_", " ").title(),
        hand_count=hand_count,
        feature_vectors=np.stack(vectors),
        display_image_path=display_image_path,
        similarity_threshold=threshold,
    )


def test_recognizes_identical_known_gesture() -> None:
    template = make_template(
        "victory",
        [unit_vector(0)],
        display_image_path="assets/display_images/victory.png",
    )
    matcher = GestureMatcher(
        TemplateRepository([template], default_similarity_threshold=0.85)
    )

    result = matcher.match(unit_vector(0))

    assert result.accepted
    assert result.gesture_id == "victory"
    assert result.label == "Victory"
    assert result.similarity == pytest.approx(1.0)
    assert result.display_image_path == "assets/display_images/victory.png"


def test_compares_all_samples_and_uses_best_score() -> None:
    victory = make_template(
        "victory",
        [unit_vector(0), unit_vector(1)],
    )
    thumbs_up = make_template("thumbs_up", [unit_vector(2)])
    matcher = GestureMatcher(TemplateRepository([victory, thumbs_up]))

    ranked = matcher.rank(unit_vector(1))

    assert [score.gesture_id for score in ranked] == ["victory", "thumbs_up"]
    assert ranked[0].similarity == pytest.approx(1.0)
    assert ranked[0].best_sample_index == 1
    assert ranked[1].similarity == pytest.approx(0.0)


def test_rejects_clearly_different_position_and_returns_score() -> None:
    matcher = GestureMatcher(
        TemplateRepository(
            [make_template("victory", [unit_vector(0)])],
            default_similarity_threshold=0.85,
        )
    )

    result = matcher.match(unit_vector(1))

    assert not result.accepted
    assert result.gesture_id is None
    assert result.label == UNKNOWN_GESTURE_LABEL
    assert result.similarity == pytest.approx(0.0)
    assert result.display_image_path is None


def test_uses_persisted_individual_threshold_before_global() -> None:
    live_vector = np.zeros(63, dtype=np.float32)
    live_vector[:2] = [0.8, 0.6]
    repository = TemplateRepository(
        [
            make_template(
                "victory",
                [unit_vector(0)],
                threshold=0.75,
            )
        ],
        default_similarity_threshold=0.9,
    )

    result = GestureMatcher(repository).match(live_vector)

    assert result.accepted
    assert result.similarity == pytest.approx(0.8)


def test_runtime_individual_threshold_overrides_persisted_threshold() -> None:
    live_vector = np.zeros(63, dtype=np.float32)
    live_vector[:2] = [0.8, 0.6]
    repository = TemplateRepository(
        [
            make_template(
                "victory",
                [unit_vector(0)],
                threshold=0.75,
            )
        ]
    )
    matcher = GestureMatcher(
        repository,
        gesture_thresholds={"victory": 0.85},
    )

    result = matcher.match(live_vector)

    assert not result.accepted
    assert result.similarity == pytest.approx(0.8)


def test_orders_tied_results_by_gesture_id() -> None:
    repository = TemplateRepository(
        [
            make_template("beta", [unit_vector(0)]),
            make_template("alpha", [unit_vector(0)]),
        ]
    )

    ranked = GestureMatcher(repository).rank(unit_vector(0))

    assert [score.gesture_id for score in ranked] == ["alpha", "beta"]


def test_does_not_compare_wrong_hand_count() -> None:
    repository = TemplateRepository([make_template("victory", [unit_vector(0)])])
    two_hand_vector = unit_vector(0, dimension=129)

    result = GestureMatcher(repository).match(two_hand_vector)

    assert not result.accepted
    assert result.gesture_id is None
    assert result.similarity == pytest.approx(0.0)


def test_recognizes_two_hand_template() -> None:
    template = make_template(
        "heart_hands",
        [unit_vector(4, dimension=129)],
        hand_count=2,
    )
    matcher = GestureMatcher(TemplateRepository([template]))

    result = matcher.match(unit_vector(4, dimension=129))

    assert result.accepted
    assert result.gesture_id == "heart_hands"
    assert result.similarity == pytest.approx(1.0)


def test_empty_repository_returns_unknown() -> None:
    result = GestureMatcher(TemplateRepository()).match(unit_vector(0))

    assert not result.accepted
    assert result.gesture_id is None
    assert result.similarity == pytest.approx(0.0)


@pytest.mark.parametrize(
    "invalid_features",
    [
        np.array([], dtype=np.float32),
        np.zeros(63, dtype=np.float32),
        np.zeros((1, 63), dtype=np.float32),
        np.full(63, np.nan, dtype=np.float32),
        np.full(63, np.inf, dtype=np.float32),
        np.ones(64, dtype=np.float32),
        None,
    ],
)
def test_invalid_or_empty_features_return_unknown_without_exception(
    invalid_features: object,
) -> None:
    repository = TemplateRepository([make_template("victory", [unit_vector(0)])])

    result = GestureMatcher(repository).match(invalid_features)  # type: ignore[arg-type]

    assert not result.accepted
    assert result.gesture_id is None
    assert result.similarity == pytest.approx(0.0)


def test_logs_every_sample_score_in_debug_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    repository = TemplateRepository(
        [
            make_template(
                "victory",
                [unit_vector(0), unit_vector(1)],
            )
        ]
    )
    matcher = GestureMatcher(repository)

    with caplog.at_level(
        logging.DEBUG,
        logger="gesture_matcher.recognition.gesture_matcher",
    ):
        matcher.match(unit_vector(0))

    messages = [
        record.getMessage()
        for record in caplog.records
        if "Puntuación gesture_id=victory" in record.getMessage()
    ]
    assert len(messages) == 2
    assert "sample=0" in messages[0]
    assert "sample=1" in messages[1]
