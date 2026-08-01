import pytest

from gesture_matcher.recognition.gesture_matcher import (
    UNKNOWN_GESTURE_LABEL,
    MatchResult,
)
from gesture_matcher.recognition.temporal_filter import TemporalFilter
from gesture_matcher.utils.config_loader import TemporalFilterConfig


def accepted(
    gesture_id: str,
    *,
    similarity: float = 0.95,
    image_path: str | None = None,
) -> MatchResult:
    return MatchResult(
        gesture_id=gesture_id,
        label=gesture_id.replace("_", " ").title(),
        similarity=similarity,
        display_image_path=image_path,
        accepted=True,
    )


def unknown(similarity: float = 0.2) -> MatchResult:
    return MatchResult(
        gesture_id=None,
        label=UNKNOWN_GESTURE_LABEL,
        similarity=similarity,
        display_image_path=None,
        accepted=False,
    )


def make_filter(
    *,
    window_size: int = 7,
    stable_frames: int = 5,
    min_consecutive_frames: int = 3,
    hold_frames: int = 3,
    hysteresis_frames: int = 1,
) -> TemporalFilter:
    return TemporalFilter(
        TemporalFilterConfig(
            window_size=window_size,
            stable_frames=stable_frames,
            min_consecutive_frames=min_consecutive_frames,
            hold_frames=hold_frames,
            hysteresis_frames=hysteresis_frames,
        )
    )


def test_confirms_dominant_gesture_from_example_sequence() -> None:
    temporal_filter = make_filter()
    sequence = [
        accepted("victory"),
        accepted("victory"),
        unknown(),
        accepted("victory"),
        accepted("victory"),
        accepted("victory"),
        accepted("victory"),
        accepted("victory"),
    ]

    outputs = [temporal_filter.update(result) for result in sequence]

    assert all(not result.accepted for result in outputs[:5])
    assert all(result.gesture_id == "victory" for result in outputs[5:])


def test_incomplete_window_without_enough_votes_remains_unknown() -> None:
    temporal_filter = make_filter()

    outputs = [temporal_filter.update(accepted("victory")) for _ in range(4)]

    assert all(not result.accepted for result in outputs)


def test_requires_consecutive_frames_after_reaching_vote_count() -> None:
    temporal_filter = make_filter(
        stable_frames=3,
        min_consecutive_frames=3,
    )
    interrupted = [
        accepted("victory"),
        unknown(),
        accepted("victory"),
        unknown(),
        accepted("victory"),
        accepted("victory"),
    ]

    outputs = [temporal_filter.update(result) for result in interrupted]
    confirmed = temporal_filter.update(accepted("victory"))

    assert all(not result.accepted for result in outputs)
    assert confirmed.gesture_id == "victory"


def test_unstable_alternation_is_not_confirmed() -> None:
    temporal_filter = make_filter(
        window_size=5,
        stable_frames=3,
        min_consecutive_frames=2,
    )
    sequence = [
        accepted("victory"),
        accepted("thumbs_up"),
    ] * 4

    outputs = [temporal_filter.update(result) for result in sequence]

    assert all(not result.accepted for result in outputs)


def test_switches_only_after_new_gesture_is_dominant_and_consecutive() -> None:
    temporal_filter = make_filter(
        stable_frames=3,
        min_consecutive_frames=2,
        hold_frames=1,
    )
    for _ in range(3):
        current = temporal_filter.update(accepted("victory"))
    assert current.gesture_id == "victory"

    transition = [temporal_filter.update(accepted("thumbs_up")) for _ in range(3)]
    switched = temporal_filter.update(accepted("thumbs_up"))

    assert all(result.gesture_id == "victory" for result in transition)
    assert switched.gesture_id == "thumbs_up"


def test_holds_last_result_for_configured_absence() -> None:
    temporal_filter = make_filter(
        window_size=2,
        stable_frames=2,
        min_consecutive_frames=2,
        hold_frames=2,
        hysteresis_frames=0,
    )
    temporal_filter.update(accepted("victory"))
    temporal_filter.update(accepted("victory"))

    held = [temporal_filter.update(unknown()) for _ in range(2)]
    expired = temporal_filter.update(unknown())

    assert all(result.gesture_id == "victory" for result in held)
    assert not expired.accepted


def test_hysteresis_keeps_active_result_below_activation_threshold() -> None:
    temporal_filter = make_filter(
        window_size=5,
        stable_frames=3,
        min_consecutive_frames=2,
        hold_frames=0,
        hysteresis_frames=1,
    )
    for _ in range(3):
        temporal_filter.update(accepted("victory"))

    dropouts = [temporal_filter.update(unknown()) for _ in range(3)]
    expired = temporal_filter.update(unknown())

    assert all(result.gesture_id == "victory" for result in dropouts)
    assert not expired.accepted


def test_transient_gesture_does_not_change_display_image() -> None:
    temporal_filter = make_filter(
        window_size=3,
        stable_frames=2,
        min_consecutive_frames=2,
        hold_frames=1,
        hysteresis_frames=0,
    )
    victory = accepted("victory", image_path="assets/display_images/victory.png")
    temporal_filter.update(victory)
    temporal_filter.update(victory)

    output = temporal_filter.update(
        accepted(
            "thumbs_up",
            image_path="assets/display_images/thumbs_up.png",
        )
    )

    assert output.gesture_id == "victory"
    assert output.display_image_path == "assets/display_images/victory.png"


def test_reset_requires_confirmation_again() -> None:
    temporal_filter = make_filter(
        window_size=2,
        stable_frames=2,
        min_consecutive_frames=2,
        hold_frames=0,
        hysteresis_frames=0,
    )
    temporal_filter.update(accepted("victory"))
    confirmed = temporal_filter.update(accepted("victory"))
    temporal_filter.reset()

    after_reset = temporal_filter.update(accepted("victory"))

    assert confirmed.accepted
    assert not after_reset.accepted


def test_unknown_input_remains_unknown_and_preserves_score() -> None:
    result = make_filter().update(unknown(0.42))

    assert not result.accepted
    assert result.gesture_id is None
    assert result.label == UNKNOWN_GESTURE_LABEL
    assert result.similarity == pytest.approx(0.42)
    assert result.display_image_path is None


def test_rejects_invalid_direct_configuration() -> None:
    config = TemporalFilterConfig(
        window_size=0,
        stable_frames=1,
        min_consecutive_frames=1,
        hold_frames=0,
        hysteresis_frames=0,
    )

    with pytest.raises(ValueError, match="window_size"):
        TemporalFilter(config)
