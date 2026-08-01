"""Estabilización temporal de resultados de reconocimiento estático."""

import logging
import math
from collections import Counter, deque

from gesture_matcher.recognition.gesture_matcher import (
    UNKNOWN_GESTURE_LABEL,
    MatchResult,
)
from gesture_matcher.utils.config_loader import TemporalFilterConfig

LOGGER = logging.getLogger(__name__)


class TemporalFilter:
    """Confirma señas dominantes y amortigua variaciones entre fotogramas.

    Una seña nueva debe alcanzar ``stable_frames`` votos dentro de la ventana y
    aparecer al final durante ``min_consecutive_frames``. Una seña ya confirmada
    se conserva con un umbral menor, definido por ``hysteresis_frames``, o durante
    ``hold_frames`` ausencias consecutivas.
    """

    def __init__(
        self,
        config: TemporalFilterConfig,
        *,
        logger: logging.Logger = LOGGER,
    ) -> None:
        _validate_config(config)
        self._config = config
        self._logger = logger
        self._history: deque[MatchResult] = deque(maxlen=config.window_size)
        self._active_result: MatchResult | None = None
        self._frames_without_active = 0

    def update(self, result: MatchResult) -> MatchResult:
        """Agrega un resultado y devuelve la salida temporalmente estabilizada."""
        if not isinstance(result, MatchResult):
            raise TypeError("result debe ser una instancia de MatchResult.")

        self._history.append(result)
        counts = self._gesture_counts()
        candidate_id = self._confirmed_candidate(counts)
        current_id = _accepted_gesture_id(result)

        if self._active_result is None:
            if candidate_id is not None:
                return self._activate(self._latest_result(candidate_id))
            return _unknown_from(result)

        active_id = self._active_result.gesture_id
        if active_id is None:
            self.reset()
            return _unknown_from(result)

        if current_id == active_id:
            self._active_result = result
            self._frames_without_active = 0
        else:
            self._frames_without_active += 1

        if candidate_id is not None and candidate_id != active_id:
            return self._activate(self._latest_result(candidate_id))

        retention_threshold = max(
            1,
            self._config.stable_frames - self._config.hysteresis_frames,
        )
        if (
            counts.get(active_id, 0) >= retention_threshold
            or self._frames_without_active <= self._config.hold_frames
        ):
            return self._active_result

        self._logger.debug(
            "Seña temporal desactivada gesture_id=%s tras %d fotogramas ausente.",
            active_id,
            self._frames_without_active,
        )
        self._active_result = None
        self._frames_without_active = 0
        return _unknown_from(result)

    def reset(self) -> None:
        """Vacía la ventana y elimina cualquier seña confirmada."""
        self._history.clear()
        self._active_result = None
        self._frames_without_active = 0

    def _gesture_counts(self) -> Counter[str]:
        return Counter(
            gesture_id
            for item in self._history
            if (gesture_id := _accepted_gesture_id(item)) is not None
        )

    def _confirmed_candidate(self, counts: Counter[str]) -> str | None:
        if not counts:
            return None

        highest_count = max(counts.values())
        leaders = [
            gesture_id for gesture_id, count in counts.items() if count == highest_count
        ]
        if len(leaders) != 1 or highest_count < self._config.stable_frames:
            return None

        candidate_id = leaders[0]
        if self._trailing_count(candidate_id) < self._config.min_consecutive_frames:
            return None
        return candidate_id

    def _trailing_count(self, gesture_id: str) -> int:
        count = 0
        for item in reversed(self._history):
            if _accepted_gesture_id(item) != gesture_id:
                break
            count += 1
        return count

    def _latest_result(self, gesture_id: str) -> MatchResult:
        for item in reversed(self._history):
            if _accepted_gesture_id(item) == gesture_id:
                return item
        raise RuntimeError(
            f"No se encontró el resultado temporal esperado: {gesture_id}."
        )

    def _activate(self, result: MatchResult) -> MatchResult:
        previous_id = (
            self._active_result.gesture_id if self._active_result is not None else None
        )
        self._active_result = result
        self._frames_without_active = 0
        self._logger.debug(
            "Seña temporal confirmada previous=%s current=%s.",
            previous_id,
            result.gesture_id,
        )
        return result


def _accepted_gesture_id(result: MatchResult) -> str | None:
    if result.accepted and isinstance(result.gesture_id, str) and result.gesture_id:
        return result.gesture_id
    return None


def _unknown_from(result: MatchResult) -> MatchResult:
    try:
        similarity = float(result.similarity)
    except (TypeError, ValueError, OverflowError):
        similarity = 0.0
    if not math.isfinite(similarity):
        similarity = 0.0
    return MatchResult(
        gesture_id=None,
        label=UNKNOWN_GESTURE_LABEL,
        similarity=similarity,
        display_image_path=None,
        accepted=False,
    )


def _validate_config(config: TemporalFilterConfig) -> None:
    integer_fields = {
        "window_size": config.window_size,
        "stable_frames": config.stable_frames,
        "min_consecutive_frames": config.min_consecutive_frames,
        "hold_frames": config.hold_frames,
        "hysteresis_frames": config.hysteresis_frames,
    }
    if any(type(value) is not int for value in integer_fields.values()):
        raise ValueError("Los parámetros de TemporalFilter deben ser enteros.")
    if config.window_size < 1:
        raise ValueError("window_size debe ser mayor o igual que 1.")
    if not 1 <= config.stable_frames <= config.window_size:
        raise ValueError("stable_frames debe estar entre 1 y window_size.")
    if not 1 <= config.min_consecutive_frames <= config.window_size:
        raise ValueError("min_consecutive_frames debe estar entre 1 y window_size.")
    if config.hold_frames < 0:
        raise ValueError("hold_frames debe ser mayor o igual que 0.")
    if not 0 <= config.hysteresis_frames < config.stable_frames:
        raise ValueError("hysteresis_frames debe estar entre 0 y stable_frames - 1.")
