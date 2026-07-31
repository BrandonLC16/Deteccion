import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from gesture_matcher.recognition.feature_extractor import FeatureExtractor
from gesture_matcher.recognition.template_builder import (
    TemplateBuilder,
    TemplateBuildError,
    TemplatePersistenceError,
    format_build_report,
    save_template_artifacts,
)
from gesture_matcher.vision.hand_detector import HandObservation


class MappingDetector:
    def __init__(
        self,
        detections: dict[int, tuple[HandObservation, ...]],
    ) -> None:
        self._detections = detections

    def detect(self, image: np.ndarray) -> tuple[HandObservation, ...]:
        return self._detections[int(image[0, 0, 0])]


def make_hand(
    handedness: str,
    *,
    wrist_x: float = 0.0,
) -> HandObservation:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [wrist_x, 0.2, 0.0]
    for index in range(1, 21):
        landmarks[index] = landmarks[0] + [
            index * 0.01,
            (index % 5 + 1) * 0.02,
            -(index % 4) * 0.005,
        ]
    return HandObservation(
        landmarks=landmarks,
        world_landmarks=landmarks.copy(),
        handedness=handedness,
        handedness_score=0.95,
    )


def add_reference(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"test image")
    return path


def mapping_loader(values: dict[str, int | None]):
    def load(path: Path) -> np.ndarray | None:
        value = values[path.name]
        if value is None:
            return None
        return np.full((2, 2, 3), value, dtype=np.uint8)

    return load


def test_builds_samples_and_reports_invalid_images(tmp_path: Path) -> None:
    references = tmp_path / "assets" / "reference_images"
    gesture = references / "victory"
    add_reference(gesture, "sample_01.jpeg")
    add_reference(gesture, "sample_02.png")
    add_reference(gesture, "no_hand.jpg")
    add_reference(gesture, "corrupt.jpeg")
    add_reference(gesture, "notes.txt")
    detector = MappingDetector(
        {
            1: (make_hand("Right"),),
            2: (make_hand("Right", wrist_x=0.2),),
            3: (),
        }
    )
    builder = TemplateBuilder(
        detector,
        FeatureExtractor(),
        image_loader=mapping_loader(
            {
                "sample_01.jpeg": 1,
                "sample_02.png": 2,
                "no_hand.jpg": 3,
                "corrupt.jpeg": None,
            }
        ),
    )

    result = builder.build(references)

    assert len(result.gestures) == 1
    assert result.gestures[0].gesture_id == "victory"
    assert result.gestures[0].feature_vectors.shape == (2, 63)
    assert result.accepted_count == 2
    assert result.rejected_count == 3
    reasons = [rejected.reason for rejected in result.rejected]
    assert any("No se detectó ninguna mano" in reason for reason in reasons)
    assert any("no pudo decodificar" in reason for reason in reasons)
    assert any("Extensión no permitida" in reason for reason in reasons)
    report = format_build_report(result, project_root=tmp_path)
    assert "Imágenes aceptadas: 2" in report
    assert "Referencias rechazadas: 3" in report
    assert "[OK] assets/reference_images/victory/sample_01.jpeg" in report
    assert "[RECHAZADA]" in report


def test_uses_majority_hand_count_and_rejects_inconsistent_sample(
    tmp_path: Path,
) -> None:
    references = tmp_path / "references"
    gesture = references / "character_pose"
    for name in ("one_a.jpeg", "one_b.jpeg", "two.jpeg"):
        add_reference(gesture, name)
    detector = MappingDetector(
        {
            1: (make_hand("Right"),),
            2: (make_hand("Left"),),
            3: (make_hand("Right", wrist_x=0.8), make_hand("Left")),
        }
    )
    builder = TemplateBuilder(
        detector,
        FeatureExtractor(),
        image_loader=mapping_loader({"one_a.jpeg": 1, "one_b.jpeg": 2, "two.jpeg": 3}),
    )

    result = builder.build(references)

    assert result.gestures[0].hand_count == 1
    assert result.gestures[0].feature_vectors.shape == (2, 63)
    assert result.accepted_count == 2
    assert result.rejected_count == 1
    assert "usa 1 mano(s)" in result.rejected[0].reason


def test_rejects_gesture_when_hand_count_is_tied(tmp_path: Path) -> None:
    references = tmp_path / "references"
    gesture = references / "ambiguous_pose"
    add_reference(gesture, "one.jpeg")
    add_reference(gesture, "two.jpeg")
    detector = MappingDetector(
        {
            1: (make_hand("Right"),),
            2: (make_hand("Left"), make_hand("Right", wrist_x=0.8)),
        }
    )
    builder = TemplateBuilder(
        detector,
        FeatureExtractor(),
        image_loader=mapping_loader({"one.jpeg": 1, "two.jpeg": 2}),
    )

    result = builder.build(references)

    assert result.gestures == ()
    assert result.accepted_count == 0
    assert result.rejected_count == 2
    assert all("empate" in rejected.reason for rejected in result.rejected)


def test_rejects_invalid_gesture_identifier(tmp_path: Path) -> None:
    references = tmp_path / "references"
    add_reference(references / "Bad Gesture", "sample.jpeg")
    builder = TemplateBuilder(
        MappingDetector({}),
        FeatureExtractor(),
        image_loader=pytest.fail,
    )

    result = builder.build(references)

    assert result.gestures == ()
    assert result.rejected_count == 1
    assert "snake_case" in result.rejected[0].reason


def test_rejects_missing_reference_directory(tmp_path: Path) -> None:
    builder = TemplateBuilder(MappingDetector({}), FeatureExtractor())

    with pytest.raises(TemplateBuildError, match="No existe"):
        builder.build(tmp_path / "missing")


def test_saves_two_hand_npz_and_versioned_metadata(tmp_path: Path) -> None:
    references = tmp_path / "assets" / "reference_images"
    gesture = references / "heart_hands"
    add_reference(gesture, "sample_01.jpeg")
    add_reference(gesture, "sample_02.jpeg")
    left = make_hand("Left", wrist_x=0.1)
    right = make_hand("Right", wrist_x=0.8)
    detector = MappingDetector({1: (right, left), 2: (left, right)})
    builder = TemplateBuilder(
        detector,
        FeatureExtractor(),
        image_loader=mapping_loader({"sample_01.jpeg": 1, "sample_02.jpeg": 2}),
    )
    result = builder.build(references)
    display_root = tmp_path / "assets" / "display_images"
    display_image = add_reference(display_root, "heart_hands.png")
    templates_path = tmp_path / "data" / "gesture_templates.npz"
    metadata_path = tmp_path / "data" / "gestures.json"

    metadata = save_template_artifacts(
        result,
        templates_path=templates_path,
        metadata_path=metadata_path,
        project_root=tmp_path,
        display_images_root=display_root,
        default_similarity_threshold=0.85,
        gesture_thresholds={"heart_hands": 0.9},
        mirror_left_hand=True,
        generated_at=datetime(2026, 7, 31, tzinfo=UTC),
    )

    assert display_image.is_file()
    with np.load(templates_path, allow_pickle=False) as stored:
        assert stored.files == ["heart_hands"]
        assert stored["heart_hands"].shape == (2, 129)
        np.testing.assert_array_equal(
            stored["heart_hands"][0],
            stored["heart_hands"][1],
        )
    stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert stored_metadata == metadata
    assert stored_metadata["format_version"] == 1
    assert stored_metadata["canonical_hand_order"] == ["Left", "Right"]
    assert stored_metadata["two_hand_relative_features"]["size"] == 3
    gesture_metadata = stored_metadata["gestures"][0]
    assert gesture_metadata["hand_count"] == 2
    assert gesture_metadata["feature_dimension"] == 129
    assert gesture_metadata["sample_count"] == 2
    assert gesture_metadata["similarity_threshold"] == pytest.approx(0.9)
    assert (
        gesture_metadata["display_image_path"]
        == "assets/display_images/heart_hands.png"
    )
    assert not list((tmp_path / "data").glob("*.tmp"))


def test_does_not_write_empty_template_artifacts(tmp_path: Path) -> None:
    result = TemplateBuilder(
        MappingDetector({1: ()}),
        FeatureExtractor(),
        image_loader=mapping_loader({"sample.jpeg": 1}),
    )
    references = tmp_path / "references"
    add_reference(references / "unknown_pose", "sample.jpeg")
    built = result.build(references)

    with pytest.raises(TemplatePersistenceError, match="plantillas vacías"):
        save_template_artifacts(
            built,
            templates_path=tmp_path / "data" / "templates.npz",
            metadata_path=tmp_path / "data" / "metadata.json",
            project_root=tmp_path,
            display_images_root=tmp_path / "display",
            default_similarity_threshold=0.85,
            gesture_thresholds={},
            mirror_left_hand=True,
        )

    assert not (tmp_path / "data").exists()
