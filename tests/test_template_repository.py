import json
from pathlib import Path

import numpy as np
import pytest

from gesture_matcher.recognition.template_repository import (
    TemplateRepository,
    TemplateRepositoryError,
)


def feature_matrix(
    hand_count: int,
    sample_count: int = 1,
) -> np.ndarray:
    dimension = 63 if hand_count == 1 else 129
    base = np.arange(1, dimension + 1, dtype=np.float32)
    return np.stack([base + index for index in range(sample_count)])


def gesture_metadata(
    gesture_id: str,
    *,
    hand_count: int = 1,
    sample_count: int = 1,
    display_image_path: str | None = None,
    similarity_threshold: float | None = None,
) -> dict[str, object]:
    handedness = ["Right"] if hand_count == 1 else ["Left", "Right"]
    dimension = 63 if hand_count == 1 else 129
    return {
        "gesture_id": gesture_id,
        "label": gesture_id.replace("_", " ").title(),
        "gesture_type": "static",
        "template_key": gesture_id,
        "hand_count": hand_count,
        "handedness_variants": [handedness],
        "sample_count": sample_count,
        "feature_dimension": dimension,
        "similarity_threshold": similarity_threshold,
        "display_image_path": display_image_path,
        "samples": [
            {
                "template_row": index,
                "source_image_path": (
                    f"assets/reference_images/{gesture_id}/sample_{index}.jpeg"
                ),
                "handedness": handedness,
            }
            for index in range(sample_count)
        ],
    }


def write_artifacts(
    project_root: Path,
    *,
    gestures: list[dict[str, object]],
    arrays: dict[str, np.ndarray],
    metadata_updates: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    data_directory = project_root / "data"
    data_directory.mkdir(parents=True)
    templates_path = data_directory / "gesture_templates.npz"
    metadata_path = data_directory / "gestures.json"
    np.savez_compressed(templates_path, **arrays)
    metadata: dict[str, object] = {
        "format_version": 1,
        "feature_dtype": "float32",
        "feature_size_per_hand": 63,
        "canonical_hand_order": ["Left", "Right"],
        "two_hand_relative_features": {
            "size": 3,
            "definition": "right_wrist_minus_left_wrist_over_mean_hand_scale",
        },
        "default_similarity_threshold": 0.85,
        "gestures": gestures,
    }
    metadata.update(metadata_updates or {})
    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return templates_path, metadata_path


def test_loads_valid_one_and_two_hand_templates(tmp_path: Path) -> None:
    display_image = tmp_path / "assets" / "display_images" / "victory.png"
    display_image.parent.mkdir(parents=True)
    display_image.write_bytes(b"display")
    gestures = [
        gesture_metadata(
            "victory",
            sample_count=2,
            display_image_path="assets/display_images/victory.png",
            similarity_threshold=0.9,
        ),
        gesture_metadata("heart_hands", hand_count=2),
    ]
    templates_path, metadata_path = write_artifacts(
        tmp_path,
        gestures=gestures,
        arrays={
            "victory": feature_matrix(1, 2),
            "heart_hands": feature_matrix(2),
        },
    )

    repository = TemplateRepository.load(
        templates_path,
        metadata_path,
        project_root=tmp_path,
    )

    assert len(repository) == 2
    assert repository.default_similarity_threshold == pytest.approx(0.85)
    assert len(repository.for_hand_count(1)) == 1
    assert len(repository.for_hand_count(2)) == 1
    victory = repository.templates[0]
    assert victory.label == "Victory"
    assert victory.sample_count == 2
    assert victory.feature_dimension == 63
    assert victory.display_image_path == "assets/display_images/victory.png"
    assert victory.similarity_threshold == pytest.approx(0.9)
    assert not victory.feature_vectors.flags.writeable


@pytest.mark.parametrize("missing_name", ["templates", "metadata"])
def test_rejects_missing_artifact(
    tmp_path: Path,
    missing_name: str,
) -> None:
    templates_path = tmp_path / "templates.npz"
    metadata_path = tmp_path / "metadata.json"
    if missing_name != "templates":
        np.savez_compressed(templates_path)
    if missing_name != "metadata":
        metadata_path.write_text("{}", encoding="utf-8")

    with pytest.raises(TemplateRepositoryError, match="No se encontró"):
        TemplateRepository.load(
            templates_path,
            metadata_path,
            project_root=tmp_path,
        )


def test_rejects_corrupt_metadata(tmp_path: Path) -> None:
    templates_path = tmp_path / "templates.npz"
    metadata_path = tmp_path / "metadata.json"
    np.savez_compressed(templates_path)
    metadata_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(TemplateRepositoryError, match="metadatos"):
        TemplateRepository.load(
            templates_path,
            metadata_path,
            project_root=tmp_path,
        )


def test_rejects_incompatible_feature_dimension(tmp_path: Path) -> None:
    gesture = gesture_metadata("victory")
    gesture["feature_dimension"] = 129
    templates_path, metadata_path = write_artifacts(
        tmp_path,
        gestures=[gesture],
        arrays={"victory": feature_matrix(1)},
    )

    with pytest.raises(TemplateRepositoryError, match="feature_dimension"):
        TemplateRepository.load(
            templates_path,
            metadata_path,
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    "invalid_vectors",
    [
        np.zeros((1, 63), dtype=np.float32),
        np.full((1, 63), np.nan, dtype=np.float32),
        np.ones((1, 62), dtype=np.float32),
    ],
)
def test_rejects_invalid_template_values_or_shape(
    tmp_path: Path,
    invalid_vectors: np.ndarray,
) -> None:
    templates_path, metadata_path = write_artifacts(
        tmp_path,
        gestures=[gesture_metadata("victory")],
        arrays={"victory": invalid_vectors},
    )

    with pytest.raises(TemplateRepositoryError):
        TemplateRepository.load(
            templates_path,
            metadata_path,
            project_root=tmp_path,
        )


def test_rejects_missing_display_image(tmp_path: Path) -> None:
    gesture = gesture_metadata(
        "victory",
        display_image_path="assets/display_images/missing.png",
    )
    templates_path, metadata_path = write_artifacts(
        tmp_path,
        gestures=[gesture],
        arrays={"victory": feature_matrix(1)},
    )

    with pytest.raises(TemplateRepositoryError, match="imagen asociada"):
        TemplateRepository.load(
            templates_path,
            metadata_path,
            project_root=tmp_path,
        )


def test_allows_empty_repository_artifacts(tmp_path: Path) -> None:
    templates_path, metadata_path = write_artifacts(
        tmp_path,
        gestures=[],
        arrays={},
    )

    repository = TemplateRepository.load(
        templates_path,
        metadata_path,
        project_root=tmp_path,
    )

    assert len(repository) == 0
    assert repository.templates == ()


def test_rejects_npz_arrays_without_metadata(tmp_path: Path) -> None:
    templates_path, metadata_path = write_artifacts(
        tmp_path,
        gestures=[],
        arrays={"orphan": feature_matrix(1)},
    )

    with pytest.raises(TemplateRepositoryError, match="sin metadatos"):
        TemplateRepository.load(
            templates_path,
            metadata_path,
            project_root=tmp_path,
        )
