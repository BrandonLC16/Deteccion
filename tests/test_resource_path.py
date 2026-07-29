from pathlib import Path

import pytest

from gesture_matcher.utils.resource_path import (
    ResourcePathError,
    get_project_root,
    resolve_project_path,
)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return tmp_path


def test_finds_project_root_from_nested_directory(project_root: Path) -> None:
    nested = project_root / "src" / "gesture_matcher"
    nested.mkdir(parents=True)

    assert get_project_root(nested) == project_root


def test_resolves_relative_resource(project_root: Path) -> None:
    expected = project_root / "config" / "config.yaml"

    assert (
        resolve_project_path("config/config.yaml", project_root=project_root)
        == expected
    )


def test_rejects_absolute_resource_path(project_root: Path) -> None:
    absolute = (project_root / "model.task").resolve()

    with pytest.raises(ResourcePathError, match="debe ser relativa"):
        resolve_project_path(absolute, project_root=project_root)


def test_reports_missing_required_resource(project_root: Path) -> None:
    with pytest.raises(ResourcePathError, match="No existe"):
        resolve_project_path(
            "models/missing.task",
            project_root=project_root,
            must_exist=True,
        )
