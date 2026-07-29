"""Resolución segura de recursos relativos a la raíz del proyecto."""

from pathlib import Path
from typing import Literal


class ResourcePathError(ValueError):
    """Indica que una ruta de recurso no es segura o no es válida."""


def get_project_root(start: Path | None = None) -> Path:
    """Encuentra la raíz recorriendo ancestros hasta hallar ``pyproject.toml``."""
    candidate = (start or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file():
            return directory

    raise ResourcePathError(
        f"No se encontró pyproject.toml desde {candidate}. "
        "Ejecuta la aplicación desde una copia completa del proyecto."
    )


def resolve_project_path(
    relative_path: str | Path,
    *,
    project_root: Path | None = None,
    must_exist: bool = False,
    expected_type: Literal["file", "directory"] | None = None,
) -> Path:
    """Resuelve una ruta relativa sin permitir que escape del proyecto."""
    configured_path = Path(relative_path)
    if configured_path.is_absolute():
        raise ResourcePathError(
            f"La ruta de recurso debe ser relativa al proyecto: {configured_path}"
        )

    root = get_project_root(project_root)
    resolved = (root / configured_path).resolve()
    if not resolved.is_relative_to(root):
        raise ResourcePathError(
            f"La ruta de recurso sale de la raíz del proyecto: {configured_path}"
        )

    if must_exist and not resolved.exists():
        raise ResourcePathError(f"No existe el recurso configurado: {resolved}")
    if expected_type == "file" and resolved.exists() and not resolved.is_file():
        raise ResourcePathError(f"El recurso no es un archivo: {resolved}")
    if expected_type == "directory" and resolved.exists() and not resolved.is_dir():
        raise ResourcePathError(f"El recurso no es un directorio: {resolved}")

    return resolved
