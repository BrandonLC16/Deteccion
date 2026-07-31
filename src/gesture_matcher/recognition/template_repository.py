"""Carga y validación de plantillas persistidas para reconocimiento."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from gesture_matcher.recognition.feature_extractor import (
    CANONICAL_HAND_ORDER,
    FEATURE_VECTOR_SIZE,
    TWO_HAND_RELATIVE_FEATURE_SIZE,
)
from gesture_matcher.recognition.template_builder import TEMPLATE_FORMAT_VERSION

FEATURE_DIMENSIONS_BY_HAND_COUNT = {
    1: FEATURE_VECTOR_SIZE,
    2: FEATURE_VECTOR_SIZE * 2 + TWO_HAND_RELATIVE_FEATURE_SIZE,
}


class TemplateRepositoryError(RuntimeError):
    """Indica que los artefactos de plantillas son inexistentes o inválidos."""


@dataclass(frozen=True)
class GestureTemplate:
    """Muestras persistidas y metadatos necesarios para reconocer una seña."""

    gesture_id: str
    label: str
    hand_count: int
    feature_vectors: np.ndarray
    display_image_path: str | None = None
    similarity_threshold: float | None = None
    handedness_variants: tuple[tuple[str, ...], ...] = ()

    @property
    def sample_count(self) -> int:
        """Cantidad de muestras disponibles para la seña."""
        return int(self.feature_vectors.shape[0])

    @property
    def feature_dimension(self) -> int:
        """Dimensión esperada del vector en vivo."""
        return int(self.feature_vectors.shape[1])


class TemplateRepository:
    """Colección inmutable de plantillas validadas."""

    def __init__(
        self,
        templates: Sequence[GestureTemplate] = (),
        *,
        default_similarity_threshold: float = 0.85,
    ) -> None:
        self._default_similarity_threshold = _probability(
            default_similarity_threshold,
            "default_similarity_threshold",
        )

        validated: list[GestureTemplate] = []
        gesture_ids: set[str] = set()
        for template in templates:
            current = _validate_template(template)
            if current.gesture_id in gesture_ids:
                raise TemplateRepositoryError(
                    f"El identificador de seña está duplicado: {current.gesture_id}"
                )
            gesture_ids.add(current.gesture_id)
            validated.append(current)
        self._templates = tuple(validated)

    @classmethod
    def load(
        cls,
        templates_path: Path,
        metadata_path: Path,
        *,
        project_root: Path,
    ) -> "TemplateRepository":
        """Carga un NPZ y su JSON, verificando que ambos sean consistentes."""
        resolved_templates = templates_path.resolve()
        resolved_metadata = metadata_path.resolve()
        root = project_root.resolve()
        if not resolved_templates.is_file():
            raise TemplateRepositoryError(
                f"No se encontró el archivo de plantillas: {resolved_templates}"
            )
        if not resolved_metadata.is_file():
            raise TemplateRepositoryError(
                f"No se encontró el archivo de metadatos: {resolved_metadata}"
            )

        metadata = _load_metadata(resolved_metadata)
        _validate_format_metadata(metadata)
        default_threshold = _probability(
            _required(metadata, "default_similarity_threshold", "metadatos"),
            "default_similarity_threshold",
        )
        raw_gestures = _list(
            _required(metadata, "gestures", "metadatos"),
            "gestures",
        )

        try:
            with np.load(resolved_templates, allow_pickle=False) as archive:
                templates = _load_templates_from_archive(
                    archive,
                    raw_gestures,
                    project_root=root,
                )
        except TemplateRepositoryError:
            raise
        except (OSError, ValueError, EOFError, BadZipFile) as exc:
            raise TemplateRepositoryError(
                f"No se pudo leer el archivo de plantillas {resolved_templates}: {exc}"
            ) from exc

        return cls(
            templates,
            default_similarity_threshold=default_threshold,
        )

    @property
    def templates(self) -> tuple[GestureTemplate, ...]:
        """Devuelve todas las plantillas en orden de metadatos."""
        return self._templates

    @property
    def default_similarity_threshold(self) -> float:
        """Umbral global persistido con las plantillas."""
        return self._default_similarity_threshold

    def for_hand_count(self, hand_count: int) -> tuple[GestureTemplate, ...]:
        """Filtra plantillas por la cantidad de manos esperada."""
        return tuple(
            template
            for template in self._templates
            if template.hand_count == hand_count
        )

    def __len__(self) -> int:
        """Devuelve la cantidad de clases cargadas."""
        return len(self._templates)


def _load_metadata(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as metadata_file:
            raw_metadata = json.load(metadata_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TemplateRepositoryError(
            f"No se pudieron leer los metadatos {path}: {exc}"
        ) from exc
    return _mapping(raw_metadata, "metadatos")


def _validate_format_metadata(metadata: Mapping[str, Any]) -> None:
    version = _required(metadata, "format_version", "metadatos")
    if type(version) is not int or version != TEMPLATE_FORMAT_VERSION:
        raise TemplateRepositoryError(
            f"Versión de plantillas no compatible: {version!r}; "
            f"se esperaba {TEMPLATE_FORMAT_VERSION}."
        )

    feature_dtype = _required(metadata, "feature_dtype", "metadatos")
    if feature_dtype != "float32":
        raise TemplateRepositoryError(
            f"feature_dtype debe ser 'float32'; se recibió {feature_dtype!r}."
        )
    feature_size = _required(metadata, "feature_size_per_hand", "metadatos")
    if feature_size != FEATURE_VECTOR_SIZE:
        raise TemplateRepositoryError(
            f"feature_size_per_hand debe ser {FEATURE_VECTOR_SIZE}."
        )

    canonical_order = _required(metadata, "canonical_hand_order", "metadatos")
    if canonical_order != list(CANONICAL_HAND_ORDER):
        raise TemplateRepositoryError(
            "canonical_hand_order no coincide con el orden Left, Right soportado."
        )

    relative_metadata = _mapping(
        _required(metadata, "two_hand_relative_features", "metadatos"),
        "two_hand_relative_features",
    )
    relative_size = _required(
        relative_metadata,
        "size",
        "two_hand_relative_features",
    )
    if relative_size != TWO_HAND_RELATIVE_FEATURE_SIZE:
        raise TemplateRepositoryError(
            "La dimensión de características relativas de dos manos no es compatible."
        )


def _load_templates_from_archive(
    archive: Any,
    raw_gestures: list[Any],
    *,
    project_root: Path,
) -> tuple[GestureTemplate, ...]:
    templates: list[GestureTemplate] = []
    used_keys: set[str] = set()
    for index, raw_gesture in enumerate(raw_gestures):
        field_name = f"gestures[{index}]"
        data = _mapping(raw_gesture, field_name)
        gesture_id = _string(data, "gesture_id", field_name)
        label = _string(data, "label", field_name)
        gesture_type = _string(data, "gesture_type", field_name)
        if gesture_type != "static":
            raise TemplateRepositoryError(
                f"{field_name}.gesture_type debe ser 'static'."
            )

        template_key = _string(data, "template_key", field_name)
        if template_key in used_keys:
            raise TemplateRepositoryError(
                f"La clave NPZ está duplicada en metadatos: {template_key}"
            )
        if template_key not in archive.files:
            raise TemplateRepositoryError(
                f"No existe la matriz {template_key!r} dentro del archivo NPZ."
            )
        used_keys.add(template_key)

        hand_count = _integer(data, "hand_count", field_name)
        if hand_count not in FEATURE_DIMENSIONS_BY_HAND_COUNT:
            raise TemplateRepositoryError(f"{field_name}.hand_count debe ser 1 o 2.")
        sample_count = _integer(data, "sample_count", field_name)
        if sample_count < 1:
            raise TemplateRepositoryError(
                f"{field_name}.sample_count debe ser mayor que cero."
            )
        feature_dimension = _integer(data, "feature_dimension", field_name)
        expected_dimension = FEATURE_DIMENSIONS_BY_HAND_COUNT[hand_count]
        if feature_dimension != expected_dimension:
            raise TemplateRepositoryError(
                f"{field_name}.feature_dimension debe ser {expected_dimension} "
                f"para {hand_count} mano(s)."
            )

        threshold = _optional_probability(
            data.get("similarity_threshold"),
            f"{field_name}.similarity_threshold",
        )
        display_image_path = _optional_resource_path(
            data.get("display_image_path"),
            project_root=project_root,
            field_name=f"{field_name}.display_image_path",
        )
        handedness_variants = _handedness_variants(
            _required(data, "handedness_variants", field_name),
            hand_count=hand_count,
            field_name=f"{field_name}.handedness_variants",
        )
        _validate_samples(
            _required(data, "samples", field_name),
            sample_count=sample_count,
            hand_count=hand_count,
            project_root=project_root,
            field_name=f"{field_name}.samples",
        )

        raw_vectors = archive[template_key]
        if raw_vectors.ndim != 2 or raw_vectors.shape != (
            sample_count,
            feature_dimension,
        ):
            raise TemplateRepositoryError(
                f"La matriz {template_key!r} debe tener forma "
                f"({sample_count}, {feature_dimension}); "
                f"se recibió {raw_vectors.shape}."
            )
        if not (
            np.issubdtype(raw_vectors.dtype, np.floating)
            or np.issubdtype(raw_vectors.dtype, np.integer)
        ):
            raise TemplateRepositoryError(
                f"La matriz {template_key!r} debe contener valores numéricos."
            )

        vectors = np.ascontiguousarray(raw_vectors, dtype=np.float32)
        templates.append(
            GestureTemplate(
                gesture_id=gesture_id,
                label=label,
                hand_count=hand_count,
                feature_vectors=vectors,
                display_image_path=display_image_path,
                similarity_threshold=threshold,
                handedness_variants=handedness_variants,
            )
        )

    unused_keys = set(archive.files) - used_keys
    if unused_keys:
        raise TemplateRepositoryError(
            "El archivo NPZ contiene matrices sin metadatos: "
            + ", ".join(sorted(unused_keys))
        )
    return tuple(templates)


def _validate_template(template: GestureTemplate) -> GestureTemplate:
    if not isinstance(template, GestureTemplate):
        raise TemplateRepositoryError(
            "Cada elemento del repositorio debe ser GestureTemplate."
        )
    gesture_id = template.gesture_id.strip()
    label = template.label.strip()
    if not gesture_id:
        raise TemplateRepositoryError("gesture_id no puede estar vacío.")
    if not label:
        raise TemplateRepositoryError(
            f"La etiqueta de {gesture_id} no puede estar vacía."
        )
    if template.hand_count not in FEATURE_DIMENSIONS_BY_HAND_COUNT:
        raise TemplateRepositoryError(
            f"La plantilla {gesture_id} debe requerir una o dos manos."
        )

    expected_dimension = FEATURE_DIMENSIONS_BY_HAND_COUNT[template.hand_count]
    values = np.asarray(template.feature_vectors)
    if values.ndim != 2 or values.shape[0] < 1:
        raise TemplateRepositoryError(
            f"La plantilla {gesture_id} debe contener una matriz 2D no vacía."
        )
    if values.shape[1] != expected_dimension:
        raise TemplateRepositoryError(
            f"La plantilla {gesture_id} debe tener dimensión {expected_dimension}; "
            f"se recibió {values.shape[1]}."
        )
    if not (
        np.issubdtype(values.dtype, np.floating)
        or np.issubdtype(values.dtype, np.integer)
    ):
        raise TemplateRepositoryError(
            f"La plantilla {gesture_id} debe contener valores numéricos."
        )

    frozen_values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(frozen_values).all():
        raise TemplateRepositoryError(
            f"La plantilla {gesture_id} contiene NaN o infinitos."
        )
    norms = np.linalg.norm(frozen_values, axis=1)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise TemplateRepositoryError(
            f"La plantilla {gesture_id} contiene una muestra con norma cero."
        )
    frozen_values.setflags(write=False)

    threshold = _optional_probability(
        template.similarity_threshold,
        f"similarity_threshold de {gesture_id}",
    )
    return GestureTemplate(
        gesture_id=gesture_id,
        label=label,
        hand_count=template.hand_count,
        feature_vectors=frozen_values,
        display_image_path=template.display_image_path,
        similarity_threshold=threshold,
        handedness_variants=tuple(template.handedness_variants),
    )


def _validate_samples(
    value: Any,
    *,
    sample_count: int,
    hand_count: int,
    project_root: Path,
    field_name: str,
) -> None:
    samples = _list(value, field_name)
    if len(samples) != sample_count:
        raise TemplateRepositoryError(
            f"{field_name} debe contener {sample_count} elementos."
        )

    rows: set[int] = set()
    for index, raw_sample in enumerate(samples):
        sample_name = f"{field_name}[{index}]"
        sample = _mapping(raw_sample, sample_name)
        row = _integer(sample, "template_row", sample_name)
        if not 0 <= row < sample_count or row in rows:
            raise TemplateRepositoryError(
                f"{sample_name}.template_row es inválido o está duplicado."
            )
        rows.add(row)
        source_path = _string(sample, "source_image_path", sample_name)
        _safe_relative_path(
            source_path,
            project_root=project_root,
            field_name=f"{sample_name}.source_image_path",
        )
        _handedness(
            _required(sample, "handedness", sample_name),
            hand_count=hand_count,
            field_name=f"{sample_name}.handedness",
        )


def _handedness_variants(
    value: Any,
    *,
    hand_count: int,
    field_name: str,
) -> tuple[tuple[str, ...], ...]:
    raw_variants = _list(value, field_name)
    if not raw_variants:
        raise TemplateRepositoryError(f"{field_name} no puede estar vacío.")
    variants = tuple(
        _handedness(
            variant,
            hand_count=hand_count,
            field_name=f"{field_name}[{index}]",
        )
        for index, variant in enumerate(raw_variants)
    )
    if len(set(variants)) != len(variants):
        raise TemplateRepositoryError(
            f"{field_name} contiene lateralidades duplicadas."
        )
    return variants


def _handedness(
    value: Any,
    *,
    hand_count: int,
    field_name: str,
) -> tuple[str, ...]:
    labels = _list(value, field_name)
    if len(labels) != hand_count or any(
        label not in CANONICAL_HAND_ORDER for label in labels
    ):
        raise TemplateRepositoryError(
            f"{field_name} debe contener {hand_count} lateralidad(es) válidas."
        )
    normalized = tuple(labels)
    if hand_count == 2 and normalized != CANONICAL_HAND_ORDER:
        raise TemplateRepositoryError(
            f"{field_name} debe respetar el orden Left, Right."
        )
    return normalized


def _optional_resource_path(
    value: Any,
    *,
    project_root: Path,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    path = _safe_relative_path(
        value,
        project_root=project_root,
        field_name=field_name,
    )
    resolved = (project_root / path).resolve()
    if not resolved.is_file():
        raise TemplateRepositoryError(
            f"No existe la imagen asociada configurada en {field_name}: {resolved}"
        )
    return path.as_posix()


def _safe_relative_path(
    value: Any,
    *,
    project_root: Path,
    field_name: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TemplateRepositoryError(f"{field_name} debe ser una ruta no vacía.")
    path = Path(value)
    if path.is_absolute():
        raise TemplateRepositoryError(f"{field_name} debe ser relativa al proyecto.")
    resolved = (project_root / path).resolve()
    if not resolved.is_relative_to(project_root):
        raise TemplateRepositoryError(f"{field_name} sale de la raíz del proyecto.")
    return path


def _optional_probability(value: Any, field_name: str) -> float | None:
    return None if value is None else _probability(value, field_name)


def _probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemplateRepositoryError(f"{field_name} debe ser un número entre 0 y 1.")
    converted = float(value)
    if not np.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise TemplateRepositoryError(f"{field_name} debe estar entre 0 y 1.")
    return converted


def _required(
    data: Mapping[str, Any],
    key: str,
    field_name: str,
) -> Any:
    if key not in data:
        raise TemplateRepositoryError(f"Falta el campo obligatorio {field_name}.{key}.")
    return data[key]


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TemplateRepositoryError(f"{field_name} debe ser un objeto.")
    return value


def _list(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TemplateRepositoryError(f"{field_name} debe ser una lista.")
    return value


def _string(
    data: Mapping[str, Any],
    key: str,
    field_name: str,
) -> str:
    value = _required(data, key, field_name)
    if not isinstance(value, str) or not value.strip():
        raise TemplateRepositoryError(f"{field_name}.{key} debe ser texto no vacío.")
    return value.strip()


def _integer(
    data: Mapping[str, Any],
    key: str,
    field_name: str,
) -> int:
    value = _required(data, key, field_name)
    if type(value) is not int:
        raise TemplateRepositoryError(f"{field_name}.{key} debe ser un entero.")
    return value
