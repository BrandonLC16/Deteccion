"""Construcción y persistencia offline de plantillas de señas estáticas."""

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

import cv2
import numpy as np

from gesture_matcher.recognition.feature_extractor import (
    CANONICAL_HAND_ORDER,
    FEATURE_VECTOR_SIZE,
    TWO_HAND_RELATIVE_FEATURE_SIZE,
    FeatureExtractionError,
    FeatureExtractor,
    HandFeatureVector,
)
from gesture_matcher.vision.hand_detector import HandDetectionError, HandObservation

TEMPLATE_FORMAT_VERSION = 1
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
GESTURE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class TemplateBuildError(RuntimeError):
    """Indica que las plantillas no pudieron construirse de forma segura."""


class TemplatePersistenceError(TemplateBuildError):
    """Indica que falló la escritura de plantillas o metadatos."""


class ReferenceImageDetector(Protocol):
    """Detector mínimo requerido por el constructor de plantillas."""

    def detect(self, image: np.ndarray) -> tuple[HandObservation, ...]:
        """Devuelve las manos encontradas en una imagen BGR."""


ImageLoader = Callable[[Path], np.ndarray | None]


@dataclass(frozen=True)
class TemplateSample:
    """Metadatos de una muestra aceptada dentro de una clase."""

    source_path: Path
    handedness: tuple[str, ...]


@dataclass(frozen=True)
class GestureTemplates:
    """Matriz de muestras y esquema compartido de una seña."""

    gesture_id: str
    hand_count: int
    feature_vectors: np.ndarray
    samples: tuple[TemplateSample, ...]


@dataclass(frozen=True)
class AcceptedImage:
    """Imagen que produjo una muestra válida."""

    path: Path
    gesture_id: str
    hand_count: int
    handedness: tuple[str, ...]


@dataclass(frozen=True)
class RejectedReference:
    """Archivo o carpeta rechazado con un motivo comprensible."""

    path: Path
    reason: str


@dataclass(frozen=True)
class TemplateBuildResult:
    """Plantillas construidas y reporte completo de validación."""

    gestures: tuple[GestureTemplates, ...]
    accepted: tuple[AcceptedImage, ...]
    rejected: tuple[RejectedReference, ...]

    @property
    def accepted_count(self) -> int:
        """Cantidad total de imágenes aceptadas."""
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        """Cantidad total de referencias rechazadas."""
        return len(self.rejected)


@dataclass(frozen=True)
class _Candidate:
    path: Path
    features: HandFeatureVector


def load_bgr_image(path: Path) -> np.ndarray | None:
    """Lee una imagen desde disco conservando tres canales BGR."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


class TemplateBuilder:
    """Recorre clases, valida imágenes y extrae muestras geométricas."""

    def __init__(
        self,
        detector: ReferenceImageDetector,
        feature_extractor: FeatureExtractor,
        *,
        image_loader: ImageLoader = load_bgr_image,
    ) -> None:
        self._detector = detector
        self._feature_extractor = feature_extractor
        self._image_loader = image_loader

    def build(self, reference_root: Path) -> TemplateBuildResult:
        """Construye plantillas desde una carpeta hija por cada seña."""
        root = reference_root.resolve()
        if not root.is_dir():
            raise TemplateBuildError(
                f"No existe el directorio de imágenes de referencia: {root}"
            )

        gesture_directories = sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )
        if not gesture_directories:
            raise TemplateBuildError(
                f"No se encontraron carpetas de señas dentro de {root}."
            )

        gestures: list[GestureTemplates] = []
        accepted: list[AcceptedImage] = []
        rejected: list[RejectedReference] = []
        for gesture_directory in gesture_directories:
            built, gesture_accepted, gesture_rejected = self._build_gesture(
                gesture_directory
            )
            rejected.extend(gesture_rejected)
            if built is not None:
                gestures.append(built)
                accepted.extend(gesture_accepted)

        return TemplateBuildResult(
            gestures=tuple(gestures),
            accepted=tuple(accepted),
            rejected=tuple(rejected),
        )

    def _build_gesture(
        self,
        gesture_directory: Path,
    ) -> tuple[
        GestureTemplates | None,
        list[AcceptedImage],
        list[RejectedReference],
    ]:
        gesture_id = gesture_directory.name
        if GESTURE_ID_PATTERN.fullmatch(gesture_id) is None:
            rejected = [
                RejectedReference(
                    path=path,
                    reason=(
                        f"La clase {gesture_id!r} debe usar un identificador "
                        "snake_case."
                    ),
                )
                for path in self._direct_children_or_directory(gesture_directory)
            ]
            return None, [], rejected

        candidates: list[_Candidate] = []
        rejected: list[RejectedReference] = []
        entries = sorted(
            gesture_directory.iterdir(),
            key=lambda path: path.name.casefold(),
        )
        for path in entries:
            if not path.is_file():
                rejected.append(
                    RejectedReference(
                        path=path,
                        reason=(
                            "Solo se admiten archivos directamente dentro de la clase."
                        ),
                    )
                )
                continue
            if path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
                rejected.append(
                    RejectedReference(
                        path=path,
                        reason=(
                            "Extensión no permitida: "
                            f"{path.suffix or '(sin extensión)'}."
                        ),
                    )
                )
                continue

            candidate, rejection = self._process_image(path)
            if candidate is not None:
                candidates.append(candidate)
            elif rejection is not None:
                rejected.append(rejection)

        if not entries:
            rejected.append(
                RejectedReference(
                    path=gesture_directory,
                    reason="La carpeta de la seña está vacía.",
                )
            )
        if not candidates:
            return None, [], rejected

        selected_count = self._select_hand_count(candidates)
        if selected_count is None:
            rejected.extend(
                RejectedReference(
                    path=candidate.path,
                    reason=(
                        "La clase tiene un empate entre muestras de una y dos manos; "
                        "no se puede definir su cantidad esperada."
                    ),
                )
                for candidate in candidates
            )
            return None, [], rejected

        selected: list[_Candidate] = []
        for candidate in candidates:
            if candidate.features.hand_count != selected_count:
                rejected.append(
                    RejectedReference(
                        path=candidate.path,
                        reason=(
                            f"La clase {gesture_id} usa {selected_count} mano(s), "
                            f"pero la imagen contiene {candidate.features.hand_count}."
                        ),
                    )
                )
            else:
                selected.append(candidate)

        matrix = np.ascontiguousarray(
            np.stack([candidate.features.vector for candidate in selected]),
            dtype=np.float32,
        )
        if not np.isfinite(matrix).all():
            raise TemplateBuildError(
                f"La matriz generada para {gesture_id} contiene NaN o infinitos."
            )
        matrix.setflags(write=False)

        samples = tuple(
            TemplateSample(
                source_path=candidate.path,
                handedness=candidate.features.handedness,
            )
            for candidate in selected
        )
        accepted = [
            AcceptedImage(
                path=candidate.path,
                gesture_id=gesture_id,
                hand_count=selected_count,
                handedness=candidate.features.handedness,
            )
            for candidate in selected
        ]
        return (
            GestureTemplates(
                gesture_id=gesture_id,
                hand_count=selected_count,
                feature_vectors=matrix,
                samples=samples,
            ),
            accepted,
            rejected,
        )

    def _process_image(
        self,
        path: Path,
    ) -> tuple[_Candidate | None, RejectedReference | None]:
        try:
            image = self._image_loader(path)
        except Exception as exc:
            return None, RejectedReference(
                path=path,
                reason=f"No se pudo leer la imagen: {exc}",
            )
        if image is None:
            return None, RejectedReference(
                path=path,
                reason="OpenCV no pudo decodificar la imagen.",
            )

        try:
            hands = self._detector.detect(image)
            if not hands:
                return None, RejectedReference(
                    path=path,
                    reason="No se detectó ninguna mano.",
                )
            features = self._feature_extractor.extract_hands(hands)
        except (HandDetectionError, FeatureExtractionError) as exc:
            return None, RejectedReference(path=path, reason=str(exc))
        except Exception as exc:
            return None, RejectedReference(
                path=path,
                reason=f"Falló el procesamiento de la imagen: {exc}",
            )

        return _Candidate(path=path, features=features), None

    @staticmethod
    def _select_hand_count(candidates: Sequence[_Candidate]) -> int | None:
        counts = Counter(candidate.features.hand_count for candidate in candidates)
        highest_frequency = max(counts.values())
        winners = [
            hand_count
            for hand_count, frequency in counts.items()
            if frequency == highest_frequency
        ]
        return winners[0] if len(winners) == 1 else None

    @staticmethod
    def _direct_children_or_directory(directory: Path) -> tuple[Path, ...]:
        children = tuple(
            sorted(directory.iterdir(), key=lambda path: path.name.casefold())
        )
        return children or (directory,)


def save_template_artifacts(
    result: TemplateBuildResult,
    *,
    templates_path: Path,
    metadata_path: Path,
    project_root: Path,
    display_images_root: Path,
    default_similarity_threshold: float,
    gesture_thresholds: Mapping[str, float],
    mirror_left_hand: bool,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Guarda matrices NPZ y metadatos JSON sin escribir parcialmente."""
    if not result.gestures:
        raise TemplatePersistenceError(
            "No hay muestras aceptadas; no se escribirán plantillas vacías."
        )

    resolved_templates = templates_path.resolve()
    resolved_metadata = metadata_path.resolve()
    if resolved_templates == resolved_metadata:
        raise TemplatePersistenceError(
            "Las plantillas y los metadatos deben usar archivos distintos."
        )

    metadata = _build_metadata(
        result,
        project_root=project_root.resolve(),
        display_images_root=display_images_root.resolve(),
        default_similarity_threshold=default_similarity_threshold,
        gesture_thresholds=gesture_thresholds,
        mirror_left_hand=mirror_left_hand,
        generated_at=generated_at or datetime.now(UTC),
    )
    arrays = {
        gesture.gesture_id: gesture.feature_vectors for gesture in result.gestures
    }

    resolved_templates.parent.mkdir(parents=True, exist_ok=True)
    resolved_metadata.parent.mkdir(parents=True, exist_ok=True)
    template_temporary = _write_npz_temporary(resolved_templates, arrays)
    metadata_temporary: Path | None = None
    try:
        metadata_temporary = _write_json_temporary(resolved_metadata, metadata)
        template_temporary.replace(resolved_templates)
        metadata_temporary.replace(resolved_metadata)
    except (OSError, TypeError, ValueError) as exc:
        raise TemplatePersistenceError(
            f"No se pudieron guardar las plantillas y metadatos: {exc}"
        ) from exc
    finally:
        template_temporary.unlink(missing_ok=True)
        if metadata_temporary is not None:
            metadata_temporary.unlink(missing_ok=True)

    return metadata


def format_build_report(
    result: TemplateBuildResult,
    *,
    project_root: Path,
) -> str:
    """Genera un reporte legible de clases e imágenes aceptadas/rechazadas."""
    lines = [
        "REPORTE DE CONSTRUCCIÓN DE PLANTILLAS",
        f"Clases generadas: {len(result.gestures)}",
        f"Imágenes aceptadas: {result.accepted_count}",
        f"Referencias rechazadas: {result.rejected_count}",
    ]

    if result.gestures:
        lines.append("")
        lines.append("Resumen por clase:")
        for gesture in result.gestures:
            lines.append(
                f"  - {gesture.gesture_id}: {len(gesture.samples)} muestra(s), "
                f"{gesture.hand_count} mano(s), "
                f"vector de {gesture.feature_vectors.shape[1]} valores"
            )

    if result.accepted:
        lines.append("")
        lines.append("Aceptadas:")
        for accepted in result.accepted:
            labels = ", ".join(accepted.handedness)
            lines.append(
                f"  [OK] {_display_path(accepted.path, project_root)} "
                f"({accepted.hand_count} mano(s): {labels})"
            )

    if result.rejected:
        lines.append("")
        lines.append("Rechazadas:")
        for rejected in result.rejected:
            lines.append(
                f"  [RECHAZADA] {_display_path(rejected.path, project_root)}: "
                f"{rejected.reason}"
            )

    return "\n".join(lines)


def _build_metadata(
    result: TemplateBuildResult,
    *,
    project_root: Path,
    display_images_root: Path,
    default_similarity_threshold: float,
    gesture_thresholds: Mapping[str, float],
    mirror_left_hand: bool,
    generated_at: datetime,
) -> dict[str, object]:
    gestures: list[dict[str, object]] = []
    for gesture in result.gestures:
        display_image = _find_display_image(
            display_images_root,
            gesture.gesture_id,
        )
        handedness_variants = sorted({sample.handedness for sample in gesture.samples})
        samples = [
            {
                "template_row": index,
                "source_image_path": _relative_path(
                    sample.source_path,
                    project_root,
                ),
                "handedness": list(sample.handedness),
            }
            for index, sample in enumerate(gesture.samples)
        ]
        gestures.append(
            {
                "gesture_id": gesture.gesture_id,
                "label": gesture.gesture_id,
                "gesture_type": "static",
                "template_key": gesture.gesture_id,
                "hand_count": gesture.hand_count,
                "handedness_variants": [
                    list(variant) for variant in handedness_variants
                ],
                "sample_count": len(gesture.samples),
                "feature_dimension": int(gesture.feature_vectors.shape[1]),
                "similarity_threshold": gesture_thresholds.get(gesture.gesture_id),
                "display_image_path": (
                    _relative_path(display_image, project_root)
                    if display_image is not None
                    else None
                ),
                "samples": samples,
            }
        )

    return {
        "format_version": TEMPLATE_FORMAT_VERSION,
        "generated_at_utc": generated_at.astimezone(UTC).isoformat(),
        "coordinate_source": "image_landmarks",
        "feature_dtype": "float32",
        "feature_size_per_hand": FEATURE_VECTOR_SIZE,
        "canonical_hand_order": list(CANONICAL_HAND_ORDER),
        "two_hand_relative_features": {
            "size": TWO_HAND_RELATIVE_FEATURE_SIZE,
            "definition": "right_wrist_minus_left_wrist_over_mean_hand_scale",
        },
        "mirror_left_hand": mirror_left_hand,
        "default_similarity_threshold": default_similarity_threshold,
        "gestures": gestures,
    }


def _find_display_image(display_root: Path, gesture_id: str) -> Path | None:
    if not display_root.is_dir():
        return None
    matches = sorted(
        (
            path
            for path in display_root.iterdir()
            if path.is_file()
            and path.stem.casefold() == gesture_id.casefold()
            and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    if len(matches) > 1:
        raise TemplatePersistenceError(
            f"Existe más de una imagen de presentación para {gesture_id}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0] if matches else None


def _write_npz_temporary(
    target: Path,
    arrays: Mapping[str, np.ndarray],
) -> Path:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.stem}-",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
            return temporary_path
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise TemplatePersistenceError(
            f"No se pudo preparar el archivo de plantillas {target}: {exc}"
        ) from exc


def _write_json_temporary(
    target: Path,
    metadata: Mapping[str, object],
) -> Path:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.stem}-",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(metadata, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            return temporary_path
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise TemplatePersistenceError(
            f"No se pudo preparar el archivo de metadatos {target}: {exc}"
        ) from exc


def _relative_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(project_root):
        raise TemplatePersistenceError(
            f"La ruta de metadatos sale del proyecto: {resolved}"
        )
    return resolved.relative_to(project_root).as_posix()


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    return (
        resolved.relative_to(root).as_posix()
        if resolved.is_relative_to(root)
        else str(resolved)
    )
