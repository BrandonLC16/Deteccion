from unittest.mock import Mock

import numpy as np
import pytest

from gesture_matcher.ui.image_overlay import (
    ImageCache,
    ImageOverlayError,
    resize_to_fit,
)


def test_resizes_and_centers_without_deforming_aspect_ratio() -> None:
    source = np.full((50, 100, 3), 255, dtype=np.uint8)

    result = resize_to_fit(source, target_width=100, target_height=100)

    assert result.shape == (100, 100, 3)
    assert np.all(result[:25] == 0)
    assert np.all(result[25:75] == 255)
    assert np.all(result[75:] == 0)
    assert not result.flags.writeable


def test_cache_loads_each_image_only_once(tmp_path) -> None:
    image_path = tmp_path / "assets" / "display_images" / "victory.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"placeholder")
    loader = Mock(return_value=np.full((10, 20, 3), 127, dtype=np.uint8))
    cache = ImageCache(
        project_root=tmp_path,
        target_width=40,
        target_height=40,
        image_loader=loader,
    )

    first = cache.get("assets/display_images/victory.jpg")
    second = cache.get("assets/display_images/victory.jpg")

    assert first is second
    assert first is not None
    assert first.shape == (40, 40, 3)
    loader.assert_called_once_with(str(image_path.resolve()))


def test_cache_remembers_failed_loads(tmp_path) -> None:
    image_path = tmp_path / "assets" / "display_images" / "broken.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not-an-image")
    loader = Mock(return_value=None)
    cache = ImageCache(
        project_root=tmp_path,
        target_width=40,
        target_height=40,
        image_loader=loader,
    )

    assert cache.get("assets/display_images/broken.jpg") is None
    assert cache.get("assets/display_images/broken.jpg") is None
    loader.assert_called_once()


def test_cache_rejects_paths_outside_project(tmp_path) -> None:
    loader = Mock()
    cache = ImageCache(
        project_root=tmp_path,
        target_width=40,
        target_height=40,
        image_loader=loader,
    )

    assert cache.get("../outside.jpg") is None
    assert cache.get("../outside.jpg") is None
    loader.assert_not_called()


def test_clear_forces_a_new_load(tmp_path) -> None:
    image_path = tmp_path / "display.jpg"
    image_path.write_bytes(b"placeholder")
    loader = Mock(return_value=np.ones((2, 2, 3), dtype=np.uint8))
    cache = ImageCache(
        project_root=tmp_path,
        target_width=2,
        target_height=2,
        image_loader=loader,
    )

    cache.get("display.jpg")
    cache.clear()
    cache.get("display.jpg")

    assert loader.call_count == 2


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 10), (10, 0), (True, 10), (10, False)],
)
def test_rejects_invalid_target_dimensions(width: object, height: object) -> None:
    with pytest.raises(ImageOverlayError):
        ImageCache(
            project_root=Mock(),
            target_width=width,  # type: ignore[arg-type]
            target_height=height,  # type: ignore[arg-type]
        )
