"""Punto de entrada y composición de dependencias de la aplicación."""

import logging

from gesture_matcher.utils.config_loader import ConfigError, load_config
from gesture_matcher.utils.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Valida la base del proyecto y prepara el arranque de la aplicación."""
    try:
        config = load_config()
    except ConfigError as exc:
        logging.basicConfig(level=logging.ERROR)
        LOGGER.error("No se pudo iniciar la aplicación: %s", exc)
        return 1

    configure_logging(config.logging)
    if not config.resources.hand_model.is_file():
        LOGGER.warning(
            "No se encontró el modelo MediaPipe en %s. "
            "La detección de manos se habilitará cuando se agregue ese archivo.",
            config.resources.hand_model,
        )

    LOGGER.info(
        "Base de gesture-matcher inicializada. "
        "La captura de cámara pertenece al siguiente incremento."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
