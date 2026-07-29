"""Configuración central del registro de la aplicación."""

import logging

from gesture_matcher.utils.config_loader import LoggingConfig


def configure_logging(config: LoggingConfig) -> None:
    """Configura el logging una sola vez durante el arranque."""
    logging.basicConfig(
        level=getattr(logging, config.level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
