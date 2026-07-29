from gesture_matcher import __version__
from gesture_matcher.utils.config_loader import AppConfig, load_config


def test_package_imports_and_default_configuration_loads() -> None:
    config = load_config()

    assert __version__ == "0.1.0"
    assert isinstance(config, AppConfig)
