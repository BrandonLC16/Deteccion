from unittest.mock import Mock

import pytest

import scripts.build_templates as build_templates_module
from gesture_matcher.recognition.template_builder import TemplateBuildError
from gesture_matcher.utils.config_loader import AppConfig, load_config


@pytest.fixture
def app_config() -> AppConfig:
    return load_config()


def test_main_builds_templates_successfully(
    monkeypatch: pytest.MonkeyPatch,
    app_config: AppConfig,
) -> None:
    run_template_build = Mock()
    configure_logging = Mock()
    monkeypatch.setattr(
        build_templates_module,
        "load_config",
        Mock(return_value=app_config),
    )
    monkeypatch.setattr(
        build_templates_module,
        "configure_logging",
        configure_logging,
    )
    monkeypatch.setattr(
        build_templates_module,
        "run_template_build",
        run_template_build,
    )

    assert build_templates_module.main() == 0
    configure_logging.assert_called_once_with(app_config.logging)
    run_template_build.assert_called_once_with(app_config)


def test_main_returns_error_when_no_templates_can_be_built(
    monkeypatch: pytest.MonkeyPatch,
    app_config: AppConfig,
) -> None:
    monkeypatch.setattr(
        build_templates_module,
        "load_config",
        Mock(return_value=app_config),
    )
    monkeypatch.setattr(build_templates_module, "configure_logging", Mock())
    monkeypatch.setattr(
        build_templates_module,
        "run_template_build",
        Mock(side_effect=TemplateBuildError("sin muestras válidas")),
    )

    assert build_templates_module.main() == 1
