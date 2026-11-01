"""Tests for the central project configuration."""

import importlib
from pathlib import Path

import config


def test_config_import_does_not_create_directories(monkeypatch):
    """Verify that importing config never touches the filesystem.

    Description:
        Reload config with Path.mkdir patched to fail.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Used to replace Path.mkdir.
    Returns:
        None: Passes when the reload succeeds without calling mkdir.
    Notes:
        Guards against reintroducing import-time creation of data folders.
    """
    def fail_mkdir(self, *args, **kwargs):
        raise AssertionError(f"config import tried to create {self}")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    importlib.reload(config)


def test_environment_variables_override_default_paths(monkeypatch, tmp_path):
    """Verify CSI300_DATA_ROOT and CSI300_QLIB_DIR override the default locations.

    Description:
        Reload config with both variables set and check the derived folders.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Used to set environment variables.
        tmp_path (Path): Temporary folder used as a stand-in data root.
    Returns:
        None: Passes when every derived path follows the overrides.
    Notes:
        Config is reloaded without the variables afterwards so other tests see defaults.
    """
    monkeypatch.setenv("CSI300_DATA_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("CSI300_QLIB_DIR", str(tmp_path / "qlib"))
    try:
        importlib.reload(config)
        assert config.DATA_ROOT == tmp_path / "root"
        assert config.DATA_DIR == tmp_path / "root" / "data"
        assert config.DATA_OUTPUT_DIR == tmp_path / "root" / "data" / "processed"
        assert config.REPORT_OUTPUT_DIR == tmp_path / "root" / "reports"
        assert config.QLIB_DATA_DIR == tmp_path / "qlib"
    finally:
        monkeypatch.delenv("CSI300_DATA_ROOT")
        monkeypatch.delenv("CSI300_QLIB_DIR")
        importlib.reload(config)
    assert config.DATA_ROOT == Path(config.DEFAULT_DATA_ROOT)
