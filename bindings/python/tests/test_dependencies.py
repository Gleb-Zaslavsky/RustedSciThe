"""Hermetic tests for the Python dependency preflight helpers."""

from __future__ import annotations

import subprocess

import pytest

import rustedscithe.dependencies as dependencies
from rustedscithe import check_dependencies


def test_dependency_preflight_reports_path_tools_and_loadable_superlu(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake_tools = {name: str(tmp_path / f"{name}.exe") for name in dependencies._TOOLS}
    monkeypatch.setattr(
        dependencies.shutil,
        "which",
        lambda executable: fake_tools.get(executable),
    )

    def run(command, **kwargs):
        executable = command[0]
        name = next(name for name, path in fake_tools.items() if path == executable)
        completed = subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{name} version 1.2\n",
            stderr="",
        )
        return completed

    monkeypatch.setattr(dependencies.subprocess, "run", run)
    bridge = tmp_path / "superlu_bridge.dll"
    bridge.write_bytes(b"fake bridge for mocked loader")
    monkeypatch.setenv("SUPERLU_LIBRARY", str(bridge))
    monkeypatch.setattr(dependencies._native, "superlu_feature_enabled", lambda: True)
    loaded: list[str] = []
    monkeypatch.setattr(dependencies.ctypes, "CDLL", lambda path: loaded.append(path))

    report = check_dependencies()

    assert set(report["tools"]) == {"gnuplot", "tcc", "gcc", "zig", "pandoc"}
    assert all(item["status"] == "available" for item in report["tools"].values())
    assert all(item["available"] for item in report["tools"].values())
    assert report["tools"]["gcc"]["version"] == "gcc version 1.2"
    assert report["superlu"] == {
        "feature_enabled": True,
        "library_path": str(bridge),
        "status": "loadable",
        "loadable": True,
        "detail": None,
    }
    assert loaded == [str(bridge.resolve())]


def test_dependency_preflight_reports_missing_tools_and_unconfigured_superlu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dependencies.shutil, "which", lambda _name: None)
    monkeypatch.delenv("SUPERLU_LIBRARY", raising=False)
    monkeypatch.setattr(dependencies._native, "superlu_feature_enabled", lambda: False)

    report = check_dependencies()

    assert all(item["status"] == "missing" for item in report["tools"].values())
    assert all(not item["available"] for item in report["tools"].values())
    assert report["superlu"]["feature_enabled"] is False
    assert report["superlu"]["status"] == "not_configured"
    assert report["superlu"]["loadable"] is None


def test_dependency_preflight_distinguishes_probe_and_library_load_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    tool_path = str(tmp_path / "gnuplot")
    monkeypatch.setattr(
        dependencies.shutil,
        "which",
        lambda name: tool_path if name == "gnuplot" else None,
    )

    def failed_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="broken tool")

    monkeypatch.setattr(dependencies.subprocess, "run", failed_run)
    bridge = tmp_path / "unloadable.so"
    bridge.write_bytes(b"not a shared library")
    monkeypatch.setenv("SUPERLU_LIBRARY", str(bridge))
    monkeypatch.setattr(dependencies._native, "superlu_feature_enabled", lambda: False)

    def fail_load(_path):
        raise OSError("invalid shared library")

    monkeypatch.setattr(dependencies.ctypes, "CDLL", fail_load)
    report = check_dependencies()

    assert report["tools"]["gnuplot"]["status"] == "probe_error"
    assert report["tools"]["gnuplot"]["available"] is True
    assert report["tools"]["gcc"]["status"] == "missing"
    assert report["superlu"]["status"] == "load_error"
    assert report["superlu"]["loadable"] is False
    assert report["superlu"]["feature_enabled"] is False


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_dependency_preflight_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        check_dependencies(timeout=timeout)
