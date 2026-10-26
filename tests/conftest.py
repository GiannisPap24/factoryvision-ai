"""Offline fixtures; every temporary file remains inside the workspace."""

import json
import socket
from pathlib import Path

import pytest

from factoryvision.data.manifest import load_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic-manifest.json"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in the default test suite.")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def manifest():
    return load_manifest(FIXTURE)
