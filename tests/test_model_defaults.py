"""Hardware-aware chat model defaults and the runtime downgrade."""

import pytest

from slack_question_analyzer import model_defaults
from slack_question_analyzer.model_defaults import (
    default_generation_model,
    PREFERRED_GENERATION_MODEL,
    FALLBACK_GENERATION_MODEL,
)
from slack_question_analyzer.group_labeler import GroupLabeler


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


@pytest.fixture
def unpinned(monkeypatch):
    monkeypatch.delenv('OLLAMA_GENERATION_MODEL', raising=False)


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv('OLLAMA_GENERATION_MODEL', 'mistral')
    assert default_generation_model() == 'mistral'


def test_big_machine_gets_preferred_model(unpinned, monkeypatch):
    monkeypatch.setattr(model_defaults, 'total_ram_gb', lambda: 16.0)
    assert default_generation_model() == PREFERRED_GENERATION_MODEL


def test_small_machine_gets_fallback_model(unpinned, monkeypatch):
    monkeypatch.setattr(model_defaults, 'total_ram_gb', lambda: 8.0)
    assert default_generation_model() == FALLBACK_GENERATION_MODEL


def test_unknown_ram_gets_fallback_model(unpinned, monkeypatch):
    monkeypatch.setattr(model_defaults, 'total_ram_gb', lambda: None)
    assert default_generation_model() == FALLBACK_GENERATION_MODEL


def test_total_ram_gb_returns_plausible_value():
    ram = model_defaults.total_ram_gb()
    assert ram is None or 0.5 < ram < 4096


def _labeler_with_tags(monkeypatch, tag_names):
    monkeypatch.setattr(
        'slack_question_analyzer.group_labeler.requests.get',
        lambda *a, **k: FakeResponse({'models': [{'name': n} for n in tag_names]}))
    return GroupLabeler(provider='ollama')


def test_labeler_downgrades_when_preferred_not_downloaded(unpinned, monkeypatch):
    monkeypatch.setattr(model_defaults, 'total_ram_gb', lambda: 32.0)
    labeler = _labeler_with_tags(monkeypatch, ['llama3.2:latest'])
    assert labeler.model == PREFERRED_GENERATION_MODEL
    assert labeler.available() is True
    assert labeler.model == FALLBACK_GENERATION_MODEL


def test_labeler_keeps_preferred_when_downloaded(unpinned, monkeypatch):
    monkeypatch.setattr(model_defaults, 'total_ram_gb', lambda: 32.0)
    labeler = _labeler_with_tags(monkeypatch, ['llama3.1:8b', 'llama3.2:latest'])
    assert labeler.available() is True
    assert labeler.model == PREFERRED_GENERATION_MODEL


def test_pinned_model_never_downgrades(monkeypatch):
    monkeypatch.setenv('OLLAMA_GENERATION_MODEL', 'mistral')
    labeler = _labeler_with_tags(monkeypatch, ['llama3.2:latest'])
    assert labeler.available() is False
    assert labeler.model == 'mistral'
