"""Tests for Ollama HTTP error friendliness."""

import pytest
import requests

from slack_question_analyzer.ollama_http import (raise_with_detail,
                                                 friendly_failure)


class FakeResponse:
    def __init__(self, ok=True, status_code=200, body=None,
                 reason='Internal Server Error'):
        self.ok = ok
        self.status_code = status_code
        self._body = body
        self.reason = reason

    def json(self):
        if self._body is None:
            raise ValueError('no json')
        return self._body

    def raise_for_status(self):
        raise requests.HTTPError(
            f"{self.status_code} Server Error: {self.reason}")


def test_ollama_error_sentence_survives():
    """The one sentence a user needs ('model requires more system memory')
    must reach the raised error, not be replaced by a bare status line."""
    resp = FakeResponse(ok=False, status_code=500, body={
        'error': 'model requires more system memory (9.3 GiB) than is '
                 'available (7.1 GiB)'})
    with pytest.raises(requests.HTTPError, match='more system memory'):
        raise_with_detail(resp)


def test_ok_response_passes_through():
    raise_with_detail(FakeResponse(ok=True))  # no exception


def test_bodyless_error_falls_back_to_status():
    with pytest.raises(requests.HTTPError, match='500'):
        raise_with_detail(FakeResponse(ok=False, status_code=500, body=None))


def test_connection_error_becomes_plain_english():
    msg = friendly_failure(requests.ConnectionError('HTTPConnectionPool(...)'),
                           'http://localhost:11434')
    assert 'ollama serve' in msg
    assert 'HTTPConnectionPool' not in msg


def test_timeout_becomes_plain_english():
    msg = friendly_failure(requests.Timeout(), 'http://localhost:11434')
    assert 'too long' in msg


def test_other_errors_pass_through_verbatim():
    assert friendly_failure(ValueError('boom'), 'x') == 'boom'


def test_embedding_error_message_is_actionable(monkeypatch):
    """End-to-end through the retry wrapper: an Ollama that is down produces
    an EmbeddingError telling the user to start Ollama, not pool internals."""
    from slack_question_analyzer.similarity_analyzer import (
        SimilarityAnalyzer, EmbeddingError)
    analyzer = SimilarityAnalyzer(provider='ollama', use_disk_cache=False)
    monkeypatch.setattr(analyzer, 'MAX_RETRIES', 1)

    def down(*a, **k):
        raise requests.ConnectionError('Max retries exceeded with url: ...')
    monkeypatch.setattr(requests, 'post', down)

    with pytest.raises(EmbeddingError, match='ollama serve'):
        analyzer._ollama_embedding('probe')
