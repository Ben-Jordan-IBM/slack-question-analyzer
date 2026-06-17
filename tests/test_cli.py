"""Tests for the command-line interface."""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from slack_question_analyzer.cli import cli

SAMPLE_CONTENT = (
    "2024-01-05\nHow do I reset my password?\n"
    "-----------------------------------------------------------\n"
    "2024-01-08\nHow can I reset my password?\n"
    "-----------------------------------------------------------\n"
    "2024-01-09\nWhat is the deploy schedule for production releases?\n"
)


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setenv('GROUP_LABELS', 'off')
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    vectors = {
        'how do i reset my password?': [1.0, 0.0, 0.0],
        'how can i reset my password?': [0.99, 0.05, 0.0],
        'what is the deploy schedule for production releases?': [0.0, 1.0, 0.0],
    }

    def fake_batch(self, texts, progress_callback=None):
        return np.array([vectors[t] for t in texts])

    monkeypatch.setattr('slack_question_analyzer.similarity_analyzer.SimilarityAnalyzer.get_embeddings_batch',
                        fake_batch)


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path / 'input.txt'
    path.write_text(SAMPLE_CONTENT, encoding='utf-8')
    return path


def test_analyze_writes_json_and_summary(fake_engine, input_file, tmp_path):
    output = tmp_path / 'results.json'
    result = CliRunner().invoke(cli, ['analyze', str(input_file), '-o', str(output),
                                      '--no-cache', '--no-labels'])

    assert result.exit_code == 0, result.output
    assert 'QUESTION ANALYSIS SUMMARY' in result.output
    assert 'Total Questions Analyzed: 3' in result.output

    saved = json.loads(output.read_text(encoding='utf-8'))
    assert saved['total_questions'] == 3
    assert saved['total_groups'] == 1


def test_analyze_markdown_output(fake_engine, input_file, tmp_path):
    output = tmp_path / 'report.md'
    result = CliRunner().invoke(cli, ['analyze', str(input_file), '-o', str(output),
                                      '--no-cache', '--no-labels', '--no-summary'])

    assert result.exit_code == 0, result.output
    assert 'QUESTION ANALYSIS SUMMARY' not in result.output  # --no-summary
    report = output.read_text(encoding='utf-8')
    assert '# Question Analysis Report' in report


def test_analyze_zip_and_multiple_files(fake_engine, input_file, tmp_path):
    """The CLI accepts zip archives and multiple inputs, like the web UI."""
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('export/day1.txt', SAMPLE_CONTENT)
    zip_path = tmp_path / 'export.zip'
    zip_path.write_bytes(buffer.getvalue())

    extra = tmp_path / 'extra.json'
    extra.write_text(json.dumps([{'text': 'How can I reset my password?'}]),
                     encoding='utf-8')

    output = tmp_path / 'combined.json'
    result = CliRunner().invoke(cli, ['analyze', str(zip_path), str(extra),
                                      '-o', str(output), '--no-cache', '--no-labels',
                                      '--no-summary'])

    assert result.exit_code == 0, result.output
    saved = json.loads(output.read_text(encoding='utf-8'))
    assert saved['total_questions'] == 4  # 3 from the zip + 1 from the json


def test_analyze_rejects_bad_threshold(input_file):
    result = CliRunner().invoke(cli, ['analyze', str(input_file), '--threshold', '2'])
    assert result.exit_code != 0
    assert '2' in result.output  # click range validation message


def test_analyze_missing_file():
    result = CliRunner().invoke(cli, ['analyze', 'does-not-exist.txt'])
    assert result.exit_code != 0


def test_doctor_reports_unreachable_ollama(monkeypatch):
    import requests as requests_module

    def refuse(*args, **kwargs):
        raise requests_module.ConnectionError('refused')

    monkeypatch.setattr('requests.get', refuse)
    result = CliRunner().invoke(cli, ['doctor'])
    assert result.exit_code == 1
    assert 'Ollama reachable' in result.output
    assert 'ollama.com/download' in result.output


def test_doctor_all_good(monkeypatch):
    class FakeResponse:
        def json(self):
            return {'models': [{'name': 'nomic-embed-text:latest'},
                               {'name': 'llama3.2:latest'}]}

    monkeypatch.setattr('requests.get', lambda *a, **k: FakeResponse())
    result = CliRunner().invoke(cli, ['doctor'])
    assert result.exit_code == 0, result.output
    assert 'All good!' in result.output


def test_validate_reports_question_count(input_file):
    result = CliRunner().invoke(cli, ['validate', str(input_file)])
    assert result.exit_code == 0, result.output
    assert 'Total questions found: 3' in result.output
