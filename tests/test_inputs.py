"""Tests for shared input loading (plain files and zipped exports)."""

import io
import json
import zipfile

import pytest

from slack_question_analyzer.inputs import contents_from_zip_bytes, load_input_files


def make_zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_zip_extraction_skips_junk():
    raw = make_zip({
        'channel/2024-01-05.json': '[{"text": "How do I reset?"}]',
        'channel/notes.txt': 'Why is this broken?',
        '__MACOSX/channel/._2024-01-05.json': 'resource fork junk',
        'channel/.hidden.json': 'hidden',
        'channel/photo.png': 'binary',
    })
    contents = contents_from_zip_bytes(raw)
    assert len(contents) == 2
    assert '[{"text": "How do I reset?"}]' in contents


def test_zip_size_limit():
    raw = make_zip({'big.txt': 'x' * 2048})
    with pytest.raises(ValueError, match='exceed'):
        contents_from_zip_bytes(raw, max_bytes=1024)


def test_zip_without_text_files():
    raw = make_zip({'image.png': 'binary'})
    with pytest.raises(ValueError, match='no .json'):
        contents_from_zip_bytes(raw)


def test_load_input_files_mixes_plain_and_zip(tmp_path):
    plain = tmp_path / 'day1.txt'
    plain.write_text('How do I reset my password?', encoding='utf-8')
    archive = tmp_path / 'export.zip'
    archive.write_bytes(make_zip({
        'day2.json': json.dumps([{'text': 'What is the deploy schedule?'}]),
    }))

    contents = load_input_files([str(plain), str(archive)])
    assert len(contents) == 2
    assert contents[0] == 'How do I reset my password?'
    assert 'deploy schedule' in contents[1]
