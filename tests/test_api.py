"""Tests for the Flask API server: jobs, history, health, and validation."""

import io
import json
import time
import zipfile
import threading

import numpy as np
import pytest

import api_server
from api_server import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server, 'ANALYSES_DIR', tmp_path / 'analyses')
    monkeypatch.setattr(api_server, 'JOBS_DIR', tmp_path / 'jobs')
    monkeypatch.setattr(api_server, '_jobs', {})
    app.config['TESTING'] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def fake_engine(monkeypatch):
    """Make QuestionAnalyzer run instantly with deterministic embeddings."""
    monkeypatch.setenv('GROUP_LABELS', 'off')  # no LLM calls in tests
    vectors = {
        'how do i reset my password?': [1.0, 0.0, 0.0],
        'how can i reset my password?': [0.99, 0.05, 0.0],
        'what is the deploy schedule for production releases?': [0.0, 1.0, 0.0],
    }

    def fake_batch(self, texts, progress_callback=None):
        if progress_callback:
            progress_callback(len(texts), len(texts))
        return np.array([vectors[t] for t in texts])

    monkeypatch.setattr('slack_question_analyzer.similarity_analyzer.SimilarityAnalyzer.get_embeddings_batch',
                        fake_batch)


SAMPLE_CONTENT = (
    "2024-01-05\nHow do I reset my password?\n"
    "-----------------------------------------------------------\n"
    "2024-01-08\nHow can I reset my password?\n"
    "-----------------------------------------------------------\n"
    "2024-01-09\nWhat is the deploy schedule for production releases?\n"
)


def wait_for_job(client, job_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f'/api/jobs/{job_id}')
        body = response.get_json()
        if body['status'] in ('done', 'error', 'cancelled'):
            return body
        time.sleep(0.05)
    pytest.fail('job did not finish in time')


def test_analyze_job_lifecycle(client, fake_engine):
    response = client.post('/api/analyze', json={'content': SAMPLE_CONTENT})
    assert response.status_code == 202
    job_id = response.get_json()['job_id']

    body = wait_for_job(client, job_id)
    assert body['status'] == 'done'
    assert body['data']['total_questions'] == 3
    assert body['data']['total_groups'] == 1
    assert body['analysis_id']
    assert body['progress']['stage'] == 'complete'


def test_analysis_is_persisted_and_listed(client, fake_engine):
    job_id = client.post('/api/analyze', json={'content': SAMPLE_CONTENT}).get_json()['job_id']
    finished = wait_for_job(client, job_id)

    listed = client.get('/api/analyses').get_json()
    assert listed['success']
    assert len(listed['analyses']) == 1
    assert listed['analyses'][0]['id'] == finished['analysis_id']
    assert listed['analyses'][0]['total_questions'] == 3

    latest = client.get('/api/analyses/latest').get_json()
    assert latest['success']
    assert latest['data']['total_questions'] == 3

    single = client.get(f"/api/analyses/{finished['analysis_id']}").get_json()
    assert single['data']['total_groups'] == 1


def test_latest_analysis_404_when_empty(client):
    assert client.get('/api/analyses/latest').status_code == 404


def test_saved_analyses_are_pruned_to_cap(client, fake_engine, monkeypatch):
    monkeypatch.setenv('MAX_SAVED_ANALYSES', '2')
    for _ in range(3):
        job_id = client.post('/api/analyze',
                             json={'content': SAMPLE_CONTENT}).get_json()['job_id']
        wait_for_job(client, job_id)
    assert len(client.get('/api/analyses').get_json()['analyses']) == 2


def test_export_analysis(client, fake_engine):
    job_id = client.post('/api/analyze', json={'content': SAMPLE_CONTENT}).get_json()['job_id']
    analysis_id = wait_for_job(client, job_id)['analysis_id']

    markdown = client.get(f'/api/analyses/{analysis_id}/export?format=md')
    assert markdown.status_code == 200
    assert markdown.mimetype == 'text/markdown'
    assert b'# Question Analysis Report' in markdown.data
    assert 'attachment' in markdown.headers['Content-Disposition']

    csv_export = client.get(f'/api/analyses/{analysis_id}/export?format=csv')
    assert csv_export.status_code == 200
    assert csv_export.mimetype == 'text/csv'
    assert b'group_rank' in csv_export.data

    json_export = client.get(f'/api/analyses/{analysis_id}/export?format=json')
    assert json_export.status_code == 200
    assert json_export.get_json()['total_questions'] == 3

    assert client.get(f'/api/analyses/{analysis_id}/export?format=pdf').status_code == 400
    assert client.get('/api/analyses/nope/export?format=md').status_code == 404


def test_faq_export(client, fake_engine):
    job_id = client.post('/api/analyze', json={'content': SAMPLE_CONTENT}).get_json()['job_id']
    analysis_id = wait_for_job(client, job_id)['analysis_id']

    faq = client.get(f'/api/analyses/{analysis_id}/export?format=faq')
    assert faq.status_code == 200
    assert faq.mimetype == 'text/markdown'
    assert b'# Draft FAQ' in faq.data
    assert b'faq-draft' in faq.headers['Content-Disposition'].encode()


def test_topic_history_dedupes_overlapping_uploads(client, tmp_path, monkeypatch):
    """The trends endpoint fingerprints occurrences (text+date), so a
    cumulative re-upload containing last month's messages doesn't double
    the curve. Two saved analyses share one occurrence; the history must
    count 3 unique occurrences, not 4."""
    analyses_dir = api_server.ANALYSES_DIR
    analyses_dir.mkdir(parents=True, exist_ok=True)

    def saved(name, questions):
        (analyses_dir / name).write_text(json.dumps({
            'groups': [{
                'topic_id': 'topic-1',
                'representative_question': questions[0][0],
                'count': len(questions),
                'avg_similarity': 0.9,
                'questions': [{'text': t, 'normalized_text': t.lower(),
                               'date': d} for t, d in questions],
            }],
            'ungrouped_questions': [],
            'metadata': {'analyzed_at': '2026-07-01T00:00:00Z'},
        }), encoding='utf-8')

    saved('20260601-000000-aaa.json', [
        ('How do I raise the SFTP timeout?', 'June 2, 2026'),
        ('Transfers time out on big files?', 'June 9, 2026'),
    ])
    saved('20260701-000000-bbb.json', [
        ('Transfers time out on big files?', 'June 9, 2026'),  # overlap
        ('What controls the SFTP wait time?', 'June 30, 2026'),
    ])

    history = client.get('/api/topics/topic-1/history')
    assert history.status_code == 200
    payload = history.get_json()
    assert payload['total_occurrences'] == 3   # not 4
    assert payload['analyses_with_topic'] == 2
    assert sum(p['count'] for p in payload['series']) == 3
    assert payload['first_asked'] == '2026-06-01'  # Monday of June 2 week

    assert client.get('/api/topics/nope/history').status_code == 404


def test_delete_analysis(client, fake_engine):
    job_id = client.post('/api/analyze', json={'content': SAMPLE_CONTENT}).get_json()['job_id']
    analysis_id = wait_for_job(client, job_id)['analysis_id']

    assert client.delete(f'/api/analyses/{analysis_id}').status_code == 200
    assert client.get(f'/api/analyses/{analysis_id}').status_code == 404
    assert client.get('/api/analyses').get_json()['analyses'] == []

    # Deleting again 404s; traversal attempts are rejected (404 from the
    # path check or 405 when routing diverts them to the GET-only UI route)
    assert client.delete(f'/api/analyses/{analysis_id}').status_code == 404
    assert client.delete('/api/analyses/..%2F..%2Fetc%2Fpasswd').status_code in (404, 405)


def test_multipart_single_file_upload(client, fake_engine):
    response = client.post('/api/analyze', data={
        'files': (io.BytesIO(SAMPLE_CONTENT.encode('utf-8')), 'export.txt'),
        'provider': 'ollama',
        'threshold': '0.85',
    }, content_type='multipart/form-data')
    assert response.status_code == 202

    body = wait_for_job(client, response.get_json()['job_id'])
    assert body['status'] == 'done'
    assert body['data']['total_questions'] == 3


def test_zip_upload_merges_day_files(client, fake_engine):
    """A zipped Slack export (per-day JSON files) is analyzed as one corpus."""
    day1 = json.dumps([{'text': 'How do I reset my password?', 'ts': '1704412800.0'}])
    day2 = json.dumps([{'text': 'What is the deploy schedule for production releases?',
                        'ts': '1704499200.0'}])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('channel/2024-01-05.json', day1)
        archive.writestr('channel/2024-01-06.json', day2)
        archive.writestr('__MACOSX/channel/._2024-01-05.json', 'junk')
        archive.writestr('channel/image.png', b'\x89PNG')
    buffer.seek(0)

    response = client.post('/api/analyze', data={
        'files': (buffer, 'slack-export.zip'),
    }, content_type='multipart/form-data')
    assert response.status_code == 202

    body = wait_for_job(client, response.get_json()['job_id'])
    assert body['status'] == 'done'
    assert body['data']['total_questions'] == 2  # merged across both day files


def test_zip_without_text_files_rejected(client):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('image.png', b'\x89PNG')
    buffer.seek(0)
    response = client.post('/api/analyze', data={
        'files': (buffer, 'export.zip'),
    }, content_type='multipart/form-data')
    assert response.status_code == 400
    assert 'no .json' in response.get_json()['error']


def test_cancel_running_job(client, monkeypatch):
    monkeypatch.setenv('GROUP_LABELS', 'off')
    started = threading.Event()
    release = threading.Event()

    def slow_batch(self, texts, progress_callback=None):
        started.set()
        release.wait(timeout=5)
        if progress_callback:
            progress_callback(1, 1)  # raises JobCancelled once flagged
        return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(
        'slack_question_analyzer.similarity_analyzer.SimilarityAnalyzer.get_embeddings_batch',
        slow_batch)

    content = ("How do I reset my password?\n"
               "-----------------------------------------------------------\n"
               "What is the deploy schedule for production releases?\n")
    job_id = client.post('/api/analyze', json={'content': content}).get_json()['job_id']

    assert started.wait(timeout=5)
    assert client.post(f'/api/jobs/{job_id}/cancel').status_code == 200
    release.set()

    body = wait_for_job(client, job_id)
    assert body['status'] == 'cancelled'
    # Cancelled runs leave no saved analysis behind
    assert client.get('/api/analyses').get_json()['analyses'] == []


def test_cancel_unknown_job(client):
    assert client.post('/api/jobs/nope/cancel').status_code == 404


def test_interrupted_jobs_recover_after_restart(client, fake_engine):
    """A job that was queued/running when the server died is re-queued."""
    job_id = 'restartedjob123'
    api_server.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with open(api_server._job_meta_path(job_id), 'w', encoding='utf-8') as f:
        json.dump({'status': 'running', 'provider': 'ollama', 'threshold': 0.85,
                   'created_at': time.time(), 'finished_at': None,
                   'analysis_id': None, 'error': None}, f)
    with open(api_server._job_content_path(job_id), 'w', encoding='utf-8') as f:
        json.dump([SAMPLE_CONTENT], f)

    assert api_server.recover_interrupted_jobs() == 1

    body = wait_for_job(client, job_id)
    assert body['status'] == 'done'
    assert body['data']['total_questions'] == 3
    assert len(client.get('/api/analyses').get_json()['analyses']) == 1


def test_done_job_status_served_from_disk_after_restart(client, fake_engine):
    job_id = client.post('/api/analyze', json={'content': SAMPLE_CONTENT}).get_json()['job_id']
    finished = wait_for_job(client, job_id)

    # Simulate a restart wiping in-memory job state
    with api_server._jobs_lock:
        api_server._jobs.clear()

    body = client.get(f'/api/jobs/{job_id}').get_json()
    assert body['status'] == 'done'
    assert body['analysis_id'] == finished['analysis_id']
    assert body['data']['total_questions'] == 3


def test_topics_endpoint(client):
    from slack_question_analyzer.topic_bank import TopicBank
    bank = TopicBank()
    bank.record({'topic': 'Password Reset', 'summary': 's',
                 'representative_question': 'How do I reset?',
                 'keywords': ['password'], 'count': 4}, [1.0, 0.0])
    bank.save()

    body = client.get('/api/topics').get_json()
    assert body['success']
    assert body['topics'][0]['topic'] == 'Password Reset'
    assert body['topics'][0]['question_count'] == 4
    assert 'centroid' not in body['topics'][0]  # internal detail stays internal


def test_rename_topic_endpoint(client):
    from slack_question_analyzer.topic_bank import TopicBank
    bank = TopicBank()
    entry = bank.record({'topic': 'Bad Name', 'summary': None,
                         'representative_question': 'How do I reset?',
                         'keywords': [], 'count': 2}, [1.0, 0.0])
    bank.save()

    response = client.patch(f"/api/topics/{entry['id']}", json={'topic': 'Password Reset'})
    assert response.status_code == 200
    assert TopicBank().entries[0]['topic'] == 'Password Reset'

    assert client.patch('/api/topics/nonexistent', json={'topic': 'X'}).status_code == 404
    assert client.patch(f"/api/topics/{entry['id']}", json={'topic': '  '}).status_code == 400


def test_delete_and_merge_topic_endpoints(client):
    from slack_question_analyzer.topic_bank import TopicBank
    bank = TopicBank()
    keep = bank.record({'topic': 'Virus Scanning', 'summary': None,
                        'representative_question': 'a?', 'keywords': [], 'count': 4},
                       [1.0, 0.0])
    junk = bank.record({'topic': 'Scanning Stuff', 'summary': None,
                        'representative_question': 'b?', 'keywords': [], 'count': 2},
                       [0.95, 0.31])
    bank.save()

    merged = client.post(f"/api/topics/{junk['id']}/merge", json={'into': keep['id']})
    assert merged.status_code == 200
    reloaded = TopicBank()
    assert len(reloaded.entries) == 1
    assert reloaded.entries[0]['question_count'] == 6

    assert client.post(f"/api/topics/{keep['id']}/merge", json={}).status_code == 400
    assert client.delete(f"/api/topics/{keep['id']}").status_code == 200
    assert TopicBank().entries == []
    assert client.delete('/api/topics/nonexistent').status_code == 404


def test_health_flags_old_ollama_version(client, monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_get(url, timeout=None):
        if url.endswith('/api/version'):
            return FakeResponse({'version': '0.3.9'})
        return FakeResponse({'models': [{'name': 'nomic-embed-text:latest'},
                                        {'name': 'llama3.2:latest'}]})

    monkeypatch.setattr(api_server.requests, 'get', fake_get)
    body = client.get('/api/health').get_json()
    assert body['status'] == 'degraded'
    assert 'too old' in body['message']
    assert body['ollama']['version'] == '0.3.9'


def test_choose_port_skips_occupied():
    import socket
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(('127.0.0.1', 0))
    busy_port = blocker.getsockname()[1]
    try:
        chosen = api_server.choose_port('127.0.0.1', busy_port)
        assert chosen != busy_port
        assert busy_port < chosen <= busy_port + 5
    finally:
        blocker.close()


def test_model_pull_endpoint(client, monkeypatch):
    monkeypatch.setattr(api_server, '_model_pulls', {})

    class FakePullResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield json.dumps({'status': 'downloading', 'total': 100, 'completed': 50}).encode()
            yield json.dumps({'status': 'success', 'total': 100, 'completed': 100}).encode()

    monkeypatch.setattr(api_server.requests, 'post', lambda *a, **k: FakePullResponse())

    # Only configured models can be pulled
    assert client.post('/api/models/pull', json={'model': 'evil/backdoor'}).status_code == 400

    assert client.post('/api/models/pull', json={'model': 'nomic-embed-text'}).status_code == 202
    deadline = time.time() + 5
    while time.time() < deadline:
        status = client.get('/api/models/pull/nomic-embed-text').get_json()
        if status['status'] in ('done', 'error'):
            break
        time.sleep(0.05)
    assert status['status'] == 'done'
    assert status['completed'] == 100


def test_model_pull_reports_errors(client, monkeypatch):
    monkeypatch.setattr(api_server, '_model_pulls', {})

    def refuse(*args, **kwargs):
        raise api_server.requests.exceptions.ConnectionError('refused')

    monkeypatch.setattr(api_server.requests, 'post', refuse)
    assert client.post('/api/models/pull', json={'model': 'llama3.2'}).status_code == 202
    deadline = time.time() + 5
    while time.time() < deadline:
        status = client.get('/api/models/pull/llama3.2').get_json()
        if status['status'] in ('done', 'error'):
            break
        time.sleep(0.05)
    assert status['status'] == 'error'

    assert client.get('/api/models/pull/never-started').status_code == 404


def test_health_ignores_provider_param_and_reports_ollama(client):
    """Local-only by design: health always reports the Ollama backend,
    whatever a stale client sends."""
    health = client.get('/api/health?provider=azure').get_json()
    assert health['provider'] == 'ollama'
    assert 'ollama' in health


def test_weekly_stats_endpoint(client, fake_engine):
    job_id = client.post('/api/analyze', json={'content': SAMPLE_CONTENT}).get_json()['job_id']
    wait_for_job(client, job_id)

    latest = client.get('/api/analyses/latest/weekly')
    assert latest.status_code == 200
    weekly = latest.get_json()['data']
    # CALENDAR weeks: Jan 8 (Mon) and Jan 9 land in the week of Jan 8-14;
    # Jan 5 (Fri) belongs to the previous week (Jan 1-7)
    assert weekly['week'] == '2024-01-08'
    assert weekly['weekLabel'] == 'Jan 8 – 14, 2024'
    assert weekly['totalThisWeek'] == 2
    assert weekly['totalLastWeek'] == 1
    assert weekly['trendWeeks'][-1] == '2024-01-08'

    # ?week= reviews a past calendar week (any date inside it)
    past = client.get('/api/analyses/latest/weekly?week=2024-01-03')
    assert past.status_code == 200
    past_weekly = past.get_json()['data']
    assert past_weekly['week'] == '2024-01-01'
    assert past_weekly['totalThisWeek'] == 1
    assert past_weekly['latestWeek'] == '2024-01-08'

    assert client.get('/api/analyses/latest/weekly?week=garbage').status_code == 400


def test_weekly_stats_404s(client):
    assert client.get('/api/analyses/latest/weekly').status_code == 404
    assert client.get('/api/analyses/nope/weekly').status_code == 404


def test_analysis_path_traversal_blocked(client):
    assert client.get('/api/analyses/..%2F..%2Fetc%2Fpasswd').status_code == 404


def test_unknown_job(client):
    assert client.get('/api/jobs/nope').status_code == 404


def test_analyze_accepts_auto_threshold(client, fake_engine):
    response = client.post('/api/analyze',
                           json={'content': SAMPLE_CONTENT, 'threshold': 'auto'})
    assert response.status_code == 202
    body = wait_for_job(client, response.get_json()['job_id'])
    assert body['status'] == 'done'


def test_analyze_validation(client):
    assert client.post('/api/analyze', json={}).status_code == 400
    assert client.post('/api/analyze', json={'content': '  '}).status_code == 400
    assert client.post('/api/analyze',
                       json={'content': 'x?', 'provider': 'bogus'}).status_code == 400
    assert client.post('/api/analyze',
                       json={'content': 'x?', 'threshold': 2}).status_code == 400
    assert client.post('/api/analyze',
                       json={'content': 'x?', 'threshold': 'abc'}).status_code == 400


def test_health_reports_unreachable_ollama(client, monkeypatch):
    import requests as requests_module

    def refuse(*args, **kwargs):
        raise requests_module.ConnectionError('connection refused')

    monkeypatch.setattr(api_server.requests, 'get', refuse)
    body = client.get('/api/health').get_json()
    assert body['status'] == 'unavailable'
    assert body['ollama']['reachable'] is False
    assert 'ollama serve' in body['message']


def test_health_reports_missing_model(client, monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {'models': [{'name': 'llama3:latest'}]}

    monkeypatch.setattr(api_server.requests, 'get', lambda *a, **k: FakeResponse())
    body = client.get('/api/health').get_json()
    assert body['status'] == 'degraded'
    assert body['ollama']['reachable'] is True
    assert body['ollama']['model_available'] is False


def test_health_ok(client, monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {'models': [{'name': 'nomic-embed-text:latest'}]}

    monkeypatch.setattr(api_server.requests, 'get', lambda *a, **k: FakeResponse())
    body = client.get('/api/health').get_json()
    assert body['status'] == 'ok'
    assert body['ollama']['model_available'] is True


def test_ui_is_served(client):
    response = client.get('/')
    assert response.status_code == 302
    assert '/ui_kits/analyzer/index.html' in response.headers['Location']

    page = client.get('/ui_kits/analyzer/index.html')
    assert page.status_code == 200
    assert b'Question Analyzer' in page.data

    assert client.get('/ui_kits/analyzer/api.js').status_code == 200
    assert client.get('/styles.css').status_code == 200


def test_topic_history_follows_merge_aliases(client, tmp_path, monkeypatch):
    """Analyses saved BEFORE a topic merge recorded the old topic_id; the
    merged-away id lives on as an alias, so the history chart must include
    that older volume instead of losing it."""
    analyses_dir = api_server.ANALYSES_DIR
    analyses_dir.mkdir(parents=True, exist_ok=True)

    bank_path = tmp_path / 'bank.json'
    bank_path.write_text(json.dumps([{
        'id': 'kept-id', 'topic': 'SFTP Timeouts', 'centroid': [1.0, 0.0],
        'question_count': 2, 'merged_ids': ['merged-away-id'],
    }]), encoding='utf-8')
    monkeypatch.setenv('TOPIC_BANK_PATH', str(bank_path))

    def saved(name, topic_id, questions):
        (analyses_dir / name).write_text(json.dumps({
            'groups': [{
                'topic_id': topic_id,
                'representative_question': questions[0][0],
                'count': len(questions),
                'avg_similarity': 0.9,
                'questions': [{'text': t, 'normalized_text': t.lower(),
                               'date': d} for t, d in questions],
            }],
            'ungrouped_questions': [],
            'metadata': {'analyzed_at': '2026-07-01T00:00:00Z'},
        }), encoding='utf-8')

    saved('20260601-000000-aaa.json', 'merged-away-id',
          [('How do I raise the SFTP timeout?', 'June 2, 2026')])
    saved('20260701-000000-bbb.json', 'kept-id',
          [('What controls the SFTP wait time?', 'June 30, 2026')])

    payload = client.get('/api/topics/kept-id/history').get_json()
    assert payload['total_occurrences'] == 2
    assert payload['analyses_with_topic'] == 2

    # Second request exercises the mtime-keyed parse cache: same answer
    payload2 = client.get('/api/topics/kept-id/history').get_json()
    assert payload2['total_occurrences'] == 2


def test_multiple_zips_share_one_decompression_budget(client, fake_engine, monkeypatch):
    """Three high-ratio archives must not each get the full budget."""
    monkeypatch.setattr(api_server, 'MAX_CONTENT_MB', 1)

    def make_zip():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('big.txt', 'How do I reset my password?\n' * 30000)  # ~0.8MB
        buf.seek(0)
        return buf

    response = client.post('/api/analyze', data={
        'files': [(make_zip(), 'a.zip'), (make_zip(), 'b.zip')],
    }, content_type='multipart/form-data')
    assert response.status_code == 400
    assert 'limit' in response.get_json()['error'].lower()


def test_set_and_clear_curated_answer(client, tmp_path, monkeypatch):
    bank_path = tmp_path / 'bank.json'
    bank_path.write_text(json.dumps([{
        'id': 'topic-1', 'topic': 'SFTP Timeouts', 'centroid': [1.0, 0.0],
        'question_count': 2,
    }]), encoding='utf-8')
    monkeypatch.setenv('TOPIC_BANK_PATH', str(bank_path))

    response = client.post('/api/topics/topic-1/answer',
                           json={'answer': 'Set transfer.timeout to 300.'})
    assert response.status_code == 200
    assert response.get_json()['curated_answer'] == 'Set transfer.timeout to 300.'

    topics = client.get('/api/topics').get_json()['topics']
    assert topics[0]['curated_answer'] == 'Set transfer.timeout to 300.'
    assert topics[0]['answer_updated']

    assert client.post('/api/topics/topic-1/answer',
                       json={'answer': ''}).status_code == 200
    topics = client.get('/api/topics').get_json()['topics']
    assert topics[0]['curated_answer'] is None

    assert client.post('/api/topics/nope/answer',
                       json={'answer': 'x'}).status_code == 404


def test_faq_effect_verdict(client, tmp_path, monkeypatch):
    """Publish-effectiveness: weekly ask rate before vs after the publish
    date, anchored to data time. Asks that stop after publishing produce
    'working'; a publish date at the data edge produces 'too_early'."""
    analyses_dir = api_server.ANALYSES_DIR
    analyses_dir.mkdir(parents=True, exist_ok=True)

    def saved(name, topic_id, questions):
        (analyses_dir / name).write_text(json.dumps({
            'groups': [{
                'topic_id': topic_id,
                'representative_question': questions[0][0],
                'count': len(questions),
                'avg_similarity': 0.9,
                'questions': [{'text': t, 'normalized_text': t.lower(),
                               'date': d} for t, d in questions],
            }],
            'ungrouped_questions': [],
            'metadata': {'analyzed_at': '2026-07-01T00:00:00Z'},
        }), encoding='utf-8')

    bank_path = tmp_path / 'bank.json'
    bank_path.write_text(json.dumps([
        {'id': 'topic-1', 'topic': 'Hot', 'centroid': [1.0, 0.0],
         'question_count': 6, 'faq_published': '2026-06-01'},
        {'id': 'topic-2', 'topic': 'Fresh', 'centroid': [0.0, 1.0],
         'question_count': 2, 'faq_published': '2026-06-28'},
    ]), encoding='utf-8')
    monkeypatch.setenv('TOPIC_BANK_PATH', str(bank_path))

    # topic-1: 6 asks in the 4 weeks before June 1, none after; topic-2
    # keeps the data anchor at June 30 (2 days after its publish date)
    saved('20260601-000000-aaa.json', 'topic-1', [
        (f'Ask {i}?', d) for i, d in enumerate([
            '2026-05-05', '2026-05-08', '2026-05-12', '2026-05-19',
            '2026-05-22', '2026-05-29'])])
    saved('20260701-000000-bbb.json', 'topic-2', [
        ('Fresh ask?', '2026-06-25'), ('Fresh ask again?', '2026-06-30')])

    effect = client.get('/api/topics/topic-1/history').get_json()['faq_effect']
    assert effect['verdict'] == 'working'
    assert effect['after_per_week'] == 0
    assert effect['change_pct'] == -100

    effect = client.get('/api/topics/topic-2/history').get_json()['faq_effect']
    assert effect['verdict'] == 'too_early'


def test_topic_history_flags_stale_curated_answer(client, tmp_path, monkeypatch):
    """Asks answered in threads AFTER the curated answer was saved surface
    as an answer_stale nudge; topics without a curated answer never do."""
    analyses_dir = api_server.ANALYSES_DIR
    analyses_dir.mkdir(parents=True, exist_ok=True)

    bank_path = tmp_path / 'bank.json'
    bank_path.write_text(json.dumps([{
        'id': 'topic-1', 'topic': 'SFTP Timeouts', 'centroid': [1.0, 0.0],
        'question_count': 2, 'curated_answer': 'Set transfer.timeout to 300.',
        'answer_updated': '2026-06-01',
    }]), encoding='utf-8')
    monkeypatch.setenv('TOPIC_BANK_PATH', str(bank_path))

    (analyses_dir / '20260701-000000-aaa.json').write_text(json.dumps({
        'groups': [{
            'topic_id': 'topic-1',
            'representative_question': 'How do I raise the SFTP timeout?',
            'count': 2, 'avg_similarity': 0.9,
            'questions': [
                {'text': 'Old ask?', 'normalized_text': 'old ask?',
                 'date': '2026-05-20', 'answered': True},
                {'text': 'New ask?', 'normalized_text': 'new ask?',
                 'date': '2026-06-20', 'answered': True},
            ],
        }],
        'ungrouped_questions': [],
        'metadata': {'analyzed_at': '2026-07-01T00:00:00Z'},
    }), encoding='utf-8')

    payload = client.get('/api/topics/topic-1/history').get_json()
    assert payload['answer_stale'] == {'new_answered': 1,
                                       'answer_updated': '2026-06-01'}

    # Clear the curated answer: the nudge disappears
    assert client.post('/api/topics/topic-1/answer',
                       json={'answer': ''}).status_code == 200
    payload = client.get('/api/topics/topic-1/history').get_json()
    assert payload['answer_stale'] is None
