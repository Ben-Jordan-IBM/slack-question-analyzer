"""
Flask API Server for Slack Question Analyzer

Serves both the REST API and the React dashboard, so the whole app runs
with a single command:

    python api_server.py        ->  http://localhost:5000

Endpoints:
    GET  /                      Dashboard UI
    GET  /api/health            Health check (verifies Ollama when configured)
    POST /api/analyze           Start an analysis job, returns {job_id}
    GET  /api/jobs/<job_id>     Job status, real progress, and result
    GET  /api/analyses          List of saved past analyses (newest first)
    GET  /api/analyses/latest   Most recent saved analysis
    GET  /api/analyses/<id>     A specific saved analysis
"""

import os
import json
import re
import time
import uuid
import logging
import threading
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_from_directory, redirect, Response
from flask_cors import CORS
from dotenv import load_dotenv

from slack_question_analyzer.analyzer import QuestionAnalyzer
from slack_question_analyzer.weekly_stats import compute_weekly_stats
from slack_question_analyzer.exporters import to_csv, to_markdown, to_faq_markdown
from slack_question_analyzer.inputs import contents_from_zip_bytes

load_dotenv()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UI_DIR = BASE_DIR / 'Question Analyzer Design System'
ANALYSES_DIR = Path(os.getenv('ANALYSES_DIR', BASE_DIR / 'analyses'))
JOBS_DIR = Path(os.getenv('JOBS_DIR', BASE_DIR / 'jobs'))

VALID_PROVIDERS = ('ollama',)  # local-only by design
MAX_CONTENT_MB = int(os.getenv('MAX_CONTENT_MB', '50'))
JOB_RETENTION_SECONDS = 3600  # finished jobs are pruned after an hour
# Local models degrade badly under parallel analyses; queue jobs by default
MAX_CONCURRENT_JOBS = int(os.getenv('MAX_CONCURRENT_JOBS', '1'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024
# Only the local UI may call the API cross-origin. A wildcard here would let
# any website the browser visits read saved analyses off localhost. 'null'
# (the file:// origin) is deliberately NOT allowed: any site can forge it
# with a sandboxed iframe, which reopens the exact hole this closes. The
# dashboard is served same-origin by this server, so it needs no CORS at all.
CORS(app, origins=[re.compile(r'^https?://(localhost|127\.0\.0\.1)(:\d+)?$')])

# ---------------------------------------------------------------------------
# Job management
#
# Jobs run in worker threads, queued behind a semaphore. Job metadata and the
# uploaded content are persisted under JOBS_DIR so that interrupted jobs can
# be re-queued after a server restart (see recover_interrupted_jobs).
# ---------------------------------------------------------------------------

_jobs = {}
_jobs_lock = threading.Lock()
_job_slots = threading.Semaphore(MAX_CONCURRENT_JOBS)

FINAL_STATUSES = ('done', 'error', 'cancelled')


class JobCancelled(Exception):
    """Raised inside a worker when the user cancels the job."""


def _job_meta_path(job_id):
    return JOBS_DIR / f"{job_id}.json"


def _job_content_path(job_id):
    return JOBS_DIR / f"{job_id}.content.json"


def _persist_job(job_id):
    """Write job metadata to disk (caller holds the lock). Best-effort."""
    job = _jobs.get(job_id)
    if job is None:
        return
    meta = {key: job.get(key) for key in
            ('status', 'analysis_id', 'error', 'created_at', 'finished_at',
             'provider', 'threshold', 'cancelled')}
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_job_meta_path(job_id), 'w', encoding='utf-8') as f:
            json.dump(meta, f)
    except OSError as e:
        logger.warning("Could not persist job %s: %s", job_id, e)


def _cleanup_job_files(job_id, meta_too=False):
    for path in ([_job_content_path(job_id), _job_meta_path(job_id)] if meta_too
                 else [_job_content_path(job_id)]):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _prune_jobs():
    """Drop finished jobs older than the retention window (caller holds lock)."""
    cutoff = time.time() - JOB_RETENTION_SECONDS
    stale = [job_id for job_id, job in _jobs.items()
             if job['status'] in FINAL_STATUSES and job['finished_at'] and job['finished_at'] < cutoff]
    for job_id in stale:
        del _jobs[job_id]
        _cleanup_job_files(job_id, meta_too=True)


def _start_job(contents, provider, threshold):
    """Register and launch an analysis job; returns the job id."""
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _prune_jobs()
        _jobs[job_id] = {
            'status': 'queued',
            'progress': {'stage': 'queued', 'completed': 0, 'total': 1},
            'result': None,
            'analysis_id': None,
            'error': None,
            'cancelled': False,
            'provider': provider,
            'threshold': threshold,
            'created_at': time.time(),
            'finished_at': None,
        }
        _persist_job(job_id)

    # Persist the content so the job survives a server restart
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_job_content_path(job_id), 'w', encoding='utf-8') as f:
            json.dump(contents, f, ensure_ascii=False)
    except OSError as e:
        logger.warning("Could not persist job content for %s: %s", job_id, e)

    thread = threading.Thread(target=_run_analysis_job,
                              args=(job_id, contents, provider, threshold),
                              daemon=True)
    thread.start()
    return job_id


def _finish_job(job_id, **fields):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            job.update(finished_at=time.time(), **fields)
            _persist_job(job_id)
    _cleanup_job_files(job_id)


def _run_analysis_job(job_id, contents, provider, threshold):
    """Worker thread: run the analysis, report progress, persist the result."""
    def on_progress(stage, completed, total):
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                if job['cancelled']:
                    raise JobCancelled()
                job['progress'] = {'stage': stage, 'completed': completed, 'total': total}

    # Queue behind any running analysis so a local Ollama isn't overloaded
    with _job_slots:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            if job['cancelled']:
                job.update(status='cancelled', finished_at=time.time())
                _persist_job(job_id)
                _cleanup_job_files(job_id)
                return
            job['status'] = 'running'
            job['progress'] = {'stage': 'starting', 'completed': 0, 'total': 1}
            _persist_job(job_id)

        def cancel_check():
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job and job['cancelled']:
                    raise JobCancelled()

        try:
            analyzer = QuestionAnalyzer(provider=provider, threshold=threshold)
            results = analyzer.analyze_contents(contents, progress_callback=on_progress,
                                                cancel_check=cancel_check)
            analysis_id = _save_analysis(results)
            _finish_job(job_id, status='done', result=results, analysis_id=analysis_id)
        except JobCancelled:
            logger.info("Analysis job %s cancelled", job_id)
            _finish_job(job_id, status='cancelled')
        except Exception as e:
            logger.exception("Analysis job %s failed", job_id)
            _finish_job(job_id, status='error', error=str(e))


def recover_interrupted_jobs():
    """
    Re-queue jobs that were queued/running when the server last stopped.
    Called at server startup. Returns the number of re-queued jobs.
    """
    if not JOBS_DIR.is_dir():
        return 0

    recovered = 0
    for meta_path in JOBS_DIR.glob('*.json'):
        if meta_path.stem.endswith('.content'):
            continue
        job_id = meta_path.stem
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get('status') not in ('queued', 'running'):
            # Finished-job metas from before a restart are invisible to
            # _prune_jobs (it only walks the in-memory dict) — sweep the
            # expired ones here so JOBS_DIR can't grow forever
            finished = meta.get('finished_at') or meta.get('created_at') or 0
            if meta.get('status') in FINAL_STATUSES and \
                    finished < time.time() - JOB_RETENTION_SECONDS:
                _cleanup_job_files(job_id, meta_too=True)
            continue

        if meta.get('cancelled'):
            # Cancelled while queued/running, then the server stopped before
            # the worker observed the flag: honor the cancel, don't resurrect
            meta.update(status='cancelled', finished_at=time.time())
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f)
            _cleanup_job_files(job_id)
            continue

        content_path = _job_content_path(job_id)
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                contents = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta.update(status='error', finished_at=time.time(),
                        error='Server restarted and the uploaded content was lost; please re-run.')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f)
            continue

        provider = (meta.get('provider')
                    or os.getenv('AI_PROVIDER', 'ollama')).strip().lower()
        threshold = meta.get('threshold')  # None = auto
        with _jobs_lock:
            _jobs[job_id] = {
                'status': 'queued',
                'progress': {'stage': 'queued', 'completed': 0, 'total': 1},
                'result': None,
                'analysis_id': None,
                'error': None,
                'cancelled': False,
                'provider': provider,
                'threshold': threshold,
                'created_at': meta.get('created_at') or time.time(),
                'finished_at': None,
            }
            _persist_job(job_id)
        threading.Thread(target=_run_analysis_job,
                         args=(job_id, contents, provider, threshold),
                         daemon=True).start()
        recovered += 1

    if recovered:
        logger.info("Re-queued %d interrupted job(s) after restart", recovered)
    return recovered


# ---------------------------------------------------------------------------
# Analysis history persistence
# ---------------------------------------------------------------------------

def _save_analysis(results):
    """Persist a completed analysis to disk and return its id."""
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    analysis_id = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    path = ANALYSES_DIR / f"{analysis_id}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)

    # Keep history bounded: prune the oldest beyond MAX_SAVED_ANALYSES (0 = keep all)
    keep = int(os.getenv('MAX_SAVED_ANALYSES', '200'))
    if keep > 0:
        saved = sorted(ANALYSES_DIR.glob('*.json'))
        for stale in saved[:-keep]:
            try:
                stale.unlink()
            except OSError:
                pass
    return analysis_id


def _list_analyses():
    """Summaries of saved analyses, newest first."""
    if not ANALYSES_DIR.is_dir():
        return []
    summaries = []
    for path in sorted(ANALYSES_DIR.glob('*.json'), reverse=True):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        top = data.get('groups') or []
        summaries.append({
            'id': path.stem,
            'analyzed_at': data.get('metadata', {}).get('analyzed_at'),
            'total_questions': data.get('total_questions', 0),
            'total_groups': data.get('total_groups', 0),
            'top_question': top[0]['representative_question'] if top else None,
        })
    return summaries


def _load_analysis(analysis_id):
    # Resolve strictly inside ANALYSES_DIR to block path traversal
    path = (ANALYSES_DIR / f"{analysis_id}.json").resolve()
    if path.parent != ANALYSES_DIR.resolve() or not path.is_file():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# UI routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return redirect('/ui_kits/analyzer/index.html')


@app.route('/<path:asset_path>')
def ui_assets(asset_path):
    """Serve the dashboard and design-system assets."""
    return send_from_directory(UI_DIR, asset_path)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check: verifies the local Ollama backend is usable."""
    health = {'status': 'ok', 'provider': 'ollama'}

    ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
    model = os.getenv('OLLAMA_MODEL', 'nomic-embed-text')
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=3)
        response.raise_for_status()
        models = [m.get('name', '') for m in response.json().get('models', [])]
        model_available = any(name == model or name.startswith(f"{model}:") for name in models)
        from slack_question_analyzer.model_defaults import (
            default_generation_model, FALLBACK_GENERATION_MODEL)
        label_model = default_generation_model()
        def _downloaded(m):
            return any(name == m or name.startswith(f"{m}:") for name in models)
        label_available = _downloaded(label_model)
        # Mirror GroupLabeler's runtime fallback: if only the small chat
        # model is downloaded, that's what will actually be used — report
        # it (and don't nag the user to download the big one)
        if (not label_available and not os.getenv('OLLAMA_GENERATION_MODEL')
                and _downloaded(FALLBACK_GENERATION_MODEL)):
            label_model, label_available = FALLBACK_GENERATION_MODEL, True
        health['ollama'] = {'reachable': True, 'model': model,
                            'model_available': model_available,
                            'label_model': label_model,
                            'label_model_available': label_available}

        # Structured output (format=<schema>) needs Ollama >= 0.5
        try:
            version = requests.get(f"{ollama_url}/api/version",
                                   timeout=3).json().get('version', '')
            health['ollama']['version'] = version
            parts = [int(p) for p in version.split('.')[:2] if p.isdigit()]
            if len(parts) == 2 and tuple(parts) < (0, 5):
                health['status'] = 'degraded'
                health['message'] = (f"Ollama {version} is too old for structured "
                                     "output — update from https://ollama.com/download")
        except (requests.RequestException, ValueError):
            pass  # older Ollama without /api/version; treated as unknown

        if not model_available:
            health['status'] = 'degraded'
            health['message'] = (f"Ollama is running but model '{model}' is not "
                                 f"pulled. Run: ollama pull {model}")
    except requests.RequestException as e:
        health['status'] = 'unavailable'
        health['ollama'] = {'reachable': False, 'model': model, 'model_available': False}
        health['message'] = (f"Cannot reach Ollama at {ollama_url} ({e.__class__.__name__}). "
                             "Start it with: ollama serve")
    return jsonify(health)


def _contents_from_upload():
    """Contents from a multipart upload: plain files and/or .zip archives."""
    files = request.files.getlist('files')
    if not files and 'file' in request.files:
        files = [request.files['file']]
    if not files:
        raise ValueError('No files uploaded (use the "files" field)')

    contents = []
    # All zips in one upload share a single decompression budget: three
    # high-ratio archives must not turn the request cap into 3x the memory
    zip_budget = MAX_CONTENT_MB * 1024 * 1024
    for storage in files:
        raw = storage.read()
        if (storage.filename or '').lower().endswith('.zip'):
            extracted = contents_from_zip_bytes(raw, max_bytes=zip_budget)
            zip_budget -= sum(len(c) for c in extracted)
            contents.extend(extracted)
        else:
            contents.append(raw.decode('utf-8', errors='replace'))
    return contents


@app.route('/api/topics', methods=['GET'])
def list_topics():
    """The learned topic bank: known topics accumulated across analyses."""
    from slack_question_analyzer.topic_bank import TopicBank
    bank = TopicBank()
    fields = ('id', 'topic', 'summary', 'representative_question', 'keywords',
              'question_count', 'analysis_count', 'first_seen', 'last_seen',
              'faq_published', 'curated_answer', 'answer_updated')
    topics = [{key: entry.get(key) for key in fields}
              for entry in sorted(bank.entries,
                                  key=lambda e: e.get('question_count', 0), reverse=True)]
    return jsonify({'success': True, 'topics': topics})


# Bank mutations are load-modify-save on one JSON file; two concurrent
# requests interleaving that cycle can silently drop one edit. Serialize.
_bank_lock = threading.Lock()


@app.route('/api/topics/<topic_id>', methods=['PATCH'])
def rename_topic(topic_id):
    """Rename a learned topic — the fix for a bad name sticking forever."""
    from slack_question_analyzer.topic_bank import TopicBank
    data = request.get_json(silent=True) or {}
    new_name = str(data.get('topic', '')).strip()
    if not new_name:
        return jsonify({'success': False, 'error': 'topic name cannot be empty'}), 400
    with _bank_lock:
        renamed = TopicBank().rename(topic_id, new_name)
    if not renamed:
        return jsonify({'success': False, 'error': 'Topic not found'}), 404
    return jsonify({'success': True, 'topic': new_name})


@app.route('/api/topics/<topic_id>', methods=['DELETE'])
def delete_topic(topic_id):
    """Remove a junk topic from the learned bank."""
    from slack_question_analyzer.topic_bank import TopicBank
    with _bank_lock:
        deleted = TopicBank().delete(topic_id)
    if not deleted:
        return jsonify({'success': False, 'error': 'Topic not found'}), 404
    return jsonify({'success': True})


@app.route('/api/topics/<topic_id>/published', methods=['POST'])
def set_topic_published(topic_id):
    """Mark a topic's FAQ as published (body: {"published": true|false}).
    The stamped date becomes a marker on the topic-history chart, so the
    user can see whether ask-volume fell after their doc went live."""
    from slack_question_analyzer.topic_bank import TopicBank
    data = request.get_json(silent=True) or {}
    with _bank_lock:
        result = TopicBank().set_published(topic_id, bool(data.get('published')))
    if result is None:
        return jsonify({'success': False, 'error': 'Topic not found'}), 404
    return jsonify({'success': True, 'faq_published': result})


@app.route('/api/topics/<topic_id>/answer', methods=['POST'])
def set_topic_answer(topic_id):
    """Save (or clear with an empty string) the curated FAQ answer for a
    topic (body: {"answer": "..."}). Once approved by a human, this wording
    becomes canonical: every FAQ export uses it instead of re-drafting."""
    from slack_question_analyzer.topic_bank import TopicBank
    data = request.get_json(silent=True) or {}
    with _bank_lock:
        result = TopicBank().set_answer(topic_id, str(data.get('answer', '')))
    if result is None:
        return jsonify({'success': False, 'error': 'Topic not found'}), 404
    return jsonify({'success': True, 'curated_answer': result})


@app.route('/api/topics/<topic_id>/merge', methods=['POST'])
def merge_topic(topic_id):
    """Merge this topic into another (body: {"into": "<target id>"})."""
    from slack_question_analyzer.topic_bank import TopicBank
    data = request.get_json(silent=True) or {}
    target_id = str(data.get('into', '')).strip()
    if not target_id:
        return jsonify({'success': False, 'error': "Missing 'into' target topic id"}), 400
    with _bank_lock:
        merged = TopicBank().merge(topic_id, target_id)
    if not merged:
        return jsonify({'success': False, 'error': 'Topic not found (or merging into itself)'}), 404
    return jsonify({'success': True})


# ---------------------------------------------------------------------------
# Model management: pull missing Ollama models from the dashboard, so new
# users never have to run 'ollama pull' by hand
# ---------------------------------------------------------------------------

_model_pulls = {}
_pulls_lock = threading.Lock()


def _pullable_models():
    from slack_question_analyzer.model_defaults import (
        default_generation_model, PREFERRED_GENERATION_MODEL, FALLBACK_GENERATION_MODEL)
    return {os.getenv('OLLAMA_MODEL', 'nomic-embed-text'),
            default_generation_model(),
            PREFERRED_GENERATION_MODEL, FALLBACK_GENERATION_MODEL}


def _run_model_pull(model):
    ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
    try:
        with requests.post(f"{ollama_url}/api/pull", json={'name': model},
                           stream=True, timeout=3600) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                info = json.loads(line)
                if info.get('error'):
                    raise RuntimeError(info['error'])
                with _pulls_lock:
                    state = _model_pulls[model]
                    state['detail'] = info.get('status', '')
                    if info.get('total'):
                        state['total'] = info['total']
                        state['completed'] = info.get('completed', 0)
        with _pulls_lock:
            _model_pulls[model]['status'] = 'done'
        logger.info("Pulled Ollama model %s", model)
    except Exception as e:
        logger.exception("Model pull failed for %s", model)
        with _pulls_lock:
            _model_pulls[model].update(status='error', detail=str(e))


@app.route('/api/models/pull', methods=['POST'])
def pull_model():
    """Start downloading a configured Ollama model (idempotent)."""
    data = request.get_json(silent=True) or {}
    model = data.get('model')
    if model not in _pullable_models():
        return jsonify({'success': False,
                        'error': f"'{model}' is not one of the configured models"}), 400
    with _pulls_lock:
        state = _model_pulls.get(model)
        if state and state['status'] == 'pulling':
            return jsonify({'success': True, 'status': 'pulling'}), 202
        _model_pulls[model] = {'status': 'pulling', 'completed': 0, 'total': 0, 'detail': ''}
    threading.Thread(target=_run_model_pull, args=(model,), daemon=True).start()
    return jsonify({'success': True, 'status': 'pulling'}), 202


@app.route('/api/models/pull/<model>', methods=['GET'])
def pull_model_status(model):
    with _pulls_lock:
        state = _model_pulls.get(model)
        if state is None:
            return jsonify({'success': False, 'error': 'No pull in progress'}), 404
        return jsonify({'success': True, **state})


@app.route('/api/analyze', methods=['POST'])
def analyze_transcript():
    """
    Start an analysis job. Two request shapes are accepted:

    JSON:       {"content": "...", "threshold": 0.85}
    Multipart:  files=<one or more .json/.txt/.csv/.zip files>,
                provider=..., threshold=...   (zips are unpacked server-side)

    Returns 202 with {"success": true, "job_id": "..."}; poll /api/jobs/<job_id>.
    """
    import zipfile

    if request.content_type and 'multipart/form-data' in request.content_type:
        provider = (request.form.get('provider')
                    or os.getenv('AI_PROVIDER', 'ollama')).strip().lower()
        threshold = request.form.get('threshold', 'auto')
        try:
            contents = _contents_from_upload()
        except (ValueError, zipfile.BadZipFile) as e:
            return jsonify({'success': False, 'error': str(e)}), 400
    else:
        data = request.get_json(silent=True)
        if not data or 'content' not in data:
            return jsonify({'success': False, 'error': 'Missing required field: content'}), 400
        if not isinstance(data['content'], str):
            return jsonify({'success': False, 'error': "'content' must be a string"}), 400
        contents = [data['content']]
        provider = (data.get('provider')
                    or os.getenv('AI_PROVIDER', 'ollama')).strip().lower()
        threshold = data.get('threshold', 'auto')

    if not any(c.strip() for c in contents):
        return jsonify({'success': False, 'error': 'Content cannot be empty'}), 400

    if provider not in VALID_PROVIDERS:
        return jsonify({
            'success': False,
            'error': f"Invalid provider '{provider}'. This tool is local-only: use 'ollama'."
        }), 400

    # 'auto' (the default): model-aware threshold that self-adjusts when
    # nothing groups; a number pins it exactly
    if threshold in (None, '', 'auto'):
        threshold = None
    else:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return jsonify({'success': False,
                            'error': "threshold must be 'auto' or a number between 0 and 1"}), 400
        if not 0.0 <= threshold <= 1.0:
            return jsonify({'success': False, 'error': 'threshold must be between 0 and 1'}), 400

    job_id = _start_job(contents, provider, threshold)
    return jsonify({'success': True, 'job_id': job_id}), 202


@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id):
    """Request cancellation of a queued or running job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({'success': False, 'error': 'Unknown job id'}), 404
        if job['status'] not in FINAL_STATUSES:
            job['cancelled'] = True
            # Persist the flag: without it, a restart before the worker
            # notices would resurrect the cancelled job and run it anyway
            _persist_job(job_id)
        return jsonify({'success': True, 'status': job['status']})


@app.route('/api/jobs/<job_id>', methods=['GET'])
def job_status(job_id):
    """Status, progress, and (when done) result of an analysis job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return _job_status_from_disk(job_id)
        payload = {
            'success': True,
            'status': job['status'],
            'progress': job['progress'],
        }
        if job['status'] == 'done':
            payload['data'] = job['result']
            payload['analysis_id'] = job['analysis_id']
        elif job['status'] == 'error':
            payload['error'] = job['error']
    return jsonify(payload)


def _job_status_from_disk(job_id):
    """Serve a job's final status from disk (e.g. after a server restart)."""
    meta_path = _job_meta_path(job_id)
    if not meta_path.is_file():
        return jsonify({'success': False, 'error': 'Unknown job id'}), 404
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return jsonify({'success': False, 'error': 'Unknown job id'}), 404

    payload = {
        'success': True,
        'status': meta.get('status'),
        'progress': {'stage': meta.get('status'), 'completed': 1, 'total': 1},
    }
    if meta.get('status') == 'done' and meta.get('analysis_id'):
        payload['analysis_id'] = meta['analysis_id']
        payload['data'] = _load_analysis(meta['analysis_id'])
    elif meta.get('status') == 'error':
        payload['error'] = meta.get('error')
    return jsonify(payload)


@app.route('/api/analyses', methods=['GET'])
def list_analyses():
    """List summaries of saved analyses, newest first."""
    return jsonify({'success': True, 'analyses': _list_analyses()})


@app.route('/api/analyses/latest', methods=['GET'])
def latest_analysis():
    """Full results of the most recent saved analysis."""
    summaries = _list_analyses()
    if not summaries:
        return jsonify({'success': False, 'error': 'No saved analyses yet'}), 404
    data = _load_analysis(summaries[0]['id'])
    return jsonify({'success': True, 'id': summaries[0]['id'], 'data': data})


@app.route('/api/analyses/latest/weekly', methods=['GET'])
def latest_weekly():
    """Week-in-Review stats for the most recent saved analysis."""
    summaries = _list_analyses()
    if not summaries:
        return jsonify({'success': False, 'error': 'No saved analyses yet'}), 404
    analysis_id = summaries[0]['id']
    data = _load_analysis(analysis_id)
    if data is None:
        return jsonify({'success': False, 'error': 'Analysis not found'}), 404
    weekly = compute_weekly_stats(data)
    if weekly is None:
        return jsonify({
            'success': False,
            'error': 'No parseable question dates in this analysis, so weekly '
                     'trends cannot be computed.'
        }), 422
    return jsonify({'success': True, 'id': analysis_id, 'data': weekly})


@app.route('/api/analyses/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Full results of a specific saved analysis."""
    data = _load_analysis(analysis_id)
    if data is None:
        return jsonify({'success': False, 'error': 'Analysis not found'}), 404
    return jsonify({'success': True, 'id': analysis_id, 'data': data})


@app.route('/api/analyses/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Delete a saved analysis."""
    path = (ANALYSES_DIR / f"{analysis_id}.json").resolve()
    if path.parent != ANALYSES_DIR.resolve() or not path.is_file():
        return jsonify({'success': False, 'error': 'Analysis not found'}), 404
    path.unlink()
    return jsonify({'success': True})


@app.route('/api/analyses/<analysis_id>/export', methods=['GET'])
def export_analysis(analysis_id):
    """Download a saved analysis as Markdown, CSV, or JSON."""
    data = _load_analysis(analysis_id)
    if data is None:
        return jsonify({'success': False, 'error': 'Analysis not found'}), 404

    fmt = request.args.get('format', 'json').lower()
    if fmt in ('md', 'markdown'):
        content, mimetype, ext = to_markdown(data), 'text/markdown', 'md'
    elif fmt == 'faq':
        from slack_question_analyzer.topic_bank import TopicBank
        bank_entries = TopicBank().entries
        published = {e['id']: e['faq_published'] for e in bank_entries
                     if e.get('id') and e.get('faq_published')}
        # Curated answers apply retroactively: exporting an analysis saved
        # BEFORE the answer was approved still uses the canonical wording.
        # The save date rides along for the staleness nudge.
        curated = {e['id']: {'answer': e['curated_answer'],
                             'updated': e.get('answer_updated')}
                   for e in bank_entries
                   if e.get('id') and e.get('curated_answer')}
        content = to_faq_markdown(data, published=published, curated=curated)
        mimetype, ext = 'text/markdown', 'md'
        analysis_id = f'{analysis_id}-faq-draft'
    elif fmt == 'csv':
        content, mimetype, ext = to_csv(data), 'text/csv', 'csv'
    elif fmt == 'json':
        content = json.dumps(data, indent=2, ensure_ascii=False)
        mimetype, ext = 'application/json', 'json'
    else:
        return jsonify({'success': False,
                        'error': f"Unknown format '{fmt}'. Use md, faq, csv, "
                                 'or json.'}), 400

    return Response(content, mimetype=mimetype, headers={
        'Content-Disposition': f'attachment; filename="analysis-{analysis_id}.{ext}"'
    })


# Per-analysis occurrence index for the history endpoint, keyed by file
# identity (mtime + size). Saved analyses are immutable once written, so
# after the first history click each file is parsed exactly once instead
# of every saved analysis being re-read from disk on every request.
_history_index = {}
_history_lock = threading.Lock()


def _analysis_occurrences(path):
    """(occurrences, newest_day) for one saved analysis: occurrences is
    {topic_id: [(normalized_text, date_str, answered), ...]}, newest_day the
    max parseable question date across the WHOLE analysis (any topic or
    ungrouped) — the corpus-time anchor for 'did asks stop after the FAQ'."""
    from slack_question_analyzer.weekly_stats import parse_question_date
    try:
        stat = path.stat()
    except OSError:
        return None
    key, ident = str(path), (stat.st_mtime_ns, stat.st_size)
    with _history_lock:
        cached = _history_index.get(key)
        if cached and cached[0] == ident:
            return cached[1], cached[2]
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    index = {}
    newest = None
    all_questions = list(data.get('ungrouped_questions') or [])
    for group in (data.get('groups') or []):
        questions = group.get('questions') or []
        all_questions.extend(questions)
        tid = group.get('topic_id')
        if not tid:
            continue
        occurrences = index.setdefault(tid, [])
        for q in questions:
            text = ' '.join((q.get('normalized_text')
                             or q.get('text') or '').split()).lower()
            if text:
                occurrences.append((text, q.get('date') or 'Unknown',
                                    q.get('answered') is True))
    for q in all_questions:
        day = parse_question_date(q.get('date'))
        if day and (newest is None or day > newest):
            newest = day
    with _history_lock:
        _history_index[key] = (ident, index, newest)
    return index, newest


@app.route('/api/topics/<topic_id>/history', methods=['GET'])
def topic_history(topic_id):
    """
    A topic's volume over time, ACROSS every saved analysis — the
    longitudinal question ('is this still growing? did my FAQ work?') that
    a single analysis can't answer. Occurrences are fingerprinted
    (normalized text + date), so overlapping uploads — the natural 'export
    the last 90 days monthly' workflow — never double-count a question.
    """
    import hashlib
    from slack_question_analyzer.weekly_stats import parse_question_date
    from slack_question_analyzer.topic_bank import TopicBank

    # Old analyses recorded ids that were later merged away — those ids
    # live on as merged_ids aliases, and their history belongs here too
    entry = TopicBank()._find(topic_id)
    wanted = {topic_id} | set((entry or {}).get('merged_ids') or [])

    # Staleness nudge input: the curated answer's save date (only when a
    # curated answer actually exists)
    answer_updated = ((entry or {}).get('answer_updated')
                      if (entry or {}).get('curated_answer') else None)
    updated_day = parse_question_date(answer_updated) if answer_updated else None
    newly_answered = 0

    seen = set()
    dated = []          # (date, text) unique occurrences
    undated = 0
    analyses_with_topic = 0
    data_end = None     # newest dated question ANYWHERE in the corpus —
    #                     a topic VANISHING from later uploads is the
    #                     strongest 'FAQ worked' signal, and only the whole
    #                     corpus can prove the data extends past the publish
    if ANALYSES_DIR.is_dir():
        for path in sorted(ANALYSES_DIR.glob('*.json')):
            parsed = _analysis_occurrences(path)
            if parsed is None:
                continue
            index, newest = parsed
            if newest and (data_end is None or newest > data_end):
                data_end = newest
            hit = False
            for tid in wanted:
                if tid not in index:
                    continue
                hit = True
                for text, date_str, answered in index[tid]:
                    fp = hashlib.sha1(
                        f'{text}|{date_str}'.encode('utf-8')).hexdigest()[:16]
                    if fp in seen:
                        continue
                    seen.add(fp)
                    day = parse_question_date(date_str)
                    if day is None:
                        undated += 1
                    else:
                        dated.append((day, text))
                        if answered and updated_day and day > updated_day:
                            newly_answered += 1
            if hit:
                analyses_with_topic += 1

    if not seen:
        return jsonify({'success': False,
                        'error': 'No saved analysis contains this topic'}), 404

    # Bucket by ISO week (Monday) — the same granularity Week in Review uses
    weeks = {}
    for day, _ in dated:
        monday = day.fromordinal(day.toordinal() - day.weekday())
        weeks[monday.isoformat()] = weeks.get(monday.isoformat(), 0) + 1
    series = [{'week': w, 'count': c} for w, c in sorted(weeks.items())]

    # Did the published FAQ work? Weekly ask rate before vs. after the
    # publish date, anchored to DATA time (the newest dated occurrence),
    # never the server clock — historical transcripts must judge fairly.
    faq_effect = None
    published = (entry or {}).get('faq_published')
    pub_day = parse_question_date(published) if published else None
    if pub_day is not None and dated:
        days = [d for d, _ in dated]
        end = max([data_end] + days) if data_end else max(days)
        days_after = (end - pub_day).days
        window = min(56, (pub_day - min(days)).days)  # up to 8 weeks baseline
        if days_after < 14:
            faq_effect = {'verdict': 'too_early',
                          'days_of_data_after': max(days_after, 0)}
        elif window < 14:
            faq_effect = {'verdict': 'no_baseline'}
        else:
            before_n = sum(1 for d in days
                           if d < pub_day and (pub_day - d).days <= window)
            after_n = sum(1 for d in days if d >= pub_day)
            before_rate = before_n / (window / 7)
            after_rate = after_n / (days_after / 7)
            if before_rate == 0:
                faq_effect = {'verdict': 'no_baseline'}
            else:
                change = (after_rate - before_rate) / before_rate
                faq_effect = {
                    'verdict': ('working' if change <= -0.5
                                else 'helping' if change <= -0.15
                                else 'not_working'),
                    'before_per_week': round(before_rate, 1),
                    'after_per_week': round(after_rate, 1),
                    'change_pct': round(change * 100),
                }

    return jsonify({
        'success': True,
        'topic_id': topic_id,
        'faq_published': (entry or {}).get('faq_published'),
        'faq_effect': faq_effect,
        # Curated answers can go stale: asks answered in threads AFTER the
        # approved wording was saved may contain fixes the doc lacks
        'answer_stale': ({'new_answered': newly_answered,
                          'answer_updated': answer_updated}
                         if newly_answered else None),
        'total_occurrences': len(seen),
        'undated': undated,
        'analyses_with_topic': analyses_with_topic,
        'series': series,
        'first_asked': series[0]['week'] if series else None,
        'last_asked': series[-1]['week'] if series else None,
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    from slack_question_analyzer import __version__
    return jsonify({
        'success': True,
        'version': __version__,
        'config': {
            'provider': os.getenv('AI_PROVIDER', 'ollama').strip().lower(),
            'threshold': (float(os.getenv('SIMILARITY_THRESHOLD'))
                          if os.getenv('SIMILARITY_THRESHOLD') else 'auto'),
            'ollama_url': os.getenv('OLLAMA_URL', 'http://localhost:11434'),
            'ollama_model': os.getenv('OLLAMA_MODEL', 'nomic-embed-text')
        }
    })


@app.errorhandler(413)
def payload_too_large(_e):
    return jsonify({'success': False,
                    'error': f'Transcript too large (limit: {MAX_CONTENT_MB}MB)'}), 413


@app.errorhandler(500)
def internal_error(_e):
    # Details are logged server-side; don't leak tracebacks to clients
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


def choose_port(host, preferred, attempts=5):
    """
    First free port starting at `preferred`. macOS reserves 5000 for AirPlay
    and other apps collide too — falling back beats failing to start.

    The probe must NOT set SO_REUSEADDR: on Windows that allows binding a
    port another process is actively listening on, which lets a forgotten
    old server instance silently keep receiving all the traffic.
    """
    import socket
    for offset in range(attempts):
        port = preferred + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):  # Windows: strict
                    probe.setsockopt(socket.SOL_SOCKET,
                                     socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind((host, port))
            if offset:
                logger.warning("Port %d is in use; using %d instead", preferred, port)
            return port
        except OSError:
            continue
    return preferred  # let the server raise the real error


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    host = os.getenv('API_HOST', '127.0.0.1')
    port = choose_port(host, int(os.getenv('API_PORT', '5000')))
    debug = os.getenv('FLASK_DEBUG', '').lower() in ('1', 'true', 'on')

    # Re-queue any jobs that were interrupted by the last shutdown. Under the
    # debug reloader this block runs in BOTH the watcher and the worker
    # process — only recover in the one that serves requests, or interrupted
    # jobs would run twice concurrently.
    if not debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        recover_interrupted_jobs()

    display_host = 'localhost' if host in ('0.0.0.0', '127.0.0.1') else host
    dashboard_url = f"http://{display_host}:{port}/"

    print("=" * 60)
    print("Slack Question Analyzer")
    print("=" * 60)
    print(f"Dashboard:    {dashboard_url}")
    print(f"Health check: {dashboard_url}api/health")
    print(f"API:          POST {dashboard_url}api/analyze")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")

    # Open the dashboard automatically (NO_BROWSER=1 disables; set in Docker)
    if os.getenv('NO_BROWSER', '').lower() not in ('1', 'true', 'on') and not debug:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(dashboard_url)).start()

    app.run(debug=debug, host=host, port=port, threaded=True)
