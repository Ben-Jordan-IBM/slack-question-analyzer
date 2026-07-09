"""Tests for the LLM prompting layer and its integration into the pipeline."""

import json

import numpy as np
import pytest

from slack_question_analyzer.group_labeler import GroupLabeler
from slack_question_analyzer.analyzer import QuestionAnalyzer


@pytest.fixture(autouse=True)
def isolated_llm_cache(tmp_path, monkeypatch):
    """Keep each test's LLM result cache in its own temp directory."""
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path / 'llm_cache'))


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def chat_response(content):
    """Shape of an Ollama /api/chat response."""
    return FakeResponse({'message': {'role': 'assistant', 'content': content}})


def patch_chat(monkeypatch, replies, captured=None):
    """Make requests.post return the given reply contents in order."""
    replies = list(replies)

    def fake_post(url, json=None, timeout=None):
        if captured is not None:
            captured.append({'url': url, 'body': json})
        return chat_response(replies.pop(0))

    monkeypatch.setattr('slack_question_analyzer.group_labeler.requests.post', fake_post)


# ---- Output parsing & validation ----

def test_parse_json_tolerates_prose():
    assert GroupLabeler._parse_json('Sure! {"topic": "VPN Access", "summary": "x"} ok')['topic'] == 'VPN Access'
    assert GroupLabeler._parse_json('not json') is None


def test_validate_label_rejects_generic_topics():
    assert GroupLabeler._validate_label({'topic': 'General Questions', 'summary': 's'}) is not None
    assert GroupLabeler._validate_label({'topic': 'Help', 'summary': 's'}) is not None
    assert GroupLabeler._validate_label({'topic': '', 'summary': 's'}) is not None
    assert GroupLabeler._validate_label({'topic': 'Antivirus Scanning', 'summary': ''}) is not None
    assert GroupLabeler._validate_label({'topic': 'Antivirus Scanning', 'summary': 's'}) is None


# ---- label_group ----

def test_label_group_uses_chat_with_schema_and_few_shot(monkeypatch):
    captured = []
    patch_chat(monkeypatch,
               ['{"topic": "Password Reset", "summary": "How to reset passwords."}'],
               captured)

    label = GroupLabeler('ollama').label_group(
        ['How do I reset my password?'], keywords=['password', 'reset'])

    assert label == {'topic': 'Password Reset', 'summary': 'How to reset passwords.'}
    body = captured[0]['body']
    assert captured[0]['url'].endswith('/api/chat')
    assert body['format']['type'] == 'object'  # full JSON schema, not just "json"
    assert body['options']['temperature'] == 0
    assert body['keep_alive'] == '30m'
    user_msg = body['messages'][1]['content']
    assert 'Example:' in user_msg                 # few-shot examples present
    assert 'Keywords: password, reset' in user_msg
    assert 'Never use vague topics' in body['messages'][0]['content']


def test_extraction_prompt_has_few_shot_and_long_excerpts(monkeypatch):
    captured = []
    patch_chat(monkeypatch, [json.dumps({'questions': []})], captured)
    long_message = 'x' * 500 + ' how do I fix this?'
    GroupLabeler('ollama').extract_questions([long_message])

    user_msg = captured[0]['body']['messages'][1]['content']
    assert 'Example messages:' in user_msg          # few-shot present
    assert 'bulk-disable' in user_msg               # the worked example
    assert 'x' * 500 in user_msg                    # 600-char excerpts, not 300
    assert captured[0]['body']['options']['num_predict'] == 800


def test_domain_context_injected_into_prompts(monkeypatch):
    monkeypatch.setenv('DOMAIN_CONTEXT', 'a webMethods MFT support channel')
    captured = []
    patch_chat(monkeypatch,
               ['{"topic": "Virus Scanning", "summary": "s"}'], captured)
    GroupLabeler('ollama').label_group(['How do I scan files?'])

    system = captured[0]['body']['messages'][0]['content']
    assert 'webMethods MFT support channel' in system


def test_label_group_retries_generic_topic_with_feedback(monkeypatch):
    captured = []
    patch_chat(monkeypatch,
               ['{"topic": "General Questions", "summary": "stuff"}',
                '{"topic": "Webhook Configuration", "summary": "How to configure webhooks."}'],
               captured)

    label = GroupLabeler('ollama').label_group(['How do I configure the webhook?'])

    assert label['topic'] == 'Webhook Configuration'
    assert len(captured) == 2
    # Retry conversation includes the bad answer and corrective feedback
    retry_messages = captured[1]['body']['messages']
    assert retry_messages[2]['role'] == 'assistant'
    assert 'too generic' in retry_messages[3]['content']


def test_label_group_returns_none_when_retry_also_fails(monkeypatch):
    patch_chat(monkeypatch, ['{"topic": "Help", "summary": "x"}',
                             '{"topic": "Misc", "summary": "x"}'])
    assert GroupLabeler('ollama').label_group(['Anything?']) is None


def test_label_results_are_cached_on_disk(monkeypatch):
    """Identical prompts are served from the cache, even across instances."""
    captured = []
    patch_chat(monkeypatch,
               ['{"topic": "Password Reset", "summary": "How to reset passwords."}'],
               captured)

    labeler = GroupLabeler('ollama')
    first = labeler.label_group(['How do I reset my password?'])
    # Saves batch during a run (min_dirty); the analyzer flushes at the end
    labeler.flush_cache()
    # A brand-new instance with the same prompt must not hit the model again
    second = GroupLabeler('ollama').label_group(['How do I reset my password?'])

    assert first == second
    assert len(captured) == 1  # one real chat call total


def test_label_group_returns_none_on_connection_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise ConnectionError('refused')

    monkeypatch.setattr('slack_question_analyzer.group_labeler.requests.post', boom)
    assert GroupLabeler('ollama').label_group(['Anything?']) is None


# ---- Other capabilities ----

def test_verify_same_topic(monkeypatch):
    captured = []
    patch_chat(monkeypatch, ['{"same_topic": true}'], captured)
    result = GroupLabeler('ollama').verify_same_topic(
        ['How do I reset my password?'], ['Password reset steps?'])
    assert result is True
    assert 'Group A:' in captured[0]['body']['messages'][1]['content']

    patch_chat(monkeypatch, ['{"same_topic": false}'])
    assert GroupLabeler('ollama').verify_same_topic(['a?'], ['b?']) is False

    # Different questions (a fresh prompt) so the cached verdict above isn't reused
    patch_chat(monkeypatch, ['{"same_topic": "maybe"}'])
    assert GroupLabeler('ollama').verify_same_topic(['c?'], ['d?']) is None


def test_summarize_analysis(monkeypatch):
    patch_chat(monkeypatch, ['{"summary": "Antivirus scanning dominates with 12 questions."}'])
    groups = [{'topic': 'Antivirus Scanning', 'count': 12,
               'representative_question': 'How do I configure scanning?'}]
    summary = GroupLabeler('ollama').summarize_analysis(groups, 30)
    assert 'Antivirus' in summary


def test_detect_questions_filters_invalid_indices(monkeypatch):
    base = GroupLabeler.MESSAGE_INDEX_BASE
    patch_chat(monkeypatch, [json.dumps({'questions': [
        {'index': base + 0, 'question': 'How do I fix the webhook?'},
        {'index': base + 99, 'question': 'out of range'},
        {'index': 2, 'question': 'example-range indices must be rejected'},
        {'index': base + 1, 'question': ''},
    ]})])
    found = GroupLabeler('ollama').detect_questions(
        ["can't get the webhook to work, stuck all day", 'deploy went fine'])
    assert found == [{'index': 0, 'question': 'How do I fix the webhook?'}]


def test_is_answered(monkeypatch):
    patch_chat(monkeypatch, ['{"verdict": "answered"}'])
    assert GroupLabeler('ollama').is_answered('How do I reset?', ['Go to settings > reset.']) is True

    patch_chat(monkeypatch, ['{"verdict": "unanswered"}'])
    assert GroupLabeler('ollama').is_answered('How do I reset?', ['Let me check.']) is False

    # 'unknown' is the model's honest abstain — treated as no verdict
    patch_chat(monkeypatch, ['{"verdict": "unknown"}'])
    assert GroupLabeler('ollama').is_answered('How do I reset?', ['hmm?']) is None


def test_available_checks_model_is_pulled(monkeypatch):
    monkeypatch.setattr('slack_question_analyzer.group_labeler.requests.get',
                        lambda *a, **k: FakeResponse({'models': [{'name': 'llama3.2:latest'}]}))
    labeler = GroupLabeler('ollama')
    labeler.model = 'llama3.2'
    assert labeler.available() is True

    other = GroupLabeler('ollama')
    other.model = 'mistral'
    assert other.available() is False


def test_available_false_when_ollama_unreachable(monkeypatch):
    import requests as requests_module

    def refuse(*args, **kwargs):
        raise requests_module.ConnectionError('refused')

    monkeypatch.setattr('slack_question_analyzer.group_labeler.requests.get', refuse)
    assert GroupLabeler('ollama').available() is False


# ---- Pipeline integration ----

SAMPLE_CONTENT = (
    "2024-01-05\nHow do I reset my password?\n"
    "-----------------------------------------------------------\n"
    "2024-01-08\nHow can I get my password changed?\n"
    "-----------------------------------------------------------\n"
    "2024-01-09\nWhat is the deploy schedule for production releases?\n"
)

VECTORS = {
    'how do i reset my password?': [1.0, 0.0, 0.0],
    'how can i get my password changed?': [0.99, 0.05, 0.0],
    'what is the deploy schedule for production releases?': [0.0, 1.0, 0.0],
}


def make_analyzer(monkeypatch, vectors=VECTORS, **kwargs):
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False, **kwargs)
    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: np.array([vectors[t] for t in texts]))
    return analyzer


def stub_llm(monkeypatch, analyzer, **methods):
    monkeypatch.setattr(analyzer.labeler, 'available', lambda: True)
    # Default: LLM-first extraction "fails" so the regex fallback runs,
    # keeping older tests' question sets unchanged
    methods.setdefault('extract_questions', lambda texts, thorough=False: None)
    methods.setdefault('consolidate_same_ask', lambda msg, texts: None)
    methods.setdefault('confirm_feature_request', lambda text, context='': True)
    for name, fn in methods.items():
        monkeypatch.setattr(analyzer.labeler, name, fn)


def test_llm_labels_and_summary_applied(monkeypatch):
    analyzer = make_analyzer(monkeypatch, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: {'topic': 'Password Reset',
                                                       'summary': 'People ask how to reset passwords.'},
             verify_same_topic=lambda a, b: None,
             detect_questions=lambda texts: [],
             summarize_analysis=lambda groups, total, themes=None: 'Password resets dominate.')

    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    group = results['groups'][0]
    assert group['topic'] == 'Password Reset'
    assert group['summary'] == 'People ask how to reset passwords.'
    assert results['executive_summary'] == 'Password resets dominate.'


def test_keyword_fallback_when_labels_disabled(monkeypatch):
    analyzer = make_analyzer(monkeypatch, label_groups=False)
    assert analyzer.labeler is None

    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    group = results['groups'][0]
    assert group['topic']  # keyword-derived
    assert 'password' in group['topic'].lower()
    assert group['summary'] is None
    assert results['executive_summary'] is None


def test_keyword_fallback_when_llm_fails(monkeypatch):
    analyzer = make_analyzer(monkeypatch, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             detect_questions=lambda texts: [],
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    assert results['groups'][0]['topic']  # fell back to keywords, not missing


def test_llm_detection_adds_missed_questions(monkeypatch):
    monkeypatch.setenv('LLM_EXTRACTION', 'on')  # regex-first + missed-pass mode
    content = (
        "2024-01-05\nHow do I reset my password?\n"
        "-----------------------------------------------------------\n"
        "2024-01-06\nstuck all day, the webhook just times out on every call\n"
    )
    vectors = {
        'how do i reset my password?': [1.0, 0.0],
        'how do i fix webhook timeouts?': [0.0, 1.0],
    }
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             detect_questions=lambda texts: [{'index': 0, 'question': 'How do I fix webhook timeouts?'}],
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(content)
    assert results['total_questions'] == 2
    texts = [q['text'] for q in results['ungrouped_questions']]
    assert 'How do I fix webhook timeouts?' in texts


def test_full_llm_extraction_mode(monkeypatch):
    """LLM_EXTRACTION=full: the LLM extracts and cleans every question."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    content = (
        "2024-01-05\nHi Team, Can I check if the agent comes pre-installed?\n"
        "-----------------------------------------------------------\n"
        "2024-01-06\nThe deploy finished fine.\n"
    )
    vectors = {'does the metering agent come pre-installed?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: [
                 {'index': 0, 'question': 'Does the Metering Agent come pre-installed?'}
             ] if 'pre-installed' in texts[0] else [],
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(content)
    assert results['total_questions'] == 1
    question = results['ungrouped_questions'][0]
    assert question['text'] == 'Does the Metering Agent come pre-installed?'
    assert question['llm_extracted'] is True


def test_full_extraction_falls_back_to_regex_on_llm_failure(monkeypatch):
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    analyzer = make_analyzer(monkeypatch, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: None,  # LLM call failed
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    assert results['total_questions'] == 3  # regex fallback kept everything


def test_auto_mode_defaults_to_llm_extraction_for_small_transcripts(monkeypatch):
    """Without any LLM_EXTRACTION setting, small transcripts get LLM-first."""
    monkeypatch.delenv('LLM_EXTRACTION', raising=False)
    vectors = {'does the agent come pre-installed?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: [
                 {'index': 0, 'question': 'Does the agent come pre-installed?'}],
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-01-05\nHi Team, can I check if the agent comes pre-installed?\n")
    assert results['total_questions'] == 1
    assert results['ungrouped_questions'][0]['llm_extracted'] is True


def test_answer_detection_counts_resolved_threads(monkeypatch):
    content = json.dumps([
        {'text': 'How do I reset my password?', 'ts': '1704412800.0'},
        {'text': 'Go to settings > security > reset.', 'ts': '1704412900.0',
         'thread_ts': '1704412800.0'},
        {'text': 'What is the deploy schedule for production releases?', 'ts': '1704499200.0'},
    ])
    vectors = {
        'how do i reset my password?': [1.0, 0.0],
        'what is the deploy schedule for production releases?': [0.0, 1.0],
    }
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             detect_questions=lambda texts: [],
             is_answered=lambda question, replies, context='': True,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(content)
    assert results['answered_questions'] == 1


# ---- Group audit (outlier eviction) ----

def test_audit_group_returns_zero_based_outliers(monkeypatch):
    labeler = GroupLabeler(provider='ollama')
    patch_chat(monkeypatch, [json.dumps({'outliers': [1]})])
    assert labeler.audit_group(['How do I install the metering agent?',
                                'How do I set up monitoring alerts?']) == [0]


def test_audit_group_clean_and_uncertain(monkeypatch):
    labeler = GroupLabeler(provider='ollama')
    patch_chat(monkeypatch, [json.dumps({'outliers': []})])
    assert labeler.audit_group(['a?', 'b?']) == []

    def boom(*a, **k):
        raise RuntimeError('ollama down')
    monkeypatch.setattr('slack_question_analyzer.group_labeler.requests.post', boom)
    # different questions: the cached verdict for ('a?', 'b?') must not apply
    assert labeler.audit_group(['c?', 'd?']) is None


def test_audit_group_rejects_evict_everything_verdict(monkeypatch):
    """'All questions are outliers' is not a meaningful audit."""
    labeler = GroupLabeler(provider='ollama')
    patch_chat(monkeypatch, [json.dumps({'outliers': [1, 2]})])
    assert labeler.audit_group(['a?', 'b?']) is None


def test_audit_group_ignores_out_of_range_indices(monkeypatch):
    labeler = GroupLabeler(provider='ollama')
    patch_chat(monkeypatch, [json.dumps({'outliers': [2, 9, 0]})])
    assert labeler.audit_group(['a?', 'b?', 'c?']) == [1]


# ---- Timeout & warm-up ----

def test_llm_timeout_configurable(monkeypatch):
    monkeypatch.setenv('LLM_TIMEOUT', '300')
    assert GroupLabeler(provider='ollama').timeout == 300
    monkeypatch.delenv('LLM_TIMEOUT')
    assert GroupLabeler(provider='ollama').timeout == 180


def test_warm_up_loads_model_without_generating(monkeypatch):
    labeler = GroupLabeler(provider='ollama')
    captured = []
    patch_chat(monkeypatch, ['ignored'], captured)
    labeler.warm_up()
    assert captured[0]['body']['messages'] == []
    assert captured[0]['body']['model'] == labeler.model


def test_warm_up_swallows_connection_errors(monkeypatch):
    import requests as _requests
    labeler = GroupLabeler(provider='ollama')

    def boom(*a, **k):
        raise _requests.ConnectionError('no ollama')
    monkeypatch.setattr('slack_question_analyzer.group_labeler.requests.post', boom)
    labeler.warm_up()  # must not raise


# ---- Fast/quality model split ----

def tags_response(monkeypatch, names):
    monkeypatch.setattr(
        'slack_question_analyzer.group_labeler.requests.get',
        lambda *a, **k: FakeResponse({'models': [{'name': n} for n in names]}))


def test_fast_model_split_when_both_downloaded(monkeypatch):
    monkeypatch.setenv('OLLAMA_GENERATION_MODEL', 'llama3.1:8b')
    tags_response(monkeypatch, ['llama3.1:8b', 'llama3.2:latest'])
    labeler = GroupLabeler(provider='ollama')
    assert labeler.available() is True
    assert labeler.model == 'llama3.1:8b'        # judgment calls
    assert labeler.fast_model == 'llama3.2'      # token-heavy extraction


def test_no_split_when_small_model_missing(monkeypatch):
    monkeypatch.setenv('OLLAMA_GENERATION_MODEL', 'llama3.1:8b')
    tags_response(monkeypatch, ['llama3.1:8b'])
    labeler = GroupLabeler(provider='ollama')
    assert labeler.available() is True
    assert labeler.fast_model == 'llama3.1:8b'


def test_fast_model_env_override(monkeypatch):
    monkeypatch.setenv('OLLAMA_GENERATION_MODEL', 'llama3.1:8b')
    monkeypatch.setenv('OLLAMA_FAST_MODEL', 'qwen2.5:3b')
    tags_response(monkeypatch, ['llama3.1:8b', 'llama3.2:latest', 'qwen2.5:3b'])
    labeler = GroupLabeler(provider='ollama')
    labeler.available()
    assert labeler.fast_model == 'qwen2.5:3b'


def test_extraction_uses_fast_model(monkeypatch):
    monkeypatch.setenv('OLLAMA_GENERATION_MODEL', 'llama3.1:8b')
    labeler = GroupLabeler(provider='ollama')
    labeler.fast_model = 'llama3.2'
    captured = []
    patch_chat(monkeypatch, [json.dumps({'questions': []})], captured)
    assert labeler.extract_questions(['How do I reset my password?']) == []
    assert captured[0]['body']['model'] == 'llama3.2'


def test_judgment_calls_use_quality_model(monkeypatch):
    monkeypatch.setenv('OLLAMA_GENERATION_MODEL', 'llama3.1:8b')
    labeler = GroupLabeler(provider='ollama')
    labeler.fast_model = 'llama3.2'
    captured = []
    patch_chat(monkeypatch, [json.dumps({'outliers': []})], captured)
    labeler.audit_group(['a?', 'b?'])
    assert captured[0]['body']['model'] == 'llama3.1:8b'


def test_warm_up_loads_both_models_when_split(monkeypatch):
    monkeypatch.setenv('OLLAMA_GENERATION_MODEL', 'llama3.1:8b')
    tags_response(monkeypatch, ['llama3.1:8b', 'llama3.2:latest'])

    class InlineThread:
        def __init__(self, target=None, args=(), daemon=None):
            self._target, self._args = target, args
        def start(self):
            self._target(*self._args)

    monkeypatch.setattr('slack_question_analyzer.group_labeler.threading.Thread',
                        InlineThread)
    labeler = GroupLabeler(provider='ollama')
    captured = []
    patch_chat(monkeypatch, ['x', 'y'], captured)
    labeler.warm_up()
    loaded = [c['body']['model'] for c in captured]
    assert loaded == ['llama3.2', 'llama3.1:8b']  # fast first, quality after
    assert all(c['body']['messages'] == [] for c in captured)


# ---- Extraction safety net ----

def test_safety_net_recovers_batch_the_fast_model_skipped(monkeypatch):
    """A fast model wrongly answering 'no questions here' must not silently
    lose a conversation: the quality model gets a second look."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'how do i configure iwhi monitoring?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        if thorough:
            return [{'index': 0, 'question': 'How do I configure IWHI monitoring?'}]
        return []  # fast model: "no questions here" (wrong)

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=extract,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-01-05\nHow do I configure IWHI monitoring?\n")
    assert results['total_questions'] == 1
    assert results['ungrouped_questions'][0]['text'] == 'How do I configure IWHI monitoring?'


def test_safety_net_rechecks_wordy_messages_with_no_regex_questions(monkeypatch):
    """Eval round 2: implicit help requests ('been stuck all morning, UI
    spins on login') and relayed wishes have NO regex-visible question, so
    the fewer-than-regex check never fired for them — three feedback
    messages and one implicit ask vanished. Any wordy message that produced
    zero asks now gets the quality-model second look."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'why does the web ui spin forever on login for one user?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    message = ('Been stuck on this all morning. For one customer the web UI '
               'just spins on login and never loads. Out of ideas.')

    def extract(texts, thorough=False):
        if thorough and any('spins on login' in t for t in texts):
            return [{'index': 0, 'question':
                     'Why does the web UI spin forever on login for one user?',
                     'type': 'troubleshooting'}]
        return []  # fast model: "no questions here" (wrong)

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=extract,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content('2024-01-05\n' + message + '\n')
    assert results['total_questions'] == 1
    assert 'login' in results['ungrouped_questions'][0]['text'].lower()


def test_safety_net_skips_terse_no_question_messages(monkeypatch):
    monkeypatch.setenv('EXTRACT_QUALITY_MAX', '0')  # isolate the recheck path
    """The zero-question recheck only fires for wordy prose — a short
    status line ('Deployed the fix, all green.') stays cheap: no thorough
    call, no questions."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    analyzer = make_analyzer(monkeypatch, vectors={}, label_groups=True)
    thorough_calls = []

    def extract(texts, thorough=False):
        if thorough:
            thorough_calls.append(texts)
        return []

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=extract,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content('2024-01-05\nDeployed the fix, all green.\n')
    assert results['total_questions'] == 0
    assert thorough_calls == []


def test_safety_net_falls_back_to_regex_when_both_models_skip(monkeypatch):
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'how do i configure iwhi monitoring?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: [],
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-01-05\nHow do I configure IWHI monitoring?\n")
    assert results['total_questions'] == 1  # regex version kept


# ---- Theme funnel ----

def test_assign_themes_maps_one_based_indices(monkeypatch):
    labeler = GroupLabeler(provider='ollama')
    patch_chat(monkeypatch, [json.dumps({'themes': [
        {'name': 'Security', 'items': [1, 3]},
        {'name': 'Transfers', 'items': [2]},
        {'name': 'Bogus', 'items': [99]},
    ]})])
    assigned = labeler.assign_themes(['Azure tokens', 'REST transfer triggers',
                                      'Certificate vault'])
    assert assigned == ['Security', 'Transfers', 'Security']


def test_assign_themes_returns_none_on_failure(monkeypatch):
    labeler = GroupLabeler(provider='ollama')

    def boom(*a, **k):
        raise RuntimeError('down')
    monkeypatch.setattr('slack_question_analyzer.group_labeler.requests.post', boom)
    assert labeler.assign_themes(['a', 'b']) is None


def test_analyzer_assigns_themes_to_groups_and_uniques(monkeypatch):
    analyzer = make_analyzer(monkeypatch, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             assign_themes=lambda items: ['Security', 'Security', 'Monitoring',
                                          'Monitoring', 'Security'])
    groups = [{'topic': 'Azure Tokens', 'count': 3, 'representative_question': 'x'},
              {'topic': 'Cert Vault', 'count': 2, 'representative_question': 'y'}]
    uniques = [{'text': 'IWHI monitoring examples?'},
               {'text': 'Metering agent pre-installed?'},
               {'text': 'Password reset?'}]

    themes = analyzer._assign_themes(groups, uniques)
    assert themes == [{'name': 'Security', 'count': 6},
                      {'name': 'Monitoring', 'count': 2}]
    assert groups[0]['theme'] == 'Security'
    assert uniques[0]['theme'] == 'Monitoring'


def test_themes_skipped_for_tiny_corpora(monkeypatch):
    analyzer = make_analyzer(monkeypatch, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             assign_themes=lambda items: ['A', 'B'])
    assert analyzer._assign_themes(
        [{'topic': 'T', 'count': 2, 'representative_question': 'x'}],
        [{'text': 'q?'}]) is None


def test_safety_net_drops_duplicate_when_fast_model_misattributes(monkeypatch):
    """Field regression: the fast model extracted the question but credited
    the WRONG message, so the true source looked skipped, got re-extracted,
    and the duplicate became a phantom 'asked 2x' group."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'how do i configure iwhi monitoring?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        # Both passes return the same question; the fast pass credits the
        # wrong message (index 1, the statement)
        return [{'index': 1 if not thorough else 0,
                 'question': 'How do I configure IWHI monitoring?'}]

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=extract,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-01-05\nHow do I configure IWHI monitoring?\n"
        "-----------------------------------------------------------\n"
        "2024-01-06\nDeploy went fine, all green here.\n")
    assert results['total_questions'] == 1  # not a phantom 2x group
    assert results['total_groups'] == 0


# ---- Prompt-pack v2 abstain paths ----

def test_label_group_abstains_with_needs_review(monkeypatch):
    """A mixed group gets NEEDS_REVIEW, not a vague papered-over label."""
    patch_chat(monkeypatch, ['{"topic": "NEEDS_REVIEW", "summary": ""}'])
    assert GroupLabeler('ollama').label_group(
        ['Metering counts?', 'SFTP keys?', 'UI crash?']) is None


def test_summary_abstains_with_needs_review(monkeypatch):
    patch_chat(monkeypatch, ['{"summary": "NEEDS_REVIEW"}'])
    groups = [{'topic': 'A', 'count': 2, 'representative_question': 'x'}]
    assert GroupLabeler('ollama').summarize_analysis(groups, 5) is None


def test_feature_requests_leave_the_support_funnel(monkeypatch):
    """Questions typed feature-request are diverted to results['feature_requests']
    and excluded from grouping, uniques, and total_questions."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'how do i reset my password?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        out = []
        for i, t in enumerate(texts):
            if 'reset' in t:
                out.append({'index': i, 'question': 'How do I reset my password?',
                            'type': 'how-to'})
            if 'graphical' in t:
                out.append({'index': i, 'question': 'Can the Action log get a graphical view?',
                            'type': 'feature-request'})
        return out

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=extract,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-01-05\nHow do I reset my password?\n"
        "-----------------------------------------------------------\n"
        "2024-01-06\nIt would be great to have a graphical Action log view\n")
    assert results['total_questions'] == 1  # support questions only
    assert len(results['feature_requests']) == 1
    assert 'graphical' in results['feature_requests'][0]['text']
    assert all('graphical' not in q['text'] for q in results['ungrouped_questions'])


# ---- Source-support invariant (misattribution guard) ----

def test_misattributed_extraction_is_reassigned_to_true_source(monkeypatch):
    # Pin the fast-model primary pass: this scenario IS fast-model
    # misattribution (small transcripts otherwise use the quality model)
    monkeypatch.setenv('EXTRACT_QUALITY_MAX', '0')
    """Ground-truth audit regression: the fast model stamped the custom-error
    question onto a metering message's index. The question inherited the
    wrong date, the metering question vanished, and the duplicate became a
    phantom 'asked 2x'. Every extraction must be supported by its claimed
    source; unsupported ones are reassigned to the message that contains
    them."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {
        'can we get a custom error code that the script can return?': [1.0, 0.0],
        'how can customers estimate their mft transactions?': [0.0, 1.0],
    }
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        if thorough:
            # quality model recovers the real metering question
            return [{'index': i, 'question':
                     'How can customers estimate their mft transactions?'}
                    for i, t in enumerate(texts) if 'estimate' in t]
        # fast model: extracts the custom-error question TWICE, once
        # credited to the metering message (index 1 = wrong)
        return [
            {'index': 0, 'question': 'Can we get a custom error code that the script can return?'},
            {'index': 1, 'question': 'Can we get a custom error code that the script can return?'},
        ]

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=extract,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-06-02\nCan we get a custom error code that the script can return?\n"
        "-----------------------------------------------------------\n"
        "2024-05-30\nHow can customers estimate their mft transactions?\n")

    texts = [q['text'] for q in results['ungrouped_questions']]
    # No phantom 2x group; both real questions present exactly once
    assert results['total_groups'] == 0
    assert results['total_questions'] == 2
    assert texts.count('Can we get a custom error code that the script can return?') == 1
    assert any('estimate' in t for t in texts)  # metering question recovered
    # The custom-error question kept its TRUE date, not the metering one's
    custom = next(q for q in results['ungrouped_questions'] if 'custom' in q['text'])
    assert custom['date'] == '2024-06-02'


def test_unsupported_extraction_dropped_when_no_source_exists(monkeypatch):
    """A hallucinated question no batch message contains is dropped, with
    the safety net recovering the real content by regex."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'how do i reset my password?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        if thorough:
            return []
        return [{'index': 0, 'question':
                 'What is the capital of France in webMethods deployments?'}]

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=extract,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-01-05\nHow do I reset my password?\n")
    texts = [q['text'] for q in results['ungrouped_questions']]
    assert all('France' not in t for t in texts)   # hallucination dropped
    assert any('password' in t.lower() for t in texts)  # regex recovered the real one


def test_dropped_unsupported_recovery_still_falls_back_to_regex(monkeypatch):
    """Safety-net bug: a hit dropped as unsupported still marked its message
    'recovered', skipping the regex fallback — the question vanished
    silently. The message must fall through to regex when its only hit dies."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'how do i configure iwhi monitoring?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        if thorough:
            # quality model hallucinates something no message contains
            return [{'index': 0, 'question': 'How do I bake sourdough bread?'}]
        return []  # fast model misses everything

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=extract,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-01-05\nHow do I configure IWHI monitoring?\n")
    texts = [q['text'] for q in results['ungrouped_questions']]
    assert all('sourdough' not in t for t in texts)
    assert any('IWHI' in t for t in texts)  # regex fallback recovered it


# ---- Same-ask consolidation & feedback confirmation (fresh-transcript round) ----

def test_rephrased_same_ask_consolidated(monkeypatch):
    """Fresh-transcript regression (DST scheduler): one ask phrased across
    two sentences became two questions. Lexical overlap can't see it (2
    shared words); the quality model picks the distinct asks."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'could the scheduler be running in the wrong timezone after dst?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        return [
            {'index': 0, 'question': 'Could the scheduler be running in the wrong timezone after DST?'},
            {'index': 0, 'question': 'Is the transfers-stopping-after-DST issue timezone-related?'},
        ]

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: True,  # judge 2 agrees: same ask
             extract_questions=extract,
             consolidate_same_ask=lambda msg, texts: [1],  # one distinct ask
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-06-04\nScheduled transfers stop right after DST. Could the "
        "scheduler be running in the wrong timezone after DST changes?\n")
    assert results['total_questions'] == 1
    assert 'wrong timezone' in results['ungrouped_questions'][0]['text']


def test_consolidation_abstain_keeps_everything(monkeypatch):
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {
        'can we trigger transfers via rest?': [1.0, 0.0],
        'is there a way to bulk-disable actions?': [0.0, 1.0],
    }
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: [
                 {'index': 0, 'question': 'Can we trigger transfers via REST?'},
                 {'index': 0, 'question': 'Is there a way to bulk-disable actions?'}],
             consolidate_same_ask=lambda msg, texts: None,  # abstain
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-06-04\nCan we trigger transfers via REST instead of the "
        "scheduler? Also is there a way to bulk-disable actions?\n")
    assert results['total_questions'] == 2


def test_unconfirmed_feature_request_stays_in_support(monkeypatch):
    """Fresh-transcript regression: the feedback lane became a dumping
    ground for misroutes. The 3B tag alone no longer diverts — the quality
    model must confirm, else the question stays in support."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'is there a cleaner pattern for routing files by prefix?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: [
                 {'index': 0, 'question': 'Is there a cleaner pattern for routing files by prefix?',
                  'type': 'feature-request'}],  # mis-tagged by the 3B
             confirm_feature_request=lambda text, context='': False,  # 8B: it's support
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-06-03\nIs there a cleaner pattern for routing files by prefix?\n")
    assert results['feature_requests'] == []
    assert results['total_questions'] == 1


def test_consolidate_same_ask_parses_and_counts(monkeypatch):
    labeler = GroupLabeler('ollama')
    patch_chat(monkeypatch, ['{"keep": [1]}'])
    keep = labeler.consolidate_same_ask('msg', ['a?', 'b rephrased?'])
    assert keep == [1]

    patch_chat(monkeypatch, ['{"keep": []}'])
    assert labeler.consolidate_same_ask('msg2', ['c?', 'd?']) is None  # not meaningful


def test_confirm_feature_request_verdicts(monkeypatch):
    labeler = GroupLabeler('ollama')
    patch_chat(monkeypatch, ['{"feature_request": true}'])
    assert labeler.confirm_feature_request('Can you add dark mode?') is True
    patch_chat(monkeypatch, ['{"feature_request": false}'])
    assert labeler.confirm_feature_request('How do I enable dark mode?') is False


def test_confirm_feature_request_receives_original_message(monkeypatch):
    """Intent lives in the asker's words: the original message (with its
    wish-phrasing) is part of the confirmation prompt."""
    labeler = GroupLabeler('ollama')
    captured = []
    patch_chat(monkeypatch, ['{"feature_request": true}'], captured)
    labeler.confirm_feature_request(
        'Does the dashboard have a heat-map view of transfer failures?',
        context='would be great to see a heat-map of failures by hour')
    user = captured[0]['body']['messages'][1]['content']
    assert 'would be great to see a heat-map' in user
    assert 'wish-phrasing' in captured[0]['body']['messages'][0]['content']


# ---- Fixture-2 round 3: split-halves must survive ----

def test_consolidation_drop_needs_verifier_agreement(monkeypatch):
    """Two-judge rule for consolidation: the consolidator nominated dropping
    the maintenance-window half of a genuine two-part message; the verifier
    says it's a DIFFERENT ask, so it stays."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {
        'can we restrict which ip ranges a partner can connect from?': [1.0, 0.0],
        'is there a way to run a transfer only during a maintenance window?': [0.0, 1.0],
    }
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        out = []
        for i, t in enumerate(texts):
            if 'IP ranges' in t or 'ip ranges' in t.lower():
                out.append({'index': i, 'question': 'Can we restrict which IP ranges a partner can connect from?'})
                out.append({'index': i, 'question': 'Is there a way to run a transfer only during a maintenance window?'})
        return out

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: False,  # judge 2: distinct asks
             extract_questions=extract,
             consolidate_same_ask=lambda msg, texts: [1],  # judge 1 over-collapses
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-06-05\nCan we restrict which IP ranges a partner connects from? "
        "And separately, can a transfer run only during a maintenance window?\n")
    assert results['total_questions'] == 2  # second half survived


def test_consolidation_drop_proceeds_when_judges_agree(monkeypatch):
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'could the scheduler be running in the wrong timezone after dst?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        return [
            {'index': 0, 'question': 'Could the scheduler be running in the wrong timezone after DST?'},
            {'index': 0, 'question': 'Is the transfers-stopping-after-DST issue timezone-related?'},
        ]

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: True,  # judge 2 agrees: same ask
             extract_questions=extract,
             consolidate_same_ask=lambda msg, texts: [1],
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-06-04\nScheduled transfers stop after DST. Could the scheduler "
        "be in the wrong timezone after DST changes?\n")
    assert results['total_questions'] == 1


def test_under_extraction_recovered_by_widened_safety_net(monkeypatch):
    """A message with TWO regex-visible questions that produced only ONE
    extraction gets the quality-model second look — the missing half is
    recovered instead of vanishing invisibly."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {
        'does mft support mutual tls for https partner endpoints?': [1.0, 0.0],
        'can mft post a transfer-complete event to an external kafka topic?': [0.0, 1.0],
        # The regex fallback may restore the verbatim segment too; after the
        # same-message collapse only the best-supported survivor remains
        'also can it post a transfer-complete event to an external kafka topic?': [0.0, 1.0],
    }
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        out = []
        for i, t in enumerate(texts):
            if 'mutual TLS' in t:
                out.append({'index': i, 'question': 'Does MFT support mutual TLS for HTTPS partner endpoints?'})
                if thorough:  # the 8B sees the second ask the 3B missed
                    out.append({'index': i, 'question': 'Can MFT post a transfer-complete event to an external Kafka topic?'})
        return out

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: False,
             extract_questions=extract,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        "2024-06-06\nDoes MFT support mutual TLS for HTTPS partner endpoints? "
        "Also, can it post a transfer-complete event to an external Kafka topic?\n")
    texts = [q['text'] for q in results['ungrouped_questions']]
    assert results['total_questions'] == 2
    assert any('Kafka' in t for t in texts)  # the lost half came back


def test_cancel_check_fires_before_llm_calls_and_propagates(monkeypatch):
    """Cancel must take effect between LLM calls, not at stage boundaries —
    and the cancellation exception must NOT be swallowed by the generic
    LLM-failure handler."""
    class Cancelled(Exception):
        pass

    labeler = GroupLabeler('ollama')

    def raise_cancel():
        raise Cancelled()
    labeler.cancel_check = raise_cancel

    patch_chat(monkeypatch, ['{"topic": "X", "summary": "y"}'])
    with pytest.raises(Cancelled):
        labeler.label_group(['Some question?'])


def test_example_leak_guard():
    """Few-shot example questions copied into extraction output are
    contamination — unless the claimed source message genuinely contains
    the ask, in which case it is a real question and survives."""
    from slack_question_analyzer.group_labeler import _looks_like_example_leak

    copy_action_msg = ('Seeing this in the server log on 10.15 when a Copy '
                       'action runs: Transfer aborted unexpectedly. '
                       'Any idea what triggers this?')
    # Verbatim prompt example, claimed against an unrelated message: leak
    assert _looks_like_example_leak(
        'Can we trigger transfers via REST instead of the scheduler?',
        copy_action_msg)
    # Same text, but the source REALLY asks it: not a leak
    assert not _looks_like_example_leak(
        'Can we trigger transfers via REST instead of the scheduler?',
        'Could we trigger transfers via REST instead of the scheduler? '
        'The scheduler keeps colliding with maintenance windows.')
    # Ordinary questions never trip the guard
    assert not _looks_like_example_leak(
        'How do I rotate SSH keys on a live listener?', copy_action_msg)
    assert not _looks_like_example_leak('', copy_action_msg)


def test_questions_from_llm_filters_example_leaks(monkeypatch):
    """The extraction funnel drops leaked examples and counts them."""
    labeler = GroupLabeler('ollama')
    base = GroupLabeler.MESSAGE_INDEX_BASE
    monkeypatch.setattr(labeler, '_generate_json', lambda *a, **k: {
        'questions': [
            {'index': base, 'question': 'How do I increase the SFTP timeout?',
             'type': 'how-to'},
            {'index': base, 'question': 'Can we trigger transfers via REST '
                                        'instead of the scheduler?',
             'type': 'is-it-possible'},
        ]})
    found = labeler.extract_questions(
        ['How do I increase the SFTP timeout? Transfers keep timing out.'])
    assert [f['question'] for f in found] == [
        'How do I increase the SFTP timeout?']
    assert labeler.stats.get('extract_example_leaks') == 1


def test_feedback_diversion_requires_wish_phrasing(monkeypatch):
    """Eval round 3: the 8B confirm diverted plain capability questions
    ('can we cap transfers per node?') out of the support funnel — killing
    a true recurrence and the Answered count. Without wish-phrasing in the
    source message, a question stays in support no matter what any model
    says."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'can we cap how many transfers run at once per node?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: [
                 {'index': 0, 'question': 'Can we cap how many transfers run at once per node?',
                  'type': 'feature-request'}],  # mis-tagged
             confirm_feature_request=lambda text, context='': True,  # 8B wrong too
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        '2024-06-03\nCan we cap how many transfers run at once per node to prevent overload?\n')
    assert results['feature_requests'] == []          # both models overruled
    assert results['total_questions'] == 1


def test_feedback_diversion_explicit_label_is_decisive(monkeypatch):
    """A message the asker explicitly headed 'Feature request:' (with wish
    phrasing) diverts deterministically — no model tag, no confirm call.
    Eval round 3 found these never reached the confirmer because the 3B
    tag was the only gate."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'can the ui let you configure retry-with-backoff?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    confirm_calls = []

    def confirm(text, context=''):
        confirm_calls.append(text)
        return False

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: [
                 {'index': 0, 'question': 'Can the UI let you configure retry-with-backoff?',
                  'type': 'is-it-possible'}],  # NOT tagged feature-request
             confirm_feature_request=confirm,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        '2024-06-03\nFeature request: the customer would like the UI to let '
        'you configure a retry-with-backoff policy without editing config files.\n')
    assert len(results['feature_requests']) == 1
    assert confirm_calls == []  # label + wish: deterministic, no LLM needed
    assert results['feature_requests'][0]['qtype'] == 'feature-request'


def test_feedback_diversion_wish_alone_needs_confirmation(monkeypatch):
    """Wish-phrasing without an explicit label opens the gate but the
    quality model decides (this is how an untagged 'would love a heat-map
    view' item still reaches feedback)."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {'can the dashboard get a heat-map view of failures?': [1.0, 0.0]}
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: [
                 {'index': 0, 'question': 'Can the dashboard get a heat-map view of failures?',
                  'type': 'is-it-possible'}],
             confirm_feature_request=lambda text, context='': True,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        '2024-06-03\nCustomer would love it if the dashboard had a heat-map '
        'view of transfer failures. Doesn\'t exist today but would help.\n')
    assert len(results['feature_requests']) == 1


def test_question_shaped_statement_not_restored_over_two_model_nos(monkeypatch):
    """Fixture-5 fabrication: 'Will post here when it's back up' is
    question-shaped to the regex (question word at start) but both models
    said the announcement contains no ask. When the quality-model call
    SUCCEEDED, only an explicit '?' can overrule it; a failed call still
    restores everything."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    analyzer = make_analyzer(monkeypatch, vectors={}, label_groups=True)
    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: [],  # both: no ask
             summarize_analysis=lambda groups, total, themes=None: None)
    results = analyzer.analyze_slack_content(
        '2024-06-09\nHeads up team - maintenance window this Saturday. '
        'Will post here when it is back up.\n')
    assert results['total_questions'] == 0

    # Same shape WITH a '?': the regex version is kept (the IWHI case)
    vectors = {'will the staging tenant stay up during the move?': [1.0, 0.0]}
    analyzer2 = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)
    stub_llm(monkeypatch, analyzer2,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: None,
             extract_questions=lambda texts, thorough=False: [],
             summarize_analysis=lambda groups, total, themes=None: None)
    results = analyzer2.analyze_slack_content(
        '2024-06-09\nWill the staging tenant stay up during the move?\n')
    assert results['total_questions'] == 1


def test_choose_bucket_presents_abstain_as_a_listed_option(monkeypatch):
    """Fixture-6: off-topic questions reached the adjudicator and it picked
    from the list every time — the abstain rule sat unused in the system
    prompt. 0 must be a VISIBLE candidate line."""
    captured = {}
    labeler = GroupLabeler('ollama')

    def fake_generate(system, user, schema, **kwargs):
        captured['user'] = user
        return {'category': 0}

    monkeypatch.setattr(labeler, '_generate_json', fake_generate)
    assert labeler.choose_bucket('Is the wiki down?', [
        {'id': 3, 'name': 'Connectivity'}, {'id': 5, 'name': 'Scheduling'}]) == 0
    assert '0 NONE OF THESE' in captured['user']


def test_compound_question_missing_half_restored_by_regex_fallback(monkeypatch):
    """End-to-end mechanism check for the two-round Kafka loss: a compound
    'and separately' sentence where the LLM only ever extracts the first
    half (fast AND thorough). The splitter makes regex see two questions,
    the under-extraction trigger fires, and the '?'-bearing missing half is
    restored by the regex fallback."""
    monkeypatch.setenv('LLM_EXTRACTION', 'full')
    vectors = {
        'does mft support mutual tls for https partner endpoints?': [1.0, 0.0],
        'can it post a transfer-complete event to an external kafka topic?': [0.0, 1.0],
    }
    analyzer = make_analyzer(monkeypatch, vectors=vectors, label_groups=True)

    def extract(texts, thorough=False):
        # The model NEVER sees the second half, no matter which model runs
        out = []
        for i, t in enumerate(texts):
            if 'mutual TLS' in t:
                out.append({'index': i, 'question':
                            'Does MFT support mutual TLS for HTTPS partner endpoints?'})
        return out

    stub_llm(monkeypatch, analyzer,
             label_group=lambda texts, keywords=None: None,
             verify_same_topic=lambda a, b: False,
             extract_questions=extract,
             summarize_analysis=lambda groups, total, themes=None: None)

    results = analyzer.analyze_slack_content(
        '2024-06-06\nQuick one - does MFT support mutual TLS for HTTPS partner '
        'endpoints, and separately, can it post a transfer-complete event to '
        'an external Kafka topic?\n')
    texts = [q['text'] for q in results['ungrouped_questions']]
    assert results['total_questions'] == 2
    assert any('Kafka' in t for t in texts), texts
