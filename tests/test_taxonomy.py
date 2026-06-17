"""The routing taxonomy: bucket definitions, embedding routing, and the
taxonomy-first funnel through the analyzer."""

import json

import numpy as np
import pytest

from slack_question_analyzer.taxonomy import Taxonomy, route_questions
from slack_question_analyzer.analyzer import QuestionAnalyzer
from slack_question_analyzer.group_labeler import GroupLabeler


# ---- Routing math (pure code) ----

def test_route_confident_ambiguous_and_outlier():
    anchors = [[1.0, 0.0], [0.0, 1.0]]
    questions = [
        [0.95, 0.1],    # clearly anchor 0
        [0.72, 0.7],    # ambiguous: nearly equidistant
        [-1.0, -1.0],   # near nothing -> outlier
    ]
    assignments, ambiguous, outliers = route_questions(
        questions, anchors, outlier_floor=0.4, ambiguity_margin=0.03)
    assert assignments == {0: 0}
    assert len(ambiguous) == 1 and ambiguous[0][0] == 1
    assert ambiguous[0][1][0] == 0  # embedding favorite listed first
    assert outliers == [2]


def test_route_single_anchor_never_ambiguous():
    assignments, ambiguous, outliers = route_questions(
        [[1.0, 0.0]], [[1.0, 0.0]], outlier_floor=0.4, ambiguity_margin=0.03)
    assert assignments == {0: 0} and not ambiguous and not outliers


# ---- Taxonomy loading ----

def write_taxonomy(tmp_path, buckets, version=7):
    path = tmp_path / 'tax.json'
    path.write_text(json.dumps({'version': version, 'buckets': buckets}),
                    encoding='utf-8')
    return str(path)


def test_taxonomy_loads_and_maps_categories(tmp_path):
    path = write_taxonomy(tmp_path, [
        {'id': 1, 'name': 'Antivirus', 'anchor': 'virus scanning', 'category': 'File Ops'},
        {'id': 2, 'name': 'Monitoring', 'anchor': 'alerts and dashboards'},
    ])
    tax = Taxonomy(path=path)
    assert tax.enabled and tax.version == 7
    assert tax.anchor_texts() == ['virus scanning', 'alerts and dashboards']
    assert tax.final_category(0) == 'File Ops'
    assert tax.final_category(1) == 'Monitoring'  # falls back to bucket name


def test_taxonomy_disabled_when_missing_or_off(tmp_path, monkeypatch):
    assert not Taxonomy(path=str(tmp_path / 'nope.json')).enabled
    path = write_taxonomy(tmp_path, [
        {'id': 1, 'name': 'A', 'anchor': 'a'}])
    monkeypatch.setenv('TAXONOMY', 'off')
    assert not Taxonomy(path=path).enabled


def test_taxonomy_rejects_malformed_buckets(tmp_path):
    path = write_taxonomy(tmp_path, [{'id': 1, 'name': 'A'}])  # no anchor
    assert not Taxonomy(path=path).enabled


def test_shipped_taxonomy_is_valid():
    tax = Taxonomy(path='taxonomy.json')
    assert tax.enabled and tax.version == 3
    assert len(tax.buckets) == 8
    assert all(b.get('category') for b in tax.buckets)
    # v2 convention: the Action log lives in File Handling
    file_handling = next(b for b in tax.buckets if b['name'] == 'File Handling')
    assert 'Action log' in file_handling['anchor']


# ---- LLM adjudication (closed choice) ----

class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def patch_chat(monkeypatch, content):
    monkeypatch.setattr(
        'slack_question_analyzer.group_labeler.requests.post',
        lambda url, json=None, timeout=None: FakeResponse(
            {'message': {'role': 'assistant', 'content': content}}))


def test_choose_bucket_returns_valid_choice(monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path / 'llm'))
    patch_chat(monkeypatch, '{"category": 6}')
    chosen = GroupLabeler('ollama').choose_bucket(
        'How do I set up e2e monitoring alerts?',
        [{'id': 6, 'name': 'Monitoring & Alerting'},
         {'id': 2, 'name': 'Metering & Licensing'}])
    assert chosen == 6


def test_choose_bucket_rejects_invented_numbers(monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path / 'llm'))
    patch_chat(monkeypatch, '{"category": 99}')
    assert GroupLabeler('ollama').choose_bucket(
        'q?', [{'id': 1, 'name': 'A'}]) is None


# ---- End-to-end: taxonomy-first funnel through the analyzer ----

SLACK = ("2024-01-05\nHow can we configure a virus scan email notification?\n"
         "-----------------------------------------------------------\n"
         "2024-01-06\nCan we move infected files to a quarantine folder?\n"
         "-----------------------------------------------------------\n"
         "2024-01-07\nHow do I set up e2e monitoring alerts?\n"
         "-----------------------------------------------------------\n"
         "2024-01-08\nCompletely unrelatable gibberish thing entirely?\n")

VECTORS = {
    # anchors
    'virus scanning, quarantine, infected files': [1.0, 0.0, 0.0],
    'monitoring, alerting, dashboards': [0.0, 1.0, 0.0],
    # questions (normalized text) — the two antivirus questions are only 0.62
    # similar to each other, but both clearly route to the antivirus anchor
    'how can we configure a virus scan email notification?': [0.9, 0.0, 0.435],
    'can we move infected files to a quarantine folder?': [0.9, 0.0, -0.435],
    'how do i set up e2e monitoring alerts?': [0.05, 0.95, 0.0],
    'completely unrelatable gibberish thing entirely?': [-0.5, -0.5, 0.7],
}


@pytest.fixture
def taxonomy_analyzer(tmp_path, monkeypatch):
    path = write_taxonomy(tmp_path, [
        {'id': 1, 'name': 'Antivirus', 'category': 'File Operations',
         'anchor': 'virus scanning, quarantine, infected files'},
        {'id': 2, 'name': 'Monitoring', 'category': 'Operations',
         'anchor': 'monitoring, alerting, dashboards'},
    ], version=3)
    monkeypatch.setenv('TAXONOMY_PATH', path)
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False,
                                label_groups=False)
    monkeypatch.setattr(
        analyzer.similarity_analyzer, 'get_embeddings_batch',
        lambda texts, progress_callback=None: np.array([VECTORS[t] for t in texts]))
    return analyzer


def test_taxonomy_funnel_end_to_end(taxonomy_analyzer):
    results = taxonomy_analyzer.analyze_slack_content(SLACK)

    # The two antivirus questions grouped INSIDE their bucket despite 0.62
    # pairwise similarity being far below the pinned 0.85 (fixed in-bucket
    # bar 0.85 applies since the user pinned... bucket coherence held them)
    assert results['total_questions'] == 4

    # Themes come from the deterministic merge map, no LLM involved
    themes = {t['name']: t['count'] for t in results['themes']}
    assert themes.get('File Operations') == 2
    assert themes.get('Operations') == 1

    # The gibberish question survived, flagged for review
    flagged = [q for q in results['ungrouped_questions'] if q.get('needs_review')]
    assert len(flagged) == 1
    assert 'gibberish' in flagged[0]['text']

    # Routing health metrics are stamped into metadata
    routing = results['metadata']['routing']
    assert routing['taxonomy_version'] == 3
    assert routing['routed'] == 3
    assert routing['needs_review'] == 1


def test_taxonomy_groups_within_bucket_at_relaxed_bar(taxonomy_analyzer, monkeypatch):
    """With auto threshold (not pinned), the in-bucket bar is the fixed
    relaxed IN_BUCKET_THRESHOLD — the adaptive gate must stay out."""
    monkeypatch.delenv('SIMILARITY_THRESHOLD')
    monkeypatch.setenv('IN_BUCKET_THRESHOLD', '0.6')
    analyzer = taxonomy_analyzer
    analyzer.similarity_analyzer.threshold_pinned = False

    results = analyzer.analyze_slack_content(SLACK)
    assert results['total_groups'] == 1  # the two antivirus questions paired
    group = results['groups'][0]
    assert group['theme'] == 'File Operations'
    assert group['bucket'] == 'Antivirus'


def test_choose_bucket_zero_is_honest_abstain(monkeypatch, tmp_path):
    """Reply 0 = fits both/neither -> quarantine, never a forced guess."""
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path / 'llm'))
    patch_chat(monkeypatch, '{"category": 0}')
    assert GroupLabeler('ollama').choose_bucket(
        'q?', [{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]) == 0


def test_route_weak_best_match_goes_to_llm_not_forced():
    """Fixture-6 regression: off-topic and too-vague questions ('is the
    wiki down?') still score ~0.5 against SOME anchor on shared vocabulary
    and were force-routed. A weak best match (under the confidence floor
    but over the outlier floor) is a closed LLM choice — where abstain
    sends it to review — never a silent embedding route."""
    anchors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    questions = [
        [0.95, 0.1, 0.0],   # strong, clear -> routed by embeddings alone
        [0.5, 0.1, 0.85],   # weak best (~0.50 cosine): off-topic-ish
    ]
    assignments, ambiguous, outliers = route_questions(
        questions, anchors, outlier_floor=0.4, ambiguity_margin=0.03,
        confidence_floor=0.55)
    assert assignments == {0: 0}            # strong + clear: routed
    assert [i for i, _ in ambiguous] == [1]  # weak best: LLM decides
    assert ambiguous[0][1][0] == 0           # embedding favorite first
    assert outliers == []                    # weak is NOT an outlier drop
