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
    assert tax.enabled and tax.version == 5
    assert len(tax.buckets) == 9
    # v4: the AI/agent capability area earned its own bucket (three
    # recurring singletons force-routed into wrong buckets was the signal)
    assert any(b['name'] == 'AI & Automation' for b in tax.buckets)
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
    assert [i for i, _, _ in ambiguous] == [1]  # weak best: LLM decides
    # Tagged 'floor' so a saturated budget sends it to review, not the
    # nearest wrong bucket
    assert ambiguous[0][2] == 'floor'
    assert ambiguous[0][1][0] == 0           # embedding favorite first
    assert outliers == []                    # weak is NOT an outlier drop


# ---- Cluster-coherence gate ----

def _coherence_setup(tmp_path, monkeypatch, member_vectors):
    """Three-bucket taxonomy + a 3-question cluster with injectable member
    vectors. Members always cluster together; where they route is the test."""
    path = write_taxonomy(tmp_path, [
        {'id': 1, 'name': 'A', 'anchor': 'anchor a'},
        {'id': 2, 'name': 'B', 'anchor': 'anchor b'},
        {'id': 3, 'name': 'C', 'anchor': 'anchor c'},
    ])
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False,
                                label_groups=False)

    texts = ['agent question one?', 'agent question two?',
             'agent question three?']
    vectors = dict(zip(texts, member_vectors))
    # One anchor per lean direction: a member scores 1.0 against "its"
    # anchor and only `mutual` against the other two
    anchor_vectors = _leaning_vectors([0, 1, 2])
    vectors['anchor a'] = anchor_vectors[0]
    vectors['anchor b'] = anchor_vectors[1]
    vectors['anchor c'] = anchor_vectors[2]

    monkeypatch.setattr(
        analyzer.similarity_analyzer, 'get_embeddings_batch',
        lambda t, progress_callback=None: np.array([vectors[x] for x in t]))

    questions = [{'text': t, 'normalized_text': t, 'original_message': t,
                  'date': 'Unknown'} for t in texts]
    return analyzer, Taxonomy(path=path), questions


def _leaning_vectors(leans, mutual=0.86):
    """Unit vectors sharing a common component (pairwise sim = `mutual`)
    while each leans toward a distinct axis given by `leans`."""
    alpha, beta = np.sqrt(mutual), np.sqrt(1 - mutual)
    common = np.zeros(4)
    common[0] = 1.0
    axes = np.eye(4)[1:]
    return [(alpha * common + beta * axes[lean]).tolist() for lean in leans]


def test_exhausted_routing_budget_sends_weak_clusters_to_review(tmp_path, monkeypatch):
    """Regression (2026-07-07 field run): with the adjudication budget
    saturated, WEAK-best (floor) clusters were force-routed to their nearest
    bucket — making the review pile structurally unreachable on busy runs.
    Without an LLM verdict, a floor case must go to review; a margin case
    (two strong candidates) may still take the embedding favorite."""
    path = write_taxonomy(tmp_path, [
        {'id': 1, 'name': 'A', 'anchor': 'anchor a'},
        {'id': 2, 'name': 'B', 'anchor': 'anchor b'},
    ])
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')
    monkeypatch.setenv('ROUTE_LLM_MAX', '0')  # budget already spent
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False,
                                label_groups=False)

    # floor case: best anchor ~0.5 (over the 0.4 outlier floor, under the
    # 0.55 confidence floor); margin case: 0.72/0.70 against the two anchors
    vectors = {
        'anchor a': [1.0, 0.0, 0.0, 0.0],
        'anchor b': [0.0, 1.0, 0.0, 0.0],
        'status check thing?': [0.5, 0.0, 0.866, 0.0],
        'strong both ways?': [0.715, 0.695, 0.076, 0.0],
    }
    monkeypatch.setattr(
        analyzer.similarity_analyzer, 'get_embeddings_batch',
        lambda t, progress_callback=None: np.array([vectors[x] for x in t]))
    questions = [{'text': t, 'normalized_text': t, 'original_message': t,
                  'date': 'Unknown'}
                 for t in ('status check thing?', 'strong both ways?')]

    groups = analyzer._group_with_taxonomy(questions, Taxonomy(path=path),
                                           verifier=None, auditor=None,
                                           known_topics=None,
                                           report=lambda *a: None)
    by_text = {g['representative_question']: g for g in groups}
    weak = by_text['status check thing?']
    strong = by_text['strong both ways?']
    assert all(q.get('needs_review') for q in weak['questions'])
    assert not any(q.get('needs_review') for q in strong['questions'])
    assert strong['bucket'] == 'A'  # embedding favorite, acceptable fallback


def test_status_request_goes_to_review_not_a_bucket(tmp_path, monkeypatch):
    """'Can someone check on <url>?' asks a PERSON to act — no category
    answers it. It must reach the review pile deterministically, even when
    embeddings would confidently route it somewhere."""
    path = write_taxonomy(tmp_path, [
        {'id': 1, 'name': 'A', 'anchor': 'anchor a'},
        {'id': 2, 'name': 'B', 'anchor': 'anchor b'},
    ])
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False,
                                label_groups=False)
    status_q = ('Can someone please check on '
                'https://prod99.a-vir.mft.example.com/?')
    vectors = {
        'anchor a': [1.0, 0.0, 0.0],
        'anchor b': [0.0, 1.0, 0.0],
    }
    monkeypatch.setattr(
        analyzer.similarity_analyzer, 'get_embeddings_batch',
        # Any normalization of the status question: embeddings LOVE bucket A
        lambda t, progress_callback=None: np.array(
            [vectors.get(x.lower(), [0.9, 0.1, 0.0]) for x in t]))
    questions = [{'text': status_q, 'normalized_text': status_q.lower(),
                  'original_message': status_q, 'date': 'Unknown'}]
    groups = analyzer._group_with_taxonomy(questions, Taxonomy(path=path),
                                           verifier=None, auditor=None,
                                           known_topics=None,
                                           report=lambda *a: None)
    assert all(q.get('needs_review') for g in groups for q in g['questions'])


def test_distinctive_token_breaks_margin_tie_without_llm(tmp_path, monkeypatch):
    """Margin-ambiguous route whose representative mentions a token unique
    to the embedding favorite (and none unique to the runner-up) is
    confirmed deterministically — the LLM is never consulted."""
    path = write_taxonomy(tmp_path, [
        {'id': 1, 'name': 'Metering', 'anchor': 'metering license entitlement counting transactions'},
        {'id': 2, 'name': 'Monitoring', 'anchor': 'monitoring alerting dashboards observing health'},
    ])
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False,
                                label_groups=False)
    q = 'How is metering counted for transfers?'
    vectors = {
        'metering license entitlement counting transactions': [1.0, 0.0, 0.0],
        'monitoring alerting dashboards observing health': [0.0, 1.0, 0.0],
        # near-tie: 0.71 vs 0.69 (margin < 0.05), both above the 0.55 floor
        q.lower(): [0.71, 0.69, 0.14],
    }
    monkeypatch.setattr(
        analyzer.similarity_analyzer, 'get_embeddings_batch',
        lambda t, progress_callback=None: np.array(
            [vectors[x.lower()] for x in t]))
    monkeypatch.setattr(analyzer, '_llm_enabled', lambda mode: True)

    from slack_question_analyzer.group_labeler import GroupLabeler
    if analyzer.labeler is None:
        analyzer.labeler = GroupLabeler('ollama')

    def never(*a, **k):
        raise AssertionError('distinctive-token evidence should have '
                             'settled this without the LLM')
    monkeypatch.setattr(analyzer.labeler, 'choose_bucket', never)

    questions = [{'text': q, 'normalized_text': q.lower(),
                  'original_message': q, 'date': 'Unknown'}]
    groups = analyzer._group_with_taxonomy(questions, Taxonomy(path=path),
                                           verifier=None, auditor=None,
                                           known_topics=None,
                                           report=lambda *a: None)
    assert groups[0]['bucket'] == 'Metering'
    assert not any(q.get('needs_review') for q in groups[0]['questions'])


def test_coherence_gate_sends_split_cluster_to_review_on_abstain(tmp_path, monkeypatch):
    """The emerging-topic case: a coherent cluster whose members each route
    to a DIFFERENT bucket must not silently take the representative's bucket.
    It goes to the LLM's closed choice; abstain -> the whole cluster is
    held for review."""
    from slack_question_analyzer.group_labeler import GroupLabeler
    analyzer, tax, questions = _coherence_setup(
        tmp_path, monkeypatch, _leaning_vectors([0, 1, 2]))
    if analyzer.labeler is None:
        analyzer.labeler = GroupLabeler('ollama')
    monkeypatch.setattr(analyzer, '_llm_enabled', lambda mode: True)
    monkeypatch.setattr(analyzer.labeler, 'choose_bucket',
                        lambda q, cands: 0)  # honest abstain

    groups = analyzer._group_with_taxonomy(questions, tax, verifier=None,
                                           auditor=None, known_topics=None,
                                           report=lambda *a: None)
    cluster = next(g for g in groups if g['count'] == 3)
    assert all(q.get('needs_review') for q in cluster['questions'])


def test_coherence_gate_leaves_member_majority_alone(tmp_path, monkeypatch):
    """Control: members that agree with each other (and the representative)
    keep their confident route WITHOUT consuming an adjudication call —
    the gate must not create over-abstention."""
    from slack_question_analyzer.group_labeler import GroupLabeler
    analyzer, tax, questions = _coherence_setup(
        tmp_path, monkeypatch, _leaning_vectors([0, 0, 0]))
    if analyzer.labeler is None:
        analyzer.labeler = GroupLabeler('ollama')
    monkeypatch.setattr(analyzer, '_llm_enabled', lambda mode: True)
    calls = []
    monkeypatch.setattr(analyzer.labeler, 'choose_bucket',
                        lambda q, cands: calls.append(q) or 0)

    groups = analyzer._group_with_taxonomy(questions, tax, verifier=None,
                                           auditor=None, known_topics=None,
                                           report=lambda *a: None)
    cluster = next(g for g in groups if g['count'] == 3)
    assert cluster.get('bucket') == 'A'
    assert calls == []  # coherent cluster never reaches the LLM
    assert not any(q.get('needs_review') for q in cluster['questions'])


def test_coherence_gate_skipped_without_an_adjudicator(tmp_path, monkeypatch):
    """Without an LLM to adjudicate, demotion would fall back to a raw
    member vote — strictly worse than the representative's confident
    route. The gate must stand down entirely."""
    analyzer, tax, questions = _coherence_setup(
        tmp_path, monkeypatch, _leaning_vectors([0, 1, 2]))
    # No labeler / LLM available (default in tests)
    groups = analyzer._group_with_taxonomy(questions, tax, verifier=None,
                                           auditor=None, known_topics=None,
                                           report=lambda *a: None)
    cluster = next(g for g in groups if g['count'] == 3)
    assert cluster.get('bucket')  # kept its confident route
    assert not any(q.get('needs_review') for q in cluster['questions'])


def test_coherence_gate_keeps_original_bucket_past_llm_budget(tmp_path, monkeypatch):
    """Past ROUTE_LLM_MAX the fallback is the FIRST candidate — which must
    be the original confident bucket, never a minority member vote."""
    from slack_question_analyzer.group_labeler import GroupLabeler
    monkeypatch.setenv('ROUTE_LLM_MAX', '0')
    analyzer, tax, questions = _coherence_setup(
        tmp_path, monkeypatch, _leaning_vectors([0, 1, 2]))
    if analyzer.labeler is None:
        analyzer.labeler = GroupLabeler('ollama')
    monkeypatch.setattr(analyzer, '_llm_enabled', lambda mode: True)
    monkeypatch.setattr(analyzer.labeler, 'choose_bucket',
                        lambda q, cands: (_ for _ in ()).throw(
                            AssertionError('budget is 0 - must not be called')))

    groups = analyzer._group_with_taxonomy(questions, tax, verifier=None,
                                           auditor=None, known_topics=None,
                                           report=lambda *a: None)
    cluster = next(g for g in groups if g['count'] == 3)
    # The representative's own anchor is bucket A (lean 0 wins for the rep)
    assert cluster.get('bucket') == 'A'
    assert not any(q.get('needs_review') for q in cluster['questions'])


def test_taxonomy_malformed_shapes_disable_instead_of_crashing(tmp_path):
    """A parseable file with the wrong SHAPE (list at top level, string
    buckets) must disable routing with a warning — not crash the analysis
    with an AttributeError."""
    for content in ('["not", "an", "object"]',
                    '{"version": 1, "buckets": ["a", "b"]}',
                    '{"version": 1, "buckets": {}}'):
        path = tmp_path / 'tax.json'
        path.write_text(content, encoding='utf-8')
        assert not Taxonomy(path=str(path)).enabled


def test_taxonomy_found_from_any_working_directory(tmp_path, monkeypatch):
    """The repo-root taxonomy.json is a fallback: running the CLI or server
    from a different directory must not silently lose the routing buckets.
    An explicit TAXONOMY_PATH still wins."""
    import os
    from slack_question_analyzer.taxonomy import Taxonomy
    monkeypatch.delenv('TAXONOMY_PATH', raising=False)
    monkeypatch.chdir(tmp_path)  # no taxonomy.json here
    assert Taxonomy().buckets  # found via the repo-root fallback

    custom = tmp_path / 'custom.json'
    custom.write_text(json.dumps({'version': 99, 'buckets': [
        {'id': 'x', 'name': 'X', 'anchor': 'questions about x'}]}),
        encoding='utf-8')
    monkeypatch.setenv('TAXONOMY_PATH', str(custom))
    assert Taxonomy().version == 99
    assert os.getenv('TAXONOMY_PATH') == str(custom)
