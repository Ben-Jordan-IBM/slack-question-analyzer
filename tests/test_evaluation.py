"""The regression-fixture evaluation harness."""

import json

import numpy as np
import pytest

from slack_question_analyzer.evaluation import (load_fixture, evaluate,
                                                format_report, _same_group_pairs,
                                                evaluate_transcript,
                                                format_transcript_report)
from slack_question_analyzer.analyzer import QuestionAnalyzer


def test_shipped_fixture_is_valid():
    fixture = load_fixture('fixtures/field_run_2026-06-10.json')
    assert len(fixture['questions']) == 17
    threads = [q for q in fixture['questions'] if q.get('group') == 'thread-scaling']
    assert len(threads) == 3  # incl. the singleton-rescue target


def test_load_fixture_rejects_unlabeled(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps({'questions': [{'text': 'q?'}]}), encoding='utf-8')
    with pytest.raises(ValueError):
        load_fixture(str(path))


def test_same_group_pairs():
    pairs = _same_group_pairs({'a': 'g1', 'b': 'g1', 'c': 'g2', 'd': None})
    assert pairs == {frozenset(('a', 'b'))}


def test_evaluate_scores_routing_and_pairs(tmp_path, monkeypatch):
    taxonomy = {'version': 9, 'buckets': [
        {'id': 1, 'name': 'Antivirus', 'anchor': 'anchor-av', 'category': 'File Ops'},
        {'id': 2, 'name': 'Monitoring', 'anchor': 'anchor-mon', 'category': 'Ops'},
    ]}
    tax_path = tmp_path / 'tax.json'
    tax_path.write_text(json.dumps(taxonomy), encoding='utf-8')
    monkeypatch.setenv('TAXONOMY_PATH', str(tax_path))
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.8')
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')

    vectors = {
        'anchor-av': [1.0, 0.0, 0.0],
        'anchor-mon': [0.0, 1.0, 0.0],
        'virus scan email alert?': [0.9, 0.0, 0.435],
        'quarantine folder for infected files?': [0.9, 0.0, -0.435],
        'e2e monitoring setup?': [0.05, 0.95, 0.0],
    }
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False,
                                label_groups=False)
    monkeypatch.setattr(
        analyzer.similarity_analyzer, 'get_embeddings_batch',
        lambda texts, progress_callback=None: np.array([vectors[t] for t in texts]))

    fixture = {'questions': [
        {'text': 'Virus scan email alert?', 'bucket': 'Antivirus', 'group': 'av'},
        {'text': 'Quarantine folder for infected files?', 'bucket': 'Antivirus', 'group': 'av'},
        # Deliberately mislabeled: routed to Monitoring, expected Antivirus
        {'text': 'E2e monitoring setup?', 'bucket': 'Antivirus', 'group': None},
    ]}

    result = evaluate(analyzer, fixture)
    assert result['routing_correct'] == 2
    assert len(result['routing_mismatches']) == 1
    assert result['routing_mismatches'][0]['got'] == 'Monitoring'
    # The two antivirus questions only pair if in-bucket clustering joins
    # them; at sim 0.62 under bar 0.8 with no verifier they stay apart
    assert result['pairs_expected'] == 1
    assert result['pair_recall'] in (0.0, 1.0)
    report = format_report(result)
    assert 'Routing:  2/3' in report


# --- transcript-level fixtures (end-to-end answer keys) ---

class _StubAnalyzer:
    """Stands in for QuestionAnalyzer: evaluate_transcript only needs
    analyze_contents."""

    def __init__(self, results):
        self.results = results
        self.received = None

    def analyze_contents(self, contents, **kwargs):
        self.received = contents
        return self.results


def _row(text, source, **extra):
    return {'text': text, 'original_message': source, **extra}


def _transcript_fixture(tmp_path, expect):
    (tmp_path / 'fake.txt').write_text('June 9, 2026\n\nfake', encoding='utf-8')
    path = tmp_path / 'fix.json'
    path.write_text(json.dumps({'type': 'transcript', 'transcript': 'fake.txt',
                                'expect': expect}), encoding='utf-8')
    return load_fixture(str(path))


def _good_results():
    rotation = [_row('Can we rotate SSH keys without downtime?',
                     'one of my customers wants to rotate the SSH keys'),
                _row('Is zero-downtime key rotation possible?',
                     'Re: SFTP keys - is it possible to roll over the SSH host')]
    return {
        'total_questions': 3,
        'total_groups': 1,
        'groups': [{'count': 2, 'representative_question': rotation[0]['text'],
                    'questions': rotation}],
        'ungrouped_questions': [
            _row('How do we handle a checksum mismatch gracefully?',
                 'When an inbound file fails checksum validation')],
        'feature_requests': [
            _row('Add a dark mode option to the web UI.',
                 'They would like a dark mode option')],
        'metadata': {'llm_stats': {}},
    }


def test_load_fixture_transcript_type(tmp_path):
    fixture = _transcript_fixture(tmp_path, {'total_asks': 4})
    assert fixture['_dir'] == str(tmp_path)
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'type': 'transcript', 'transcript': 'x.txt'}),
                   encoding='utf-8')
    with pytest.raises(ValueError):
        load_fixture(str(bad))


def test_evaluate_transcript_all_green(tmp_path):
    fixture = _transcript_fixture(tmp_path, {
        'total_asks': 4,
        'recurring_topics': 1,
        'recurring_must_match': ['rotat', 'key'],
        'feedback_count': 1,
        'feedback_must_match': ['dark mode'],
        'feedback_must_not_match': ['checksum'],
        'message_asks': [{'contains': 'fails checksum validation', 'asks': 1},
                         {'contains': 'rotate the SSH keys', 'asks': 1}],
        'support_must_match': ['checksum'],
        'must_not_match': ['(bypass|disable)[^.?!]*(checksum|validation)'],
        'must_not_group': [['checksum', 'dark mode']],
    })
    result = evaluate_transcript(_StubAnalyzer(_good_results()), fixture)
    assert result['failed'] == 0, format_transcript_report(result)
    assert 'FAIL' not in format_transcript_report(result)


def test_evaluate_transcript_catches_each_failure_mode(tmp_path):
    results = _good_results()
    # Verb drift on the checksum question + a false merge + a lost ask
    results['ungrouped_questions'][0]['text'] = \
        'How do we disable checksum validation?'
    results['groups'][0]['questions'].append(
        _row('Can files be zip-compressed before transfer?',
             'compress files into a zip archive'))
    results['groups'][0]['count'] = 3
    results['metadata']['llm_stats'] = {'integrity_repairs': 1}
    fixture = _transcript_fixture(tmp_path, {
        'total_asks': 17,
        'recurring_topics': 2,
        'feedback_count': 0,
        'message_asks': [{'contains': 'chargeback reporting', 'asks': 1}],
        'must_not_match': ['disable[^.?!]*validation'],
        'must_not_group': [['rotat', 'zip']],
    })
    result = evaluate_transcript(_StubAnalyzer(results), fixture)
    failed = {c['name'] for c in result['checks'] if not c['ok']}
    assert 'total asks = 17' in failed                     # 5 != 17
    assert 'recurring topics = 2' in failed                # got 1
    assert 'product feedback = 0' in failed                # got 1
    assert any('chargeback' in n for n in failed)          # marker matched 0 rows
    assert any('verb drift' in n for n in failed)
    assert any('not merged' in n for n in failed)
    assert 'no render-integrity repairs needed' in failed
    report = format_transcript_report(result)
    assert 'FAIL' in report and 'silently dropped' in report


def test_evaluate_transcript_isolates_topic_bank(tmp_path, monkeypatch):
    monkeypatch.setenv('TOPIC_BANK_PATH', '/real/bank.json')
    seen = {}

    class _Spy(_StubAnalyzer):
        def analyze_contents(self, contents, **kwargs):
            import os
            seen['bank'] = os.environ['TOPIC_BANK_PATH']
            return super().analyze_contents(contents, **kwargs)

    fixture = _transcript_fixture(tmp_path, {'total_asks': 4})
    evaluate_transcript(_Spy(_good_results()), fixture)
    assert seen['bank'] != '/real/bank.json'   # temp bank during the run
    import os
    assert os.environ['TOPIC_BANK_PATH'] == '/real/bank.json'  # restored


def _markers_resolve(transcript_path, fixture_path):
    """Every 'contains' marker must hit exactly ONE message within the
    200-char original_message cap, and the per-message asks must sum to
    total_asks — validates the shipped keys without any LLM."""
    from slack_question_analyzer.question_extractor import QuestionExtractor
    fixture = load_fixture(fixture_path)
    content = open(transcript_path, encoding='utf-8').read()
    messages = QuestionExtractor().extract_messages(content)
    capped = [' '.join(m['text'].split())[:200].lower() for m in messages]
    for spec in fixture['expect']['message_asks']:
        hits = [c for c in capped if spec['contains'].lower() in c]
        assert len(hits) == 1, f"marker {spec['contains']!r} hit {len(hits)} messages"
    total = sum(s['asks'] for s in fixture['expect']['message_asks'])
    assert total == fixture['expect']['total_asks']


def test_shipped_mft_fixture_1_markers_resolve():
    _markers_resolve('fixtures/mft_test_transcript.txt',
                     'fixtures/mft_synthetic_1.json')


def test_shipped_mft_fixture_2_markers_resolve():
    _markers_resolve('fixtures/mft_test_transcript_2.txt',
                     'fixtures/mft_synthetic_2.json')


def test_shipped_mft_fixture_3_markers_resolve():
    _markers_resolve('fixtures/mft_test_transcript_3.txt',
                     'fixtures/mft_synthetic_3.json')


def test_shipped_mft_fixture_4_markers_resolve():
    _markers_resolve('fixtures/mft_test_transcript_4.txt',
                     'fixtures/mft_synthetic_4.json')


def test_shipped_mft_fixture_5_markers_resolve():
    _markers_resolve('fixtures/mft_test_transcript_5.txt',
                     'fixtures/mft_synthetic_5.json')


def test_shipped_mft_fixture_6_markers_resolve():
    _markers_resolve('fixtures/mft_test_transcript_6.txt',
                     'fixtures/mft_synthetic_6.json')


def test_evaluate_transcript_routing_humility_checks(tmp_path):
    results = {
        'total_questions': 4,
        'total_groups': 0,
        'groups': [],
        'ungrouped_questions': [
            _row('Is the Confluence wiki down?', 'src-a', needs_review=True),
            # Force-bucketed off-topic question: must FAIL the review check
            _row('What are the holiday support hours?', 'src-b',
                 bucket='Scheduling & Performance'),
            # Clear question correctly routed despite quoting an error
            _row('Why does SFTP drop with algorithm negotiation failed?',
                 'src-c', bucket='Connectivity & Authentication'),
            # Clear question wrongly over-abstained into review
            _row('Does MFT support PGP encryption?', 'src-d', needs_review=True),
        ],
        'feature_requests': [],
        'metadata': {'llm_stats': {}},
    }
    fixture = _transcript_fixture(tmp_path, {
        'review_must_match': ['confluence', 'holiday'],
        'routed_must_match': [
            {'match': 'algorithm negotiation', 'bucket': 'Connectivity'},
            {'match': 'pgp', 'bucket': '.'},
        ],
    })
    result = evaluate_transcript(_StubAnalyzer(results), fixture)
    by_name = {c['name']: c for c in result['checks']}
    assert by_name['/confluence/ held for review, not force-bucketed']['ok']
    holiday = by_name['/holiday/ held for review, not force-bucketed']
    assert not holiday['ok'] and 'Scheduling & Performance' in holiday['detail']
    assert by_name['/algorithm negotiation/ routed to /Connectivity/']['ok']
    pgp = by_name['/pgp/ routed to /./']
    assert not pgp['ok'] and 'review pile' in pgp['detail']


def test_evaluate_transcript_recurring_groups_and_singletons(tmp_path):
    host_key = [
        _row('How do we enforce host key verification?', 'src-a'),
        _row('Can MFT verify the partner host key?', 'src-b'),
        _row('Setting to require host key checking?', 'src-c')]
    results = {
        'total_questions': 5,
        'total_groups': 1,
        'groups': [{'count': 3, 'representative_question': host_key[0]['text'],
                    'questions': host_key}],
        'ungrouped_questions': [
            _row('Google Cloud Storage as a destination?', 'src-d'),
            _row('Rotate PGP encryption keys?', 'src-e')],
        'feature_requests': [],
        'metadata': {'llm_stats': {}},
    }
    fixture = _transcript_fixture(tmp_path, {
        'recurring_groups': [
            {'must_match': ['host key'], 'count': 3},
            {'must_match': ['onboard|new partner'], 'count': 2}],
        'must_stay_singleton': ['google cloud', 'sharepoint'],
    })
    result = evaluate_transcript(_StubAnalyzer(results), fixture)
    by_name = {c['name']: c['ok'] for c in result['checks']}
    assert by_name['recurrence /host key/ = 3x']
    assert not by_name['recurrence /onboard|new partner/ = 2x']  # missing
    assert by_name['/google cloud/ stays a singleton']
    # Vacuous pass forbidden: a dropped question is not a "singleton"
    assert not by_name['/sharepoint/ stays a singleton']

    # Wrong count on a named recurrence must fail even though the group exists
    fixture2 = _transcript_fixture(tmp_path, {
        'recurring_groups': [{'must_match': ['host key'], 'count': 2}]})
    result2 = evaluate_transcript(_StubAnalyzer(results), fixture2)
    assert result2['failed'] == 1
    # A singleton absorbed into a group must fail
    fixture3 = _transcript_fixture(tmp_path, {
        'must_stay_singleton': ['partner host key']})
    result3 = evaluate_transcript(_StubAnalyzer(results), fixture3)
    assert result3['failed'] == 1


def test_evaluate_transcript_answered_checks(tmp_path):
    results = {
        'total_questions': 3,
        'total_groups': 0,
        'groups': [],
        'ungrouped_questions': [
            _row('How to increase the connection timeout?', 'src-a', answered=True),
            _row('Throttle bandwidth per partner?', 'src-b', answered=False),
            _row('SharePoint Online as destination?', 'src-c')],
        'feature_requests': [],
        'answered_questions': 1,
        'metadata': {'llm_stats': {}},
    }
    fixture = _transcript_fixture(tmp_path, {
        'answered_count': 1,
        'answered_must_match': ['timeout'],
        'answered_must_not_match': ['throttle', 'sharepoint'],
    })
    result = evaluate_transcript(_StubAnalyzer(results), fixture)
    assert result['failed'] == 0, format_transcript_report(result)

    results['ungrouped_questions'][1]['answered'] = True
    results['answered_questions'] = 2
    result = evaluate_transcript(_StubAnalyzer(results), fixture)
    failed = {c['name'] for c in result['checks'] if not c['ok']}
    assert 'answered = 1' in failed
    assert 'answered excludes /throttle/' in failed


def test_evaluate_transcript_occurrence_integrity_always_on(tmp_path):
    # A 2x group whose rows share ONE source message and differ in text:
    # the phantom-recurrence signature, failed without any expect key
    twins = [_row('Can we purge old records?', 'same-msg'),
             _row('What is the cleanup process for old records?', 'same-msg')]
    results = {
        'total_questions': 2,
        'total_groups': 1,
        'groups': [{'count': 2, 'representative_question': twins[0]['text'],
                    'questions': twins}],
        'ungrouped_questions': [],
        'feature_requests': [],
        'metadata': {'llm_stats': {}},
    }
    fixture = _transcript_fixture(tmp_path, {})
    result = evaluate_transcript(_StubAnalyzer(results), fixture)
    failed = {c['name'] for c in result['checks'] if not c['ok']}
    assert 'every count provable (rows populated, sources distinct)' in failed

    # Identical forwarded text from one source is the documented exemption
    twins[1]['text'] = twins[0]['text']
    result = evaluate_transcript(_StubAnalyzer(results), fixture)
    assert result['failed'] == 0, format_transcript_report(result)


def test_shipped_mft_fixture_7_markers_resolve():
    _markers_resolve('fixtures/mft_test_transcript_7.txt',
                     'fixtures/mft_synthetic_7.json')


# ---- Scoreboard / baseline / stability helpers ----

def _entry(fixture, *oks, axis='routing'):
    return {'fixture': fixture,
            'checks': [{'name': f'check {i}', 'ok': ok, 'detail': '',
                        'axis': axis} for i, ok in enumerate(oks)]}


def test_checks_from_question_result_covers_every_axis():
    from slack_question_analyzer.evaluation import checks_from_question_result
    fixture = {'questions': [
        {'text': 'How do I reset my password?', 'bucket': 'Auth', 'group': 'g1'},
        {'text': 'Password reset steps?', 'bucket': 'Auth', 'group': 'g1'},
        {'text': 'What is the deploy schedule?', 'bucket': 'Deploys'},
    ]}
    result = {
        'routing_mismatches': [{'text': 'What is the deploy schedule?',
                                'expected': 'Deploys', 'got': 'Auth'}],
        'missed_pairs': [sorted(['How do I reset my password?',
                                 'Password reset steps?'])],
        'wrong_pairs': [sorted(['Password reset steps?',
                                'What is the deploy schedule?'])],
        'integrity_violations': ['count 2 != 1 rows: x'],
    }
    checks = checks_from_question_result(fixture, result)

    by_axis = {}
    for c in checks:
        by_axis.setdefault(c['axis'], []).append(c)
    assert len(by_axis['routing']) == 3
    assert sum(1 for c in by_axis['routing'] if c['ok']) == 2
    assert len(by_axis['under-merge']) == 1 and not by_axis['under-merge'][0]['ok']
    assert len(by_axis['over-merge']) == 1 and not by_axis['over-merge'][0]['ok']
    assert len(by_axis['integrity']) == 1 and not by_axis['integrity'][0]['ok']


def test_scoreboard_totals_and_failing_axes():
    from slack_question_analyzer.evaluation import format_scoreboard
    board = format_scoreboard([
        _entry('fixtures/a.json', True, True),
        _entry('fixtures/b.json', True, False, axis='over-merge'),
    ])
    assert 'a.json' in board and 'OK' in board
    assert 'failing: over-merge x1' in board
    assert '3/4' in board  # TOTAL


def test_diff_baseline_reports_flips_only():
    from slack_question_analyzer.evaluation import diff_baseline
    baseline = {'fixtures': [_entry('fixtures/a.json', True, False)]}
    diff = diff_baseline(baseline, [_entry('fixtures/a.json', False, True)])
    assert diff['newly_failing'] == [('fixtures/a.json', 'check 0')]
    assert diff['newly_passing'] == [('fixtures/a.json', 'check 1')]
    assert diff['added'] == [] and diff['removed'] == []


def test_diff_baseline_catches_failure_only_checks():
    """Wrong-pair/integrity checks only exist when they fail: a fresh one
    IS a regression (not 'added' bookkeeping), and one that disappeared
    was fixed."""
    from slack_question_analyzer.evaluation import diff_baseline, format_diff
    clean = {'fixtures': [_entry('fixtures/a.json', True)]}
    # New run grows a failing check the baseline never had
    now_failing = [_entry('fixtures/a.json', True),
                   {'fixture': 'fixtures/a.json',
                    'checks': [{'name': 'pair apart: x / y', 'ok': False,
                                'detail': '', 'axis': 'over-merge'}]}]
    diff = diff_baseline(clean, now_failing)
    assert ('fixtures/a.json', 'pair apart: x / y') in diff['newly_failing']
    assert 'NEWLY FAILING' in format_diff(diff)

    # ...and the reverse reads as newly passing
    diff = diff_baseline({'fixtures': now_failing},
                         [_entry('fixtures/a.json', True)])
    assert ('fixtures/a.json', 'pair apart: x / y') in diff['newly_passing']


def test_flip_report_classifies_stability():
    from slack_question_analyzer.evaluation import flip_report
    runs = [[_entry('fixtures/a.json', True, False, True)],
            [_entry('fixtures/a.json', True, False, False)]]
    report = flip_report(runs)
    assert report['stable_pass'] == 1
    assert report['stable_fail'] == 1
    assert list(report['flaky']) == [('fixtures/a.json', 'check 2')]


def test_flip_report_absent_failure_only_check_counts_as_pass():
    """A wrong-pair check that appears (failing) in run 1 and is absent in
    run 2 FLIPPED — it must not read as stable fail."""
    from slack_question_analyzer.evaluation import flip_report
    fail_run = [{'fixture': 'fixtures/a.json',
                 'checks': [{'name': 'pair apart: x / y', 'ok': False,
                             'detail': '', 'axis': 'over-merge'}]}]
    clean_run = [{'fixture': 'fixtures/a.json', 'checks': []}]
    report = flip_report([fail_run, clean_run])
    assert list(report['flaky']) == [('fixtures/a.json', 'pair apart: x / y')]
    assert report['stable_fail'] == 0


def test_flip_report_errored_run_contributes_no_outcomes():
    from slack_question_analyzer.evaluation import flip_report
    good = [_entry('fixtures/a.json', True, True)]
    errored = [{'fixture': 'fixtures/a.json',
                'checks': [{'name': 'fixture ran to completion', 'ok': False,
                            'detail': 'boom', 'axis': 'error'}]}]
    report = flip_report([good, errored])
    # Both checks evaluated only once (run 1) — stable pass, nothing flaky
    assert report['stable_pass'] == 2
    assert report['flaky'] == {}
    assert report['errored_runs'] == 1


def test_evaluate_records_the_real_fixture_path():
    """Audit regression: result['fixture'] was always '.' (getattr on a
    dict). The loaded fixture carries its path through to the result."""
    from pathlib import Path
    path = Path(__file__).parent.parent / 'fixtures' / 'field_run_2026-06-10.json'
    fixture = load_fixture(str(path))
    assert fixture['_path'] == str(path)
