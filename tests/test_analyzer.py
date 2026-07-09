"""End-to-end tests for the QuestionAnalyzer pipeline with a fake provider."""

import csv
import json

import numpy as np
import pytest

from slack_question_analyzer.analyzer import QuestionAnalyzer

SAMPLE_CONTENT = (
    "2024-01-05\n"
    "How do I reset my password?\n"
    "-----------------------------------------------------------\n"
    "2024-01-08\n"
    "How can I reset my password?\n"
    "-----------------------------------------------------------\n"
    "2024-01-09\n"
    "What is the deploy schedule for production releases?\n"
)


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('GROUP_LABELS', 'off')  # keyword topics only; no LLM in tests
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False)

    # Deterministic fake embeddings: password questions overlap, deploy doesn't
    vectors = {
        'how do i reset my password?': [1.0, 0.0, 0.0],
        'how can i reset my password?': [0.99, 0.05, 0.0],
        'what is the deploy schedule for production releases?': [0.0, 1.0, 0.0],
    }

    def fake_batch(texts, progress_callback=None):
        return np.array([vectors[t] for t in texts])

    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch', fake_batch)
    return analyzer


def test_full_pipeline(analyzer):
    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)

    assert results['total_questions'] == 3
    assert results['total_groups'] == 1
    assert len(results['ungrouped_questions']) == 1

    group = results['groups'][0]
    assert group['count'] == 2
    assert 'password' in group['keywords']
    assert group['topic']  # keyword-derived fallback label
    assert group['date_range'] == {'first_asked': '2024-01-05', 'last_asked': '2024-01-08'}
    assert results['metadata']['provider'] == 'ollama'


ANNOUNCEMENT_MIX = (
    "2024-02-01\n"
    ":rocket: New & Improved Widget Sales Kit is Here!\n"
    "We're excited to share the latest version of the Sales Kit.\n"
    "* :sparkles: Spotlight - enhanced positioning for pitches\n"
    "* :movie_camera: Demo-ready assets - a compelling Elevator Pitch\n"
    "* :open_file_folder: Simplified structure\n"
    "Why should customers care?\n"
    "Explore the new kit here. Happy selling! cc: @sales\n"
    "-----------------------------------------------------------\n"
    "2024-02-02\n"
    "How do I configure retry limits for failed transfers?\n"
)


def _orthogonal_fake_embeddings():
    """Every distinct text gets its own axis: nothing ever groups, and no
    test needs to predict the exact extracted phrasings up front."""
    index = {}

    def fake_batch(texts, progress_callback=None):
        for t in texts:
            index.setdefault(t, len(index))
        dim = max(8, len(index))
        return np.array([[1.0 if j == index[t] else 0.0 for j in range(dim)]
                         for t in texts])
    return fake_batch


def test_announcements_are_context_only(monkeypatch):
    """A broadcast post must produce zero questions — its rhetorical
    headers ('Why should customers care?') are the #1 phantom source."""
    monkeypatch.setenv('GROUP_LABELS', 'off')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False)
    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch',
                        _orthogonal_fake_embeddings())
    results = analyzer.analyze_slack_content(ANNOUNCEMENT_MIX)

    texts = ([q['text'] for g in results['groups'] for q in g['questions']]
             + [q['text'] for q in results['ungrouped_questions']])
    assert any('retry limits' in t for t in texts)
    assert not any('customers care' in t.lower() for t in texts)
    assert results['total_questions'] == 1


def test_announcement_filter_off_restores_old_behavior(monkeypatch):
    monkeypatch.setenv('GROUP_LABELS', 'off')
    monkeypatch.setenv('ANNOUNCEMENT_FILTER', 'off')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False)
    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch',
                        _orthogonal_fake_embeddings())
    results = analyzer.analyze_slack_content(ANNOUNCEMENT_MIX)

    texts = ([q['text'] for g in results['groups'] for q in g['questions']]
             + [q['text'] for q in results['ungrouped_questions']])
    assert any('customers care' in t.lower() for t in texts)


def test_filler_question_marks_do_not_defeat_the_single_ask_cap(analyzer):
    """'Anyone have any thoughts?' is deleted as an extraction, but its '?'
    used to keep the source looking multi-ask — standing down the
    single-ask cap exactly on the messages that needed it."""
    source = ('Our MFT customer will create 8600 scheduled actions and is '
              'worried about threads. Is there any way around this? '
              'Anyone have any thoughts?')
    assert analyzer._ask_mark_count(source) == 1
    assert 'way around this?' in analyzer._ask_sentence(source)

    q1 = {'text': 'Is there any way around running out of threads?',
          'normalized_text': 'is there any way around running out of threads?',
          'original_message': source, 'date': 'Unknown'}
    q2 = {'text': 'Can thread use be improved by scaling vertically?',
          'normalized_text': 'can thread use be improved by scaling vertically?',
          'original_message': source, 'date': 'Unknown'}
    analyzer._dropped = []
    kept = analyzer._enforce_single_ask_cap([q1, q2])
    assert len(kept) == 1  # single real '?' -> one ask survives


def test_so_lead_declarative_is_a_restatement_marker(analyzer):
    """'so customer just need to point at the metering server?' restates
    the message's ask in declarative word order — deterministic collapse.
    'So how do we...?' keeps question word order and is a real follow-up."""
    rx = QuestionAnalyzer._RESTATE_MARKER_RE
    assert rx.match('so customer just need to configure the metering server?')
    assert rx.match('So, the agent comes pre-installed then?')
    assert not rx.match('So how do we configure the metering server?')
    assert not rx.match('so can we also disable the schedule?')
    assert not rx.match('so what happens on failure?')


def test_answer_grounding_rejects_invented_facts():
    """The FAQ draft may only contain facts from the replies: numbers and
    identifier-ish tokens must appear verbatim, content words mostly so."""
    source = ('Set transfer.timeout to 300 in the partner settings and '
              'restart the listener afterwards.')
    ok = QuestionAnalyzer._answer_grounded(
        'Set transfer.timeout to 300 in the partner settings, then restart '
        'the listener.', source)
    assert ok is None  # grounded

    # Invented number
    assert '600' in QuestionAnalyzer._answer_grounded(
        'Set transfer.timeout to 600 in the partner settings.', source)
    # Invented setting name
    assert 'session.max_retries' in QuestionAnalyzer._answer_grounded(
        'Set session.max_retries to 300.', source)
    # Answer mostly from the model's own knowledge
    problem = QuestionAnalyzer._answer_grounded(
        'Enable clustering mode and increase the database connection pool '
        'before contacting support about licensing.', source)
    assert problem is not None


def test_faq_drafts_stored_on_groups(monkeypatch, analyzer):
    """A grounded draft lands on the group; an ungrounded one is discarded
    and the group keeps no draft (the export quotes raw replies instead)."""
    from slack_question_analyzer.group_labeler import GroupLabeler
    if analyzer.labeler is None:
        analyzer.labeler = GroupLabeler('ollama')
    monkeypatch.setattr(analyzer, '_llm_enabled', lambda mode: True)

    def fake_draft(question, replies, validator=None):
        # Simulate the real flow: run the caller's validator on a canned
        # answer, exactly like _generate_json would
        answer = 'Set transfer.timeout to 300 and restart the listener.'
        if validator and validator({'answer': answer}) is not None:
            return None
        return answer
    monkeypatch.setattr(analyzer.labeler, 'draft_answer', fake_draft)

    groups = [{
        'count': 2,
        'representative_question': 'How do I raise the SFTP timeout?',
        'questions': [
            {'text': 'How do I raise the SFTP timeout?', 'answered': True,
             'replies': ['Set transfer.timeout to 300 in the partner '
                         'settings and restart the listener.',
                         'thanks, that worked!']},
            {'text': 'Transfers time out on big files?', 'answered': False,
             'replies': []},
        ],
    }]
    analyzer._draft_faq_answers(groups, lambda *a: None)
    assert groups[0]['draft_answer'].startswith('Set transfer.timeout')

    # Same group, but the replies never mention those facts -> rejected
    groups[0]['questions'][0]['replies'] = ['Ask in the infra channel.']
    del groups[0]['draft_answer']
    analyzer._draft_faq_answers(groups, lambda *a: None)
    assert 'draft_answer' not in groups[0]


def test_enumerated_asks_detected_after_whitespace_collapse():
    """Sources are whitespace-collapsed before the enumeration check, so
    line-start '1.' markers vanish — a numbered SEQUENCE must count as an
    enumeration anyway. Regression: 'They would like to 1. ... 2. ...'
    (single '?') lost item 1 to the single-ask cap."""
    from slack_question_analyzer.textutil import canonical_source
    rx = QuestionAnalyzer._ENUMERATED_ASKS_RE
    enumerated = canonical_source(
        "customer product feedback. They would like to\n"
        "1. See a graphical representation of the Action log.\n"
        "2. Is there a way to group Actions?")
    assert rx.search(enumerated)
    # Version numbers and timestamps must not read as enumerations
    assert not rx.search(canonical_source(
        "How do I upgrade from 10.15 to 12.1 without downtime?"))
    assert not rx.search(canonical_source(
        "The transfer failed at 10.15 AM and again at 11.30 AM, why?"))


def test_date_range_sorts_by_calendar_date():
    # String sort puts "April 9, 2025" before "June 3, 2024" — the range
    # must reflect calendar order, not lexical order
    questions = [{'date': 'June 3, 2024'}, {'date': 'April 9, 2025'}]
    assert QuestionAnalyzer._date_range(questions) == {
        'first_asked': 'June 3, 2024',
        'last_asked': 'April 9, 2025',
    }


def test_empty_result_keeps_provenance(monkeypatch):
    """When every extracted question is dropped, the provenance trail and the
    full result schema must still ship — that is when the audit trail matters
    most."""
    monkeypatch.setenv('GROUP_LABELS', 'off')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False)
    results = analyzer.analyze_slack_content("2024-01-05\nAnyone seen this before?")

    assert results['total_questions'] == 0
    assert results['dropped_questions'], "the dropped filler must be on the trail"
    assert results['dropped_questions'][0]['reason']
    for key in ('threads_present', 'answered_questions', 'themes',
                'executive_summary'):
        assert key in results


def test_internal_audit_flag_never_ships_without_taxonomy(analyzer, monkeypatch):
    """The audit tie marker is consumed by the routing tiebreak — but that
    only runs on the taxonomy path. With taxonomy off it must still be
    stripped before results render."""
    sa = analyzer.similarity_analyzer
    original = sa.group_similar_questions

    def tainted(*args, **kwargs):
        groups = original(*args, **kwargs)
        groups[0]['questions'][0]['_audit_flagged'] = True
        return groups

    monkeypatch.setattr(sa, 'group_similar_questions', tainted)
    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    assert '_audit_flagged' not in json.dumps(results)


def test_threshold_suggestion_when_nothing_groups(monkeypatch):
    """With a too-strict threshold, results carry stats and a suggestion."""
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.95')
    monkeypatch.setenv('GROUP_LABELS', 'off')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False)

    vectors = {
        'how do i reset my password?': [1.0, 0.0, 0.0],
        'how can i reset my password?': [0.9, np.sqrt(1 - 0.81), 0.0],  # sim 0.9 < 0.95
        'what is the deploy schedule for production releases?': [0.0, 0.0, 1.0],
    }
    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: np.array([vectors[t] for t in texts]))

    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    assert results['total_groups'] == 0

    stats = results['metadata']['similarity_stats']
    assert stats['max'] == 0.9
    suggestion = QuestionAnalyzer.suggested_threshold(results)
    assert suggestion == 0.88  # just below the best pair


def test_no_threshold_suggestion_when_groups_exist(analyzer):
    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    assert results['total_groups'] == 1
    assert QuestionAnalyzer.suggested_threshold(results) is None


def test_analyze_contents_merges_multiple_files(analyzer):
    """Messages from several files (e.g. a zipped export) form one corpus."""
    file1 = json.dumps([{'text': 'How do I reset my password?', 'ts': '1704412800.0'}])
    file2 = json.dumps([{'text': 'How can I reset my password?', 'ts': '1704672000.0'}])

    results = analyzer.analyze_contents([file1, file2])
    assert results['total_questions'] == 2
    assert results['total_groups'] == 1  # grouped across file boundaries
    assert results['groups'][0]['count'] == 2


def test_empty_content(analyzer):
    results = analyzer.analyze_slack_content("")
    assert results['total_questions'] == 0
    assert results['groups'] == []
    assert results['ungrouped_questions'] == []


def test_json_export(analyzer, tmp_path):
    output_file = tmp_path / 'results.json'
    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    analyzer.save_results(results, str(output_file))

    saved = json.loads(output_file.read_text(encoding='utf-8'))
    assert saved['total_questions'] == 3


def test_csv_export(analyzer, tmp_path):
    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    output_file = tmp_path / 'results.csv'
    analyzer.export_csv(results, str(output_file))

    with open(output_file, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))

    assert rows[0][0] == 'group_rank'
    # The flat file carries the full second axis: kind/theme/type/answered
    assert rows[0][-4:] == ['kind', 'theme', 'type', 'answered']
    # 2 grouped questions + 1 ungrouped + header
    assert len(rows) == 4
    kinds = {r[7] for r in rows[1:]}
    assert kinds == {'grouped', 'unique'}


def test_markdown_export(analyzer, tmp_path):
    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    output_file = tmp_path / 'results.md'
    analyzer.export_markdown(results, str(output_file))

    report = output_file.read_text(encoding='utf-8')
    assert '# Question Analysis Report' in report
    assert 'asked 2 times' in report
    assert 'Unique Questions (1)' in report
    # No replies in the sample: Answered must be absent, not shown as 0
    assert 'Answered (via thread replies)' not in report


def test_exports_carry_feedback_answered_and_provenance(analyzer, tmp_path):
    """The exports are a frontend too: feedback rows, answered status,
    needs-review flags, and the provenance trail must all survive into
    CSV/Markdown, not just the dashboard."""
    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    results['feature_requests'] = [{'text': 'Add dark mode please.',
                                    'date': '2024-01-05',
                                    'qtype': 'feature-request'}]
    results['dropped_questions'] = [{'text': 'Same ask reworded?',
                                     'date': '2024-01-05', 'source': 'm1',
                                     'reason': 'same-message rephrasing (lexical)'}]
    results['threads_present'] = True
    results['answered_questions'] = 1
    results['groups'][0]['answered'] = 1
    results['ungrouped_questions'][0]['answered'] = False
    results['ungrouped_questions'][0]['needs_review'] = True

    import csv as _csv
    csv_path = tmp_path / 'r.csv'
    analyzer.export_csv(results, str(csv_path))
    with open(csv_path, newline='', encoding='utf-8') as f:
        rows = list(_csv.reader(f))
    kinds = {r[7] for r in rows[1:]}
    assert 'feedback' in kinds and 'needs review' in kinds
    answered_cells = {r[10] for r in rows[1:]}
    assert 'no' in answered_cells  # per-question answered status exported

    md_path = tmp_path / 'r.md'
    analyzer.export_markdown(results, str(md_path))
    report = md_path.read_text(encoding='utf-8')
    assert 'Product Feedback (1)' in report
    assert 'Answered (via thread replies):** 1' in report
    assert 'Answered occurrences: 1' in report
    assert 'Removed During Analysis (1)' in report
    assert 'same-message rephrasing' in report
    assert 'needs review' in report


def test_keywords_contrast_against_corpus(analyzer):
    """Words common to the whole corpus ('customer') characterize nothing;
    group-specific words ('antivirus') must win. Fillers are stopworded."""
    def q(text):
        return {'text': text, 'normalized_text': text.lower()}

    group = [q('Customer just needs antivirus scanning enabled?'),
             q('Customer asks how antivirus quarantine works?')]
    rest = [q('Customer just needs the transfer scheduled?'),
            q('Customer just needs webhook retries configured?')]
    corpus = group + rest

    keywords = analyzer._extract_keywords(
        group, analyzer._corpus_doc_freq(corpus), len(corpus))
    assert keywords[0] == 'antivirus'
    assert 'just' not in keywords and 'needs' not in keywords
    assert keywords[1] != 'customer'  # corpus-wide word can't outrank specifics


def test_same_message_rephrasings_collapse(analyzer):
    """Field regression: the extractor rewrote ONE complaint from two angles
    ('what is the error' / 'why does it fail'), inflating the question count.
    Same message + moderate content-word overlap = one ask. Distinct
    multi-questions from one message share few words and must survive."""
    def q(text, source):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': source}

    questions = [
        q('What is the antivirus scanning error when copying to Target System?', 'm1'),
        q('Why does the Copy Task to Target System fail due to an antivirus scanning error?', 'm1'),
        q('Can we trigger transfers via REST instead of the scheduler?', 'm2'),
        q('Is there a way to bulk-disable actions?', 'm2'),
        # Same text as the m1 question but a DIFFERENT message: a real repeat
        q('What is the antivirus scanning error when copying to Target System?', 'm3'),
    ]
    kept = analyzer._collapse_same_message_rephrasings(questions)
    texts = [k['text'] for k in kept]
    assert len(kept) == 4
    # Exactly ONE of the two m1 rewrites survives (which one is decided by
    # source support, then completeness — not input order)
    m1_texts = [t for t in texts if 'antivirus' in t]
    assert len([k for k in kept if k['original_message'] == 'm1']) == 1
    # The m3 repeat (identical text, different message) always survives
    assert any(k['original_message'] == 'm3' for k in kept)
    assert len(m1_texts) == 2  # one from m1 + the m3 repeat


def test_rephrasing_collapse_keeps_best_supported_phrasing(analyzer):
    """When two rephrasings collapse, the survivor is the one the source
    message vouches for — an extraction that borrowed vocabulary from
    elsewhere (prompt examples, neighbor messages) must not win even when
    it comes first."""
    source = ('Customer wants to bulk-deactivate a set of scheduled actions. '
              'Is there an existing API or UI option to disable many at once?')

    def q(text):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': source}

    contaminated = q('Is there a way to bulk-disable actions in MFT?')
    genuine = q('Is there an existing API or UI option to bulk-deactivate '
                'a set of scheduled actions?')
    kept = analyzer._collapse_same_message_rephrasings([contaminated, genuine])
    assert [k['text'] for k in kept] == [genuine['text']]
    # Provenance records the dropped phrasing
    assert any(d['text'] == contaminated['text'] for d in analyzer._dropped)


def test_template_boilerplate_does_not_collapse_distinct_asks(analyzer):
    """Fixture-4 regression class: two DIFFERENT asks rewritten onto one
    template ('Can we X in wM MFT (SaaS)?' / 'Can we Y in wM MFT (SaaS)?')
    share only filler and product boilerplate — that is zero same-ask
    evidence and both must survive the lexical pass."""
    def q(text):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': 'm-two-things'}

    questions = [
        q('Can we set per-folder retention policies in wM MFT (SaaS)?'),
        q('Can we auto-purge files older than N days in wM MFT (SaaS)?'),
    ]
    kept = analyzer._collapse_same_message_rephrasings(questions)
    assert len(kept) == 2


def test_date_collision_phantom_dropped(analyzer):
    """Invariant: identical text on two dates is illegal unless each copy's
    own source contains it. The backfilled phantom dies; the genuine copy
    and genuine cross-date repeats survive."""
    def q(text, date, source):
        return {'text': text, 'normalized_text': text.lower(), 'date': date,
                'original_message': source}

    custom = 'Can we get a custom error that the script can return?'
    metering = 'How can users check their own transaction statistics?'
    questions = [
        q(custom, 'June 2, 2026', custom),            # genuine: source contains it
        q(custom, 'May 30, 2026', metering),          # phantom: source is a metering msg
        q(metering, 'May 30, 2026', metering),        # genuine
        # genuine cross-date repeat: both sources contain the text
        q('How do I reset my password?', 'June 1, 2026', 'How do I reset my password?'),
        q('How do I reset my password?', 'June 3, 2026', 'How do I reset my password?'),
    ]
    kept = analyzer._enforce_date_integrity(questions)
    dates_for_custom = [k['date'] for k in kept if k['text'] == custom]
    assert dates_for_custom == ['June 2, 2026']  # phantom May 30 copy dropped
    assert sum(1 for k in kept if 'password' in k['text']) == 2  # repeats kept

def test_same_source_rephrases_never_count_as_recurrence(analyzer):
    """Fixture-2 round 4: same-message rephrases that slipped past
    consolidation clustered into a phantom 'asked 2x'. Invariant: within a
    group, one occurrence per source message — unless the texts are
    identical (distinct short messages can share the same text).
    Fixture-4 round 2: the extra row is EJECTED to its own singleton, not
    deleted — it may be a distinct ask the clusterer wrongly merged, and
    deleting it was a silent drop (it took the Answered metric with it)."""
    phantom = {'count': 2, 'bucket': 'File Handling', 'questions': [
        {'text': 'How can I normalize file encoding during transfers?',
         'normalized_text': 'how can i normalize file encoding during transfers?',
         'original_message': 'm4'},
        {'text': 'What is the right way to handle encoding?',
         'normalized_text': 'what is the right way to handle encoding?',
         'original_message': 'm4'},
    ]}
    genuine = {'count': 2, 'questions': [
        {'text': 'How do I reset my password?',
         'normalized_text': 'how do i reset my password?',
         'original_message': 'How do I reset my password?'},
        {'text': 'How do I reset my password?',
         'normalized_text': 'how do i reset my password?',
         'original_message': 'How do I reset my password?'},
    ]}
    cross_message = {'count': 2, 'questions': [
        {'text': 'Limit concurrent transfers per node?',
         'normalized_text': 'limit concurrent transfers per node?',
         'original_message': 'm10'},
        {'text': 'Cap how many transfers run at once per node?',
         'normalized_text': 'cap how many transfers run at once per node?',
         'original_message': 'm11'},
    ]}
    groups = [phantom, genuine, cross_message]
    analyzer._collapse_same_source_occurrences(groups)
    assert phantom['count'] == 1        # not a 2x recurrence anymore
    assert genuine['count'] == 2        # identical-text repeats untouched
    assert cross_message['count'] == 2  # genuine recurrence untouched
    # The m4 message claims no separate asks (no enumeration), so its
    # same-cluster second row is a rephrase: DROPPED with provenance
    assert len(groups) == 3
    assert any(d['text'] == 'What is the right way to handle encoding?'
               for d in analyzer._dropped)


def test_same_source_distinct_asks_eject_when_message_enumerates(analyzer):
    """The T6 class: a 'two things: 1. ... 2. ...' message's asks wrongly
    clustered together must EJECT (both stay on the page), because the
    message itself claims separate asks."""
    source = ('Two things for the setup: 1. Can we set per-folder retention '
              'policies? 2. Can we auto-purge files older than N days?')
    group = {'count': 2, 'bucket': 'Install, Upgrade & Admin', 'questions': [
        {'text': 'Can we set per-folder retention policies?',
         'normalized_text': 'can we set per-folder retention policies?',
         'original_message': source},
        {'text': 'Can we auto-purge files older than N days?',
         'normalized_text': 'can we auto-purge files older than n days?',
         'original_message': source},
    ]}
    groups = [group]
    analyzer._collapse_same_source_occurrences(groups)
    assert group['count'] == 1
    assert len(groups) == 2             # ejected singleton appended
    assert groups[1]['count'] == 1
    assert 'auto-purge' in groups[1]['questions'][0]['text']
    assert groups[1]['bucket'] == 'Install, Upgrade & Admin'
    assert sum(g['count'] for g in groups) == 2  # nothing lost

def test_render_integrity_repairs_unprovable_groups(analyzer):
    """Exit invariant: a group may only render a count it can prove with
    rows. Empty rows are stripped, and a 2x that can't show two distinct
    sources (or identical text throughout) is demoted to singletons."""
    empty_row = {'count': 2, 'representative_question': 'r', 'avg_similarity': 0.9,
                 'questions': [
                     {'text': 'Real question?', 'normalized_text': 'real question?',
                      'original_message': 'm1'},
                     {'text': '', 'normalized_text': '', 'original_message': 'm2'}]}
    phantom = {'count': 2, 'representative_question': 'p', 'avg_similarity': 0.9,
               'questions': [
                   {'text': 'Rewrite one?', 'normalized_text': 'rewrite one?',
                    'original_message': 'm4'},
                   {'text': 'Rewrite two?', 'normalized_text': 'rewrite two?',
                    'original_message': 'm4'}]}
    genuine = {'count': 2, 'representative_question': 'g', 'avg_similarity': 0.9,
               'questions': [
                   {'text': 'Same q?', 'normalized_text': 'same q?',
                    'original_message': 'a'},
                   {'text': 'Same q reworded?', 'normalized_text': 'same q reworded?',
                    'original_message': 'b'}]}
    out = analyzer._enforce_render_integrity([empty_row, phantom, genuine])

    counts = sorted(g['count'] for g in out)
    assert counts == [1, 1, 1, 2]            # phantom demoted to 2 singletons
    assert all(q['text'] for g in out for q in g['questions'])  # no empty rows
    two = next(g for g in out if g['count'] == 2)
    assert two['representative_question'] == 'g'  # only the provable 2x survives


def test_dropped_questions_provenance_in_results(analyzer):
    """Nothing is silently consumed: removed questions become records."""
    content = (
        "2024-01-05\nHow do I reset my password?\n"
        "-----------------------------------------------------------\n"
        "2024-01-09\nWhat is the deploy schedule for production releases?\n"
    )
    results = analyzer.analyze_slack_content(content)
    assert 'dropped_questions' in results
    assert results['dropped_questions'] == []  # nothing dropped on clean input

def test_total_questions_derived_from_rendered_rows(monkeypatch):
    """The 'Questions logged' tile must equal the rows on the page. Two
    surviving same-message rephrases that cluster get collapsed by the exit
    invariant — and the total must reflect that, not the pre-grouping list."""
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('GROUP_LABELS', 'off')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False)
    vectors = {
        'how can we normalize encoding during transfers?': [1.0, 0.0],
        'what is the right way to deal with wrong-coded files?': [0.99, 0.14],
    }
    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: np.array([vectors[t] for t in texts]))

    # One message, two low-overlap rephrases (lexical collapse can't see
    # them, no LLM consolidation with labels off) -> they cluster -> the
    # exit invariant sees same source + same cluster + a message that
    # claims no separate asks: the rephrase is dropped (with provenance)
    # and the total tile equals exactly the rendered rows
    results = analyzer.analyze_slack_content(
        "2024-06-03\nHow can we normalize encoding during transfers? "
        "What is the right way to deal with wrong-coded files?\n")
    rendered = (sum(g['count'] for g in results['groups'])
                + len(results['ungrouped_questions']))
    assert results['total_questions'] == rendered == 1
    assert results['total_groups'] == 0  # no phantom 'asked 2x'
    assert any('rephrase' in d['reason'] for d in results['dropped_questions'])


def test_rephrase_collapse_folds_suffixes(analyzer):
    """Fixture-2 fake 2x: 'retry policy for failed transfers' vs 'a
    transfer that fails ... try again' scored ZERO exact-token overlap and
    survived as a phantom recurrence. Light suffix folding makes
    fails/failed/transfer/transfers shared content."""
    def q(text):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': 'm1'}

    questions = [
        q('How do I set up a retry policy for failed transfers?'),
        q('Can a transfer that fails automatically retry a few times before giving up?'),
    ]
    kept = analyzer._collapse_same_message_rephrasings(questions)
    assert len(kept) == 1


def test_content_free_rhetorical_filler_dropped(analyzer):
    """'Anyone seen this before?' is built from pronouns alone — nothing to
    answer in any context. The models keep leaking these (the two-judge
    consolidation once PROTECTED one), so the extraction prompt's own
    rhetorical list is enforced in code, with provenance."""
    def q(text):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': 'm1', 'date': 'June 8, 2026'}

    questions = [
        q('Could the scheduler be running on the wrong timezone?'),
        q('Anyone seen this before?'),
        q('Any thoughts?'),
        # Content-bearing 'anyone' questions are REAL and survive
        q('Does anyone know if the wiki is down?'),
        q('Has anyone seen 403 errors after the upgrade?'),
    ]
    kept = analyzer._drop_rhetorical_filler(questions)
    texts = [k['text'] for k in kept]
    assert 'Anyone seen this before?' not in texts
    assert 'Any thoughts?' not in texts
    assert 'Does anyone know if the wiki is down?' in texts
    assert 'Has anyone seen 403 errors after the upgrade?' in texts
    assert len(kept) == 3
    dropped = [d['text'] for d in analyzer._dropped]
    assert 'Anyone seen this before?' in dropped


def test_restatement_marker_collapses_zero_overlap_rephrase(analyzer):
    """'I mean is there a built-in way to gzip the payload?' shares no
    content words with 'Can we compress files before sending?' — but the
    leading marker IS the same-ask evidence, said by the asker themselves."""
    def q(text):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': 'm6'}

    questions = [
        q('Can we compress files before sending?'),
        q('I mean is there a built-in way to gzip or zip the payload prior to transfer?'),
    ]
    kept = analyzer._collapse_same_message_rephrasings(questions)
    assert len(kept) == 1

    # The marker only binds within ONE message: across messages it's inert
    cross = [q('Can we compress files before sending?'),
             {'text': 'Basically can transfers resume after an interruption?',
              'normalized_text': 'basically can transfers resume after an interruption?',
              'original_message': 'OTHER'}]
    assert len(analyzer._collapse_same_message_rephrasings(cross)) == 2


def test_single_ask_cap_on_unenumerated_single_question_mark_message(analyzer):
    """Invariant: an unenumerated message with at most one '?' asks at most
    one question — a second extraction is the model rewriting context into
    an extra ask. Enumerated messages and multi-'?' messages are exempt."""
    capped_source = ('Customer on 10.15.2 wants to know if MFT can handle '
                     'files approx. 5GB in size. Is there a max size?')
    enum_source = ('Two things: 1. Can we set retention policies? '
                   '2. Can we auto-purge old files?')
    multi_q_source = 'Can we do X? And how do we configure Y?'

    def q(text, source):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': source}

    questions = [
        q('Is there a max single-transfer size limit?', capped_source),
        q('Does MFT handle very large payloads approx. 5GB?', capped_source),
        q('Can we set retention policies?', enum_source),
        q('Can we auto-purge old files?', enum_source),
        q('Can we do X?', multi_q_source),
        q('How do we configure Y?', multi_q_source),
    ]
    kept = analyzer._enforce_single_ask_cap(questions)
    texts = [k['text'] for k in kept]
    # Capped message: one survivor (best source support wins)
    assert sum(1 for k in kept if k['original_message'] == capped_source) == 1
    # Enumerated and multi-'?' messages keep both asks
    assert 'Can we set retention policies?' in texts
    assert 'Can we auto-purge old files?' in texts
    assert 'Can we do X?' in texts and 'How do we configure Y?' in texts
    assert any('extra extraction' in d['reason'] for d in analyzer._dropped)

    # Truncated sources (at the identity cap) are exempt: clipping can hide
    # the '?'s and enumeration markers that prove multiple asks
    from slack_question_analyzer.textutil import SOURCE_KEY_LEN
    long_source = ('x' * (SOURCE_KEY_LEN - 1) + 'y')[:SOURCE_KEY_LEN]
    pair = [q('First ask?', long_source), q('Second ask?', long_source)]
    assert len(analyzer._enforce_single_ask_cap(pair)) == 2


def test_single_ask_cap_keeps_the_question_sentence_rewrite(analyzer):
    """The lone '?' marks the asker's actual question: when both candidates
    are verbatim-supported by the whole message, the survivor must rewrite
    the '?'-sentence, not the surrounding context."""
    source = ('How do I increase the SFTP connection timeout? The customer '
              'large transfers keep timing out before they finish.')

    def q(text):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': source}

    questions = [
        q('Why are the customer large transfers timing out before they finish?'),
        q('How do I increase the SFTP connection timeout?'),
    ]
    kept = analyzer._enforce_single_ask_cap(questions)
    assert len(kept) == 1
    assert 'increase the SFTP connection timeout' in kept[0]['text']


def test_enumerated_siblings_locked_separate_from_all_collapse_passes(analyzer, monkeypatch):
    """PRECEDENCE RULE (fixture 7): a message that explicitly enumerates
    separate asks had its split decided at extraction, on the asker's own
    words — no collapse pass may merge the siblings. Consolidation once
    deleted 'max retry count' as a 'rephrasing' of its enumerated sibling."""
    source = ('Two things for wM MFT (SaaS): 1. can we set a max retry count '
              'per transfer? 2. can we tag transfers with a custom label?')

    def q(text):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': source}

    siblings = [q('Can we set a max retry count per transfer?'),
                q('Can we tag transfers with a custom label for reporting?')]
    # Lexical collapse: locked
    assert len(analyzer._collapse_same_message_rephrasings(list(siblings))) == 2
    # Single-ask cap: exempt (enumerated, multiple '?')
    assert len(analyzer._enforce_single_ask_cap(list(siblings))) == 2
    # LLM consolidation: never even consulted for enumerated messages
    analyzer_llm_called = []
    if analyzer.labeler is None:
        from slack_question_analyzer.group_labeler import GroupLabeler
        analyzer.labeler = GroupLabeler('ollama')
    monkeypatch.setattr(analyzer.labeler, 'available', lambda: True)
    monkeypatch.setattr(analyzer.labeler, 'consolidate_same_ask',
                        lambda msg, texts: analyzer_llm_called.append(msg) or [1])
    monkeypatch.setattr(analyzer.labeler, 'verify_same_topic', lambda a, b: True)
    kept = analyzer._consolidate_same_ask(list(siblings))
    assert len(kept) == 2
    assert analyzer_llm_called == []


def test_topic_label_must_be_grounded_in_member_text(analyzer):
    """A group of failure-alert questions was once labeled 'Transfer
    Retries' — words its members never said. Labels describe, never invent."""
    group = {'questions': [
        {'text': 'How do we set up alerting when a transfer fails?'},
        {'text': 'What is the way to get an alert on transfer failures?'},
    ]}
    assert analyzer._topic_grounded('Transfer Failure Alerting', group)
    assert analyzer._topic_grounded('Failure Alerts', group)
    assert not analyzer._topic_grounded('Transfer Retries', group)
    assert not analyzer._topic_grounded('', group)


def test_routing_tiebreak_splits_contested_cross_category_member(analyzer, monkeypatch):
    """Third-judge rule: audit said different (1), verifier said same (1) —
    confident routing disagreement breaks the tie and splits the member.
    Same-category disagreement or unconfident routes leave the group alone."""
    vectors = {
        'how do we enforce host key verification?': [1.0, 0.0, 0.0],
        'how do we onboard a new vendor end to end?': [0.0, 1.0, 0.0],
    }
    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: np.array(
                            [vectors[t] for t in texts]))
    anchors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def q(text, flagged=False):
        d = {'text': text, 'normalized_text': text.lower(),
             'original_message': text}
        if flagged:
            d['_audit_flagged'] = True
        return d

    contested = q('How do we onboard a new vendor end to end?', flagged=True)
    group = {'representative_question': 'How do we enforce host key verification?',
             'questions': [q('How do we enforce host key verification?'),
                           contested],
             'count': 2, 'avg_similarity': 0.82}
    out = analyzer._routing_tiebreak([group], anchors)
    counts = sorted(g['count'] for g in out)
    assert counts == [1, 1]              # contested member split out
    assert '_audit_flagged' not in contested  # internal flag never leaks

    # Same category on both sides: the tie stands, group survives intact
    vectors['how do we onboard a new vendor end to end?'] = [0.95, 0.2, 0.0]
    contested2 = q('How do we onboard a new vendor end to end?', flagged=True)
    group2 = {'representative_question': 'How do we enforce host key verification?',
              'questions': [q('How do we enforce host key verification?'),
                            contested2],
              'count': 2, 'avg_similarity': 0.82}
    out2 = analyzer._routing_tiebreak([group2], anchors)
    assert [g['count'] for g in out2] == [2]


def test_enumeration_requires_enumeration_position(analyzer):
    """'(synthetic test set 2) ' matched the bare \\d[.)] pattern and faked
    an asker-declared split for a whole message. Digit enumeration must
    appear at message start or after a colon/semicolon."""
    R = analyzer._ENUMERATED_ASKS_RE
    assert not R.search('Header (test set 2) How do I set up a retry policy?')
    assert not R.search('Upgraded to 10.15. Transfers fail now.')
    assert R.search('Two things: 1. retry count? 2. custom labels?')
    assert R.search('1. Can we restrict IP ranges? 2. Maintenance windows?')
    assert R.search('does it support TLS, and separately, Kafka events?')


def test_same_source_fragments_are_dropped_with_provenance(analyzer):
    """A fragment contained in a longer extraction from the same message is
    never a second ask — it used to rank and route as one."""
    source = ('Is there a way to configure the retry count for outbound '
              'transfers to partner endpoints?')

    def q(text):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': source, 'date': 'Unknown'}

    fragment = q('Is there a way to configure the retry')
    whole = q('Is there a way to configure the retry count for outbound '
              'transfers to partner endpoints?')
    analyzer._dropped = []
    kept = analyzer._drop_same_source_fragments([fragment, whole])
    assert kept == [whole]
    assert analyzer._dropped and 'fragment' in analyzer._dropped[0]['reason']


def test_fragment_guard_leaves_enumerated_and_distinct_asks_alone(analyzer):
    enumerated = ('Two things: 1. can we set a retry count? '
                  '2. can we set a retry count per partner?')

    def q(text, source):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': source, 'date': 'Unknown'}

    # Enumerated siblings are locked even when one contains the other
    siblings = [q('Can we set a retry count?', enumerated),
                q('Can we set a retry count per partner?', enumerated)]
    analyzer._dropped = []
    assert len(analyzer._drop_same_source_fragments(list(siblings))) == 2
    # Distinct (non-substring) asks from one message are untouched
    plain = 'Does it support TLS? Can it post events to Kafka?'
    distinct = [q('Does it support TLS?', plain),
                q('Can it post events to Kafka?', plain)]
    assert len(analyzer._drop_same_source_fragments(list(distinct))) == 2


def test_fragment_guard_spares_separately_punctuated_asks(analyzer):
    """'Can we retry? ... can we retry automatically on a schedule?' is two
    real asks — the guard once dropped the first as a 'fragment'."""
    source = ('Can we retry failed transfers? And more importantly can we '
              'retry failed transfers automatically on a schedule?')

    def q(text):
        return {'text': text, 'normalized_text': text.lower(),
                'original_message': source, 'date': 'Unknown'}

    asks = [q('Can we retry failed transfers?'),
            q('Can we retry failed transfers automatically on a schedule?')]
    analyzer._dropped = []
    assert len(analyzer._drop_same_source_fragments(list(asks))) == 2


def test_interjection_prefixed_filler_still_drops(analyzer):
    analyzer._dropped = []
    kept = analyzer._drop_rhetorical_filler([
        {'text': 'Ugh, Mondays, right?', 'normalized_text': 'ugh mondays right?',
         'original_message': 'Ugh, Mondays, right?', 'date': 'Unknown'}])
    assert kept == []
    assert analyzer._dropped


def test_equal_count_groups_rank_by_recency(monkeypatch):
    """A 2x asked this week outranks a 2x from months ago — ties must not
    break on embedding cohesion."""
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('GROUP_LABELS', 'off')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False)

    vectors = {
        'how do i reset my password?': [1.0, 0.0, 0.0],
        'how can i reset my password?': [0.99, 0.05, 0.0],
        'how do i rotate the sftp certificate?': [0.0, 1.0, 0.0],
        'how can i rotate the sftp certificate?': [0.05, 0.99, 0.0],
    }
    monkeypatch.setattr(
        analyzer.similarity_analyzer, 'get_embeddings_batch',
        lambda texts, progress_callback=None: np.array([vectors[t] for t in texts]))

    content = (
        "2024-01-05\nHow do I reset my password?\n"
        "-----------------------------------------------------------\n"
        "2024-01-06\nHow can I reset my password?\n"
        "-----------------------------------------------------------\n"
        "2024-03-01\nHow do I rotate the SFTP certificate?\n"
        "-----------------------------------------------------------\n"
        "2024-03-02\nHow can I rotate the SFTP certificate?\n"
    )
    results = analyzer.analyze_slack_content(content)
    assert [g['count'] for g in results['groups']] == [2, 2]
    # Certificate group (last asked Mar 2) outranks password (last Jan 6)
    assert 'certificate' in results['groups'][0]['representative_question'].lower()


def test_source_support_matches_token_stems_not_substrings():
    """'port' must not match inside 'support' — this score gates source
    verification, survivor ranking, and date integrity."""
    support = QuestionAnalyzer._source_support
    assert support('can we use port forwarding',
                   'we support many transfer modes') == 0.0
    # Genuine rewrites still score high, inflections included
    assert support('why do transfers keep failing',
                   'our transfer failed again this morning') > 0.6


def test_flagged_minority_splits_as_one_group(analyzer, monkeypatch):
    """Field case (fixture 7): a timeout group swallowed a 3-member alerts
    family; the audit flagged each alert, the verifier overruled each 1-1,
    and per-member tiebreaks never fired because the MIXED group's rep
    routes ambiguously. A coherent flagged minority that confidently routes
    to a different bucket splits out AS A UNIT — counts intact."""
    vectors = {
        'what setting controls the transfer timeout?': [1.0, 0.0, 0.0],
        'how do i extend the timeout window?': [0.98, 0.1, 0.0],
        'how do i get an alert when a transfer fails?': [0.0, 1.0, 0.0],
        'how do we configure failure alerts?': [0.1, 0.98, 0.0],
    }
    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: np.array(
                            [vectors[t] for t in texts]))
    anchors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def q(text, flagged=False):
        d = {'text': text, 'normalized_text': text.lower(),
             'original_message': text}
        if flagged:
            d['_audit_flagged'] = True
        return d

    group = {'representative_question': 'What setting controls the transfer timeout?',
             'questions': [
                 q('What setting controls the transfer timeout?'),
                 q('How do I extend the timeout window?'),
                 q('How do I get an alert when a transfer fails?', flagged=True),
                 q('How do we configure failure alerts?', flagged=True),
             ],
             'count': 4, 'avg_similarity': 0.79}
    out = analyzer._routing_tiebreak([group], anchors)
    assert sorted(g['count'] for g in out) == [2, 2]
    alert_group = next(g for g in out if 'alert' in g['representative_question'].lower())
    assert len(alert_group['questions']) == 2


def test_flagged_minority_routing_same_bucket_stays_whole(analyzer, monkeypatch):
    """Control: when the flagged minority routes to the SAME bucket as the
    rest, the tie stands and the group survives intact."""
    vectors = {
        'what setting controls the transfer timeout?': [1.0, 0.0, 0.0],
        'how do i extend the timeout window?': [0.98, 0.1, 0.0],
        'why do transfers abort after five minutes?': [0.97, 0.05, 0.1],
        'what limits how long a transfer may run?': [0.96, 0.05, 0.15],
    }
    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: np.array(
                            [vectors[t] for t in texts]))
    anchors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def q(text, flagged=False):
        d = {'text': text, 'normalized_text': text.lower(),
             'original_message': text}
        if flagged:
            d['_audit_flagged'] = True
        return d

    group = {'representative_question': 'What setting controls the transfer timeout?',
             'questions': [
                 q('What setting controls the transfer timeout?'),
                 q('How do I extend the timeout window?'),
                 q('Why do transfers abort after five minutes?', flagged=True),
                 q('What limits how long a transfer may run?', flagged=True),
             ],
             'count': 4, 'avg_similarity': 0.9}
    out = analyzer._routing_tiebreak([group], anchors)
    assert [g['count'] for g in out] == [4]


def test_minority_split_fires_even_when_the_rep_routes_ambiguously(analyzer, monkeypatch):
    """The round-3 finding: a mixed group's rep sits within the ambiguity
    margin of some THIRD anchor, so a global-confidence tiebreak never
    fires. Mutual preference — each side prefers its own bucket over the
    other side's — is the real disagreement evidence."""
    vectors = {
        # Rep: prefers anchor0 but within 0.05 of anchor2 -> globally ambiguous
        'what setting controls the transfer timeout?': [0.70, 0.10, 0.68, 0.2],
        'how do i get an alert when a transfer fails?': [0.10, 0.95, 0.10, 0.0],
        'how do we configure failure alerts?': [0.12, 0.93, 0.12, 0.0],
        'how do i extend the timeout window?': [0.69, 0.1, 0.67, 0.25],
    }
    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: np.array(
                            [vectors[t] for t in texts]))
    anchors = np.array([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]])

    def q(text, flagged=False):
        d = {'text': text, 'normalized_text': text.lower(),
             'original_message': text}
        if flagged:
            d['_audit_flagged'] = True
        return d

    group = {'representative_question': 'What setting controls the transfer timeout?',
             'questions': [
                 q('What setting controls the transfer timeout?'),
                 q('How do I extend the timeout window?'),
                 q('How do I get an alert when a transfer fails?', flagged=True),
                 q('How do we configure failure alerts?', flagged=True),
             ],
             'count': 4, 'avg_similarity': 0.79}
    out = analyzer._routing_tiebreak([group], anchors)
    assert sorted(g['count'] for g in out) == [2, 2]


def test_answer_grounding_tolerates_wording_artifacts():
    """Formatting differences are not fact differences: '1,000' grounds
    '1000', Latin abbreviations pass, and hyphenated plain words ('built-in',
    're-enable') ground through their parts — while invented identifiers and
    numbers are still rejected."""
    source = ('Raise the queue limit to 1,000 in the admin console. The '
              'retry feature is built in; you can re-enable it after a '
              'restart from the settings page.')
    ok = QuestionAnalyzer._answer_grounded(
        'Raise the queue limit to 1000 in the admin console. The retry '
        'feature is built-in — re-enable it from the settings page, '
        'e.g. after a restart.', source)
    assert ok is None  # grounded despite comma / hyphen / e.g. artifacts

    # Real identifiers and numbers still must match the source
    assert 'server.xml' in QuestionAnalyzer._answer_grounded(
        'Edit server.xml in the admin console to raise the queue limit.',
        source)
    assert '5000' in QuestionAnalyzer._answer_grounded(
        'Raise the queue limit to 5000 in the admin console.', source)


def test_faq_drafting_skips_groups_with_curated_answers(monkeypatch, analyzer):
    """A human-approved answer carried from the bank is canonical — the
    drafting pass must not spend an LLM call replacing it."""
    from slack_question_analyzer.group_labeler import GroupLabeler
    if analyzer.labeler is None:
        analyzer.labeler = GroupLabeler('ollama')
    monkeypatch.setattr(analyzer, '_llm_enabled', lambda mode: True)

    def never(*args, **kwargs):
        raise AssertionError('curated groups must not be re-drafted')
    monkeypatch.setattr(analyzer.labeler, 'draft_answer', never)

    groups = [{
        'count': 2,
        'curated_answer': 'Set transfer.timeout to 300.',
        'representative_question': 'How do I raise the SFTP timeout?',
        'questions': [
            {'text': 'How do I raise the SFTP timeout?', 'answered': True,
             'replies': ['Set transfer.timeout to 300.']},
        ],
    }]
    analyzer._draft_faq_answers(groups, lambda *a: None)
    assert 'draft_answer' not in groups[0]
