"""Tests for the learned topic bank."""

import json

import numpy as np

from slack_question_analyzer.topic_bank import TopicBank
from slack_question_analyzer.analyzer import QuestionAnalyzer


def make_group(topic='Password Reset', count=3):
    return {
        'topic': topic,
        'summary': 'People ask how to reset passwords.',
        'representative_question': 'How do I reset my password?',
        'keywords': ['password', 'reset'],
        'count': count,
    }


def test_record_and_match_roundtrip(tmp_path):
    path = tmp_path / 'bank.json'
    bank = TopicBank(path=str(path))
    bank.record(make_group(), [1.0, 0.0])
    bank.save()

    reloaded = TopicBank(path=str(path))
    hit = reloaded.match([0.95, 0.31], threshold=0.85)  # ~0.95 similarity
    assert hit is not None
    assert hit['topic'] == 'Password Reset'

    assert reloaded.match([0.0, 1.0], threshold=0.85) is None  # orthogonal


def test_matched_entries_accumulate_history(tmp_path):
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    entry = bank.record(make_group(count=3), [1.0, 0.0])
    updated = bank.record(make_group(topic='ignored new name', count=2),
                          [1.0, 0.0], matched=entry)

    assert updated is entry
    assert entry['topic'] == 'Password Reset'  # established name kept
    assert entry['question_count'] == 5
    assert entry['analysis_count'] == 2


def test_entries_never_match_across_embedding_models(tmp_path):
    """Same dimensions, different model: vectors live in different spaces."""
    path = tmp_path / 'bank.json'
    old = TopicBank(path=str(path), model='old-model')
    old.record(make_group(), [1.0, 0.0])
    old.save()

    new = TopicBank(path=str(path), model='new-model')
    assert new.match([1.0, 0.0], threshold=0.5) is None
    # Same model still matches
    same = TopicBank(path=str(path), model='old-model')
    assert same.match([1.0, 0.0], threshold=0.5) is not None


def test_concurrent_bank_instances_merge_on_save(tmp_path):
    """Two instances writing the same bank keep each other's topics."""
    path = str(tmp_path / 'bank.json')
    first = TopicBank(path=path, model='m')
    second = TopicBank(path=path, model='m')  # loaded before first saves

    first.record(make_group(topic='Virus Scanning'), [1.0, 0.0])
    first.save()
    second.record(make_group(topic='SFTP Connections'), [0.0, 1.0])
    second.save()

    reloaded = TopicBank(path=path, model='m')
    names = {e['topic'] for e in reloaded.entries}
    assert names == {'Virus Scanning', 'SFTP Connections'}


def test_deleted_topics_stay_deleted_after_merge_on_save(tmp_path):
    path = str(tmp_path / 'bank.json')
    bank = TopicBank(path=path, model='m')
    entry = bank.record(make_group(), [1.0, 0.0])
    bank.save()

    assert bank.delete(entry['id']) is True
    bank.save()  # merge-on-save must not resurrect the tombstoned entry
    assert TopicBank(path=path).entries == []


def test_delete_and_merge(tmp_path):
    bank = TopicBank(path=str(tmp_path / 'bank.json'), model='m')
    keep = bank.record(make_group(topic='Virus Scanning', count=4), [1.0, 0.0])
    junk = bank.record(make_group(topic='Scanning Stuff', count=2), [0.95, 0.31])

    assert bank.merge(junk['id'], keep['id']) is True
    assert len(bank.entries) == 1
    assert keep['topic'] == 'Virus Scanning'
    assert keep['question_count'] == 6

    assert bank.delete(keep['id']) is True
    assert bank.entries == []
    assert bank.delete('nonexistent') is False
    assert bank.merge('a', 'b') is False


def test_keyword_fallback_topics_are_not_banked(monkeypatch):
    """Junk keyword names must never stick in the bank."""
    analyzer = make_analyzer(monkeypatch)
    monkeypatch.setattr(analyzer.labeler, 'label_group',
                        lambda texts, keywords=None: None)  # LLM gives nothing

    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    group = results['groups'][0]
    assert group['topic']  # keyword fallback name shown in this analysis...
    assert 'topic_id' not in group  # ...but not learned
    assert TopicBank().entries == []


def test_bank_match_floor_ignores_loose_thresholds(monkeypatch):
    """Auto-adjusted low grouping thresholds can't cause loose bank matches."""
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.5')  # very loose grouping
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')
    monkeypatch.setenv('LLM_EXTRACTION', 'off')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False,
                                label_groups=True)

    # Bank knows a topic only ~0.7 similar to the new group: below the 0.8 floor
    bank = TopicBank(model='test-embed')
    bank.record({'topic': 'Unrelated Topic', 'summary': None,
                 'representative_question': 'x?', 'keywords': [], 'count': 2},
                [0.7, float(np.sqrt(1 - 0.49))])
    bank.save()

    def fake_batch(texts, progress_callback=None):
        for t in texts:
            analyzer.similarity_analyzer.embeddings_cache.set(t, VECTORS[t])
        return np.array([VECTORS[t] for t in texts])

    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch', fake_batch)
    monkeypatch.setattr(analyzer.labeler, 'available', lambda: True)
    monkeypatch.setattr(analyzer.labeler, 'verify_same_topic', lambda a, b: None)
    monkeypatch.setattr(analyzer.labeler, 'summarize_analysis', lambda g, t, themes=None: None)
    monkeypatch.setattr(analyzer.labeler, 'label_group',
                        lambda texts, keywords=None: {'topic': 'Password Reset',
                                                      'summary': 's'})

    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    # The loose bank entry must NOT have claimed the group
    assert results['groups'][0]['topic'] == 'Password Reset'


def test_dimension_mismatch_entries_are_skipped(tmp_path):
    """Old entries from a different embedding model can't poison matches."""
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    bank.record(make_group(), [1.0, 0.0, 0.0])  # 3-dim entry
    assert bank.match([1.0, 0.0], threshold=0.5) is None  # 2-dim query


def test_disabled_bank_is_inert(tmp_path):
    path = tmp_path / 'bank.json'
    bank = TopicBank(path=str(path), enabled=False)
    assert bank.record(make_group(), [1.0, 0.0]) is None
    bank.save()
    assert not path.exists()


# ---- Pipeline integration: labels stay stable across analyses ----

SAMPLE_CONTENT = (
    "2024-01-05\nHow do I reset my password?\n"
    "-----------------------------------------------------------\n"
    "2024-01-08\nHow can I reset my password?\n"
)

VECTORS = {
    'how do i reset my password?': [1.0, 0.0],
    'how can i reset my password?': [0.99, np.sqrt(1 - 0.99 ** 2)],
}


def make_analyzer(monkeypatch):
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')  # no embed prefix
    monkeypatch.setenv('LLM_EXTRACTION', 'off')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False,
                                label_groups=True)

    def fake_batch(texts, progress_callback=None):
        # Populate the in-memory cache so group centroids can be computed
        for t in texts:
            analyzer.similarity_analyzer.embeddings_cache.set(t, VECTORS[t])
        return np.array([VECTORS[t] for t in texts])

    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch', fake_batch)
    monkeypatch.setattr(analyzer.labeler, 'available', lambda: True)
    monkeypatch.setattr(analyzer.labeler, 'verify_same_topic', lambda a, b: None)
    monkeypatch.setattr(analyzer.labeler, 'summarize_analysis', lambda g, t, themes=None: None)
    return analyzer


def test_bank_keeps_topic_names_stable_across_analyses(monkeypatch):
    # First analysis: the LLM names the group; the bank learns it
    first = make_analyzer(monkeypatch)
    monkeypatch.setattr(first.labeler, 'label_group',
                        lambda texts, keywords=None: {'topic': 'Password Reset',
                                                      'summary': 'Resets.'})
    results1 = first.analyze_slack_content(SAMPLE_CONTENT)
    assert results1['groups'][0]['topic'] == 'Password Reset'
    assert results1['groups'][0]['seen_in_analyses'] == 1

    # Second analysis (fresh instance): the bank labels it — the LLM must
    # not be asked again, and the name must be identical
    second = make_analyzer(monkeypatch)

    def never(texts, keywords=None):
        raise AssertionError('bank should have labeled this group')

    monkeypatch.setattr(second.labeler, 'label_group', never)
    results2 = second.analyze_slack_content(SAMPLE_CONTENT)
    group = results2['groups'][0]
    assert group['topic'] == 'Password Reset'
    # The second run analyzed the IDENTICAL transcript: every occurrence is
    # already fingerprinted, so the 'recurring xN' evidence must NOT inflate
    # (re-analyzing old data is not a recurrence)
    assert group['seen_in_analyses'] == 1


def test_seeding_pre_loads_curated_topics(monkeypatch, tmp_path):
    """An empty bank is pre-loaded from seed_topics.json on first analysis."""
    seed_file = tmp_path / 'seeds.json'
    seed_file.write_text(json.dumps([
        {'topic': 'Password Reset', 'question': 'How do I reset my password?'},
    ]), encoding='utf-8')
    monkeypatch.setenv('SEED_TOPICS_PATH', str(seed_file))

    analyzer = make_analyzer(monkeypatch)

    def never(texts, keywords=None):
        raise AssertionError('the seeded bank should label this group')

    monkeypatch.setattr(analyzer.labeler, 'label_group', never)
    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)

    group = results['groups'][0]
    assert group['topic'] == 'Password Reset'  # named by the seed, not the LLM
    assert group['seen_in_analyses'] == 1      # first real sighting
    assert group['topic_id']

    # Bank now has the seed entry only (matched, not duplicated)
    bank = TopicBank()
    assert len(bank.entries) == 1
    assert bank.entries[0]['question_count'] == 2


def test_seeding_runs_once(monkeypatch, tmp_path):
    seed_file = tmp_path / 'seeds.json'
    seed_file.write_text(json.dumps([
        {'topic': 'Password Reset', 'question': 'How do I reset my password?'},
    ]), encoding='utf-8')
    monkeypatch.setenv('SEED_TOPICS_PATH', str(seed_file))

    for _ in range(2):
        analyzer = make_analyzer(monkeypatch)
        monkeypatch.setattr(analyzer.labeler, 'label_group',
                            lambda texts, keywords=None: None)
        analyzer.analyze_slack_content(SAMPLE_CONTENT)

    assert len(TopicBank().entries) == 1  # no duplicate seeds


def test_repo_seed_file_is_valid():
    """The shipped seed file: valid JSON, unique questions, named topics."""
    import json as json_module
    from pathlib import Path
    seeds = json_module.loads(Path('seed_topics.json').read_text(encoding='utf-8'))
    assert len(seeds) == 150
    questions = [s['question'] for s in seeds]
    assert len(set(questions)) == len(questions)
    for seed in seeds:
        assert seed['topic'].strip()
        assert seed['question'].strip().endswith('?')


def test_rename_updates_bank(tmp_path):
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    entry = bank.record(make_group(topic='Bad Name'), [1.0, 0.0])

    assert bank.rename(entry['id'], 'Virus Scanning') is True
    assert bank.rename('nonexistent', 'X') is False

    reloaded = TopicBank(path=str(tmp_path / 'bank.json'))
    assert reloaded.entries[0]['topic'] == 'Virus Scanning'


def _group_with_questions(texts_dates, topic='SFTP Timeouts'):
    return {
        'topic': topic,
        'summary': 'Timeout questions.',
        'representative_question': texts_dates[0][0],
        'keywords': ['sftp', 'timeout'],
        'count': len(texts_dates),
        'questions': [{'normalized_text': t, 'text': t, 'date': d}
                      for t, d in texts_dates],
    }


def test_overlapping_reupload_does_not_inflate_counts(tmp_path):
    """Regression: 'export the last 90 days' every month re-uploads most of
    last month's messages. Occurrences are fingerprinted (text+date), so an
    already-recorded question must not bump question_count, analysis_count,
    or re-blend the centroid."""
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    g1 = _group_with_questions([('how do i raise the sftp timeout?', 'June 3, 2026'),
                                ('sftp transfers time out on big files?', 'June 9, 2026')])
    entry = bank.record(g1, [1.0, 0.0])
    assert entry['question_count'] == 2 and entry['analysis_count'] == 1

    # Full overlap: the identical export uploaded again
    again = bank.record(g1, [1.0, 0.0], matched=entry)
    assert again['question_count'] == 2      # unchanged
    assert again['analysis_count'] == 1      # no new evidence -> no bump

    # Partial overlap: one old occurrence + one genuinely new one
    g2 = _group_with_questions([('sftp transfers time out on big files?', 'June 9, 2026'),
                                ('what controls the sftp wait time?', 'July 1, 2026')])
    updated = bank.record(g2, [1.0, 0.0], matched=entry)
    assert updated['question_count'] == 3    # only the new one counted
    assert updated['analysis_count'] == 2


def test_merge_unions_fingerprints(tmp_path):
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    a = bank.record(_group_with_questions([('question a?', 'June 1, 2026')],
                                          topic='A'), [1.0, 0.0])
    b = bank.record(_group_with_questions([('question b?', 'June 2, 2026')],
                                          topic='B'), [0.0, 1.0])
    assert bank.merge(a['id'], b['id']) is True
    target = bank._find(b['id'])
    assert len(target['seen']) == 2
    # Re-upload of the merged-away side's transcript is still already-seen
    re_up = bank.record(_group_with_questions([('question a?', 'June 1, 2026')],
                                              topic='A'), [1.0, 0.0],
                        matched=target)
    assert re_up['question_count'] == target['question_count'] == 2


def test_legacy_groups_without_questions_keep_old_counting(tmp_path):
    """Question-less groups (older callers, question-level fixtures) can't be
    fingerprinted — they keep the historical count-everything behavior."""
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    entry = bank.record(make_group(count=3), [1.0, 0.0])
    updated = bank.record(make_group(count=2), [1.0, 0.0], matched=entry)
    assert updated['question_count'] == 5
    assert updated['analysis_count'] == 2


def test_set_published_stamps_and_clears(tmp_path):
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    entry = bank.record(make_group(), [1.0, 0.0])
    stamped = bank.set_published(entry['id'], True)
    assert stamped and len(stamped) == 10  # YYYY-MM-DD

    reloaded = TopicBank(path=str(tmp_path / 'bank.json'))
    assert reloaded._find(entry['id'])['faq_published'] == stamped
    assert reloaded.set_published(entry['id'], False) == ''
    assert 'faq_published' not in TopicBank(
        path=str(tmp_path / 'bank.json'))._find(entry['id'])
    assert bank.set_published('nope', True) is None


def test_bank_wrong_shape_file_starts_fresh(tmp_path):
    """A dict-shaped (or otherwise wrong) bank file must not crash — the
    'unreadable -> start fresh' contract covers wrong shapes too."""
    path = tmp_path / 'bank.json'
    path.write_text('{"oops": "a dict, not a list"}', encoding='utf-8')

    bank = TopicBank(path=str(path))  # no AttributeError
    assert bank.entries == []

    entry = bank.record(make_group(topic='Fresh Start'), [1.0, 0.0])
    assert entry is not None
    bank.save()  # merge path must tolerate the wrong-shape file on disk too

    reloaded = TopicBank(path=str(path))
    assert [e['topic'] for e in reloaded.entries] == ['Fresh Start']


def test_bank_off_disables_learning(monkeypatch):
    monkeypatch.setenv('TOPIC_BANK', 'off')
    analyzer = make_analyzer(monkeypatch)
    monkeypatch.setattr(analyzer.labeler, 'label_group',
                        lambda texts, keywords=None: {'topic': 'Password Reset',
                                                      'summary': 'Resets.'})
    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    assert 'seen_in_analyses' not in results['groups'][0]


def test_full_overlap_reupload_keeps_last_seen(tmp_path):
    """Re-analyzing old data is not a sighting: a full-overlap re-upload
    must not move last_seen forward (it would fake topic freshness)."""
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    g = _group_with_questions([('how do i raise the sftp timeout?', 'June 3, 2026')])
    entry = bank.record(g, [1.0, 0.0])
    entry['last_seen'] = '2026-06-03'  # simulate an old sighting stamp
    bank.record(g, [1.0, 0.0], matched=entry)
    assert entry['last_seen'] == '2026-06-03'


def test_duplicate_occurrences_fingerprint_once(tmp_path):
    """Two members with identical text+date are ONE occurrence identity —
    the badge and the history chart must count them the same way."""
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    g = _group_with_questions([('how do i reset my password?', 'June 3, 2026'),
                               ('how do i reset my password?', 'June 3, 2026')])
    entry = bank.record(g, [1.0, 0.0])
    assert entry['question_count'] == 1
    assert len(entry['seen']) == 1


def test_merge_carries_published_flag_and_alias_ids(tmp_path):
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    a = bank.record(_group_with_questions([('question a?', 'June 1, 2026')],
                                          topic='A'), [1.0, 0.0])
    b = bank.record(_group_with_questions([('question b?', 'June 2, 2026')],
                                          topic='B'), [0.0, 1.0])
    c = bank.record(_group_with_questions([('question c?', 'June 3, 2026')],
                                          topic='C'), [0.7, 0.7])
    a['faq_published'] = '2026-06-15'

    assert bank.merge(a['id'], b['id']) is True
    target = bank._find(b['id'])
    # The published date survives (latest wins when both sides have one)
    assert target['faq_published'] == '2026-06-15'
    # The source id lives on as an alias for saved-analysis history
    assert target['merged_ids'] == [a['id']]

    # Chained merge: aliases travel transitively
    assert bank.merge(b['id'], c['id']) is True
    final = bank._find(c['id'])
    assert set(final['merged_ids']) == {a['id'], b['id']}


def test_seed_centroid_blends_instead_of_being_replaced(tmp_path):
    """A curated seed is recorded with count 0; its first real sighting must
    BLEND with the seed centroid, not overwrite it wholesale."""
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    seed = {'topic': 'Password Reset', 'summary': None,
            'representative_question': 'How do I reset my password?',
            'keywords': [], 'count': 0, 'questions': []}
    entry = bank.record(seed, [1.0, 0.0])
    assert entry['question_count'] == 0

    g = _group_with_questions([('password reset link broken?', 'June 3, 2026')])
    bank.record(g, [0.0, 1.0], matched=entry)
    assert entry['question_count'] == 1  # honest count: only real sightings
    # Blended halfway between seed [1,0] and sighting [0,1], not equal to
    # the sighting alone
    v = np.asarray(entry['centroid'])
    assert abs(v[0] - v[1]) < 1e-6 and v[0] > 0.5


def test_set_answer_stores_clears_and_persists(tmp_path):
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    entry = bank.record(make_group(), [1.0, 0.0])
    stored = bank.set_answer(entry['id'], '  Reset it from the admin console.  ')
    assert stored == 'Reset it from the admin console.'

    reloaded = TopicBank(path=str(tmp_path / 'bank.json'))
    hit = reloaded._find(entry['id'])
    assert hit['curated_answer'] == 'Reset it from the admin console.'
    assert len(hit['answer_updated']) == 10  # YYYY-MM-DD

    assert reloaded.set_answer(entry['id'], '') == ''
    cleared = TopicBank(path=str(tmp_path / 'bank.json'))._find(entry['id'])
    assert 'curated_answer' not in cleared and 'answer_updated' not in cleared
    assert bank.set_answer('nope', 'x') is None


def test_merge_keeps_curated_answer(tmp_path):
    """The kept topic's own answer wins; the absorbed topic's answer fills
    the gap when the target has none."""
    bank = TopicBank(path=str(tmp_path / 'bank.json'))
    a = bank.record(_group_with_questions([('question a?', 'June 1, 2026')],
                                          topic='A'), [1.0, 0.0])
    b = bank.record(_group_with_questions([('question b?', 'June 2, 2026')],
                                          topic='B'), [0.0, 1.0])
    bank.set_answer(a['id'], 'Answer from A.')
    assert bank.merge(a['id'], b['id']) is True
    assert bank._find(b['id'])['curated_answer'] == 'Answer from A.'

    c = bank.record(_group_with_questions([('question c?', 'June 3, 2026')],
                                          topic='C'), [0.7, 0.7])
    bank.set_answer(c['id'], 'Answer from C, the kept topic.')
    assert bank.merge(b['id'], c['id']) is True
    assert bank._find(c['id'])['curated_answer'] == 'Answer from C, the kept topic.'
