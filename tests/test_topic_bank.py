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
    assert group['seen_in_analyses'] == 2


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


def test_bank_off_disables_learning(monkeypatch):
    monkeypatch.setenv('TOPIC_BANK', 'off')
    analyzer = make_analyzer(monkeypatch)
    monkeypatch.setattr(analyzer.labeler, 'label_group',
                        lambda texts, keywords=None: {'topic': 'Password Reset',
                                                      'summary': 'Resets.'})
    results = analyzer.analyze_slack_content(SAMPLE_CONTENT)
    assert 'seen_in_analyses' not in results['groups'][0]
