"""Tests for similarity grouping and the embedding cache."""

import numpy as np
import pytest

from slack_question_analyzer.similarity_analyzer import SimilarityAnalyzer, EmbeddingCache, EmbeddingError


def make_analyzer(monkeypatch, threshold='0.85'):
    monkeypatch.setenv('SIMILARITY_THRESHOLD', threshold)
    # A model without a task prefix, so fakes can match on raw text
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')
    return SimilarityAnalyzer(provider='ollama', use_disk_cache=False)


def question(text):
    return {'text': text, 'normalized_text': text.lower(), 'date': 'Unknown',
            'original_message': text}


def test_groups_similar_questions(monkeypatch):
    analyzer = make_analyzer(monkeypatch)

    # Two near-identical vectors and one orthogonal one
    fake_embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.05, 0.0],
        [0.0, 1.0, 0.0],
    ])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake_embeddings)

    questions = [question('How do I reset my password?'),
                 question('How can I reset my password?'),
                 question('What is the deploy schedule?')]
    groups = analyzer.group_similar_questions(questions)

    assert len(groups) == 2
    assert groups[0]['count'] == 2
    assert groups[1]['count'] == 1
    assert 0.0 <= groups[0]['avg_similarity'] <= 1.0


def test_empty_input_returns_no_groups(monkeypatch):
    analyzer = make_analyzer(monkeypatch)
    assert analyzer.group_similar_questions([]) == []


def test_invalid_threshold_rejected(monkeypatch):
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '1.5')
    with pytest.raises(ValueError):
        SimilarityAnalyzer(provider='ollama', use_disk_cache=False)

    monkeypatch.setenv('SIMILARITY_THRESHOLD', 'abc')
    with pytest.raises(ValueError):
        SimilarityAnalyzer(provider='ollama', use_disk_cache=False)


def test_invalid_provider_rejected(monkeypatch):
    with pytest.raises(ValueError):
        SimilarityAnalyzer(provider='something-else')


def test_batch_raises_when_provider_fails(monkeypatch):
    analyzer = make_analyzer(monkeypatch)
    analyzer.MAX_RETRIES = 1

    def boom(text):
        raise EmbeddingError('connection refused')

    monkeypatch.setattr(analyzer, '_ollama_embedding', boom)
    with pytest.raises(EmbeddingError):
        analyzer.get_embeddings_batch(['some question'])


def test_batch_uses_cache_and_dedupes(monkeypatch):
    analyzer = make_analyzer(monkeypatch)
    calls = []

    def fake_embedding(text):
        calls.append(text)
        return [1.0, 0.0]

    monkeypatch.setattr(analyzer, '_ollama_embedding', fake_embedding)
    result = analyzer.get_embeddings_batch(['a', 'a', 'b'])

    assert sorted(calls) == ['a', 'b']  # 'a' embedded once despite appearing twice
    assert result.shape == (3, 2)

    # Second run should be fully served from cache
    calls.clear()
    analyzer.get_embeddings_batch(['a', 'b'])
    assert calls == []


def test_exact_duplicates_need_no_embeddings(monkeypatch):
    """Identical questions are grouped with zero AI calls."""
    analyzer = make_analyzer(monkeypatch)
    calls = []
    monkeypatch.setattr(analyzer, '_ollama_embedding',
                        lambda text: calls.append(text) or [1.0, 0.0])

    questions = [question('How do I reset my password?') for _ in range(3)]
    groups = analyzer.group_similar_questions(questions)

    assert calls == []  # no embeddings fetched at all
    assert len(groups) == 1
    assert groups[0]['count'] == 3
    assert groups[0]['avg_similarity'] == 1.0


def test_duplicates_embed_once_per_distinct_question(monkeypatch):
    analyzer = make_analyzer(monkeypatch)
    vectors = {'how do i reset my password?': [1.0, 0.0],
               'what is the deploy schedule?': [0.0, 1.0]}
    calls = []

    def fake_embedding(text):
        calls.append(text)
        return vectors[text]

    monkeypatch.setattr(analyzer, '_ollama_embedding', fake_embedding)

    questions = [question('How do I reset my password?'),
                 question('How do I reset my password?'),
                 question('What is the deploy schedule?')]
    groups = analyzer.group_similar_questions(questions)

    assert len(calls) == 2  # one embedding per distinct question, not three
    assert groups[0]['count'] == 2
    assert groups[1]['count'] == 1


def test_lexical_near_duplicates_merge_without_ai(monkeypatch):
    """Rewordings sharing >=90% of tokens merge before any embedding call."""
    analyzer = make_analyzer(monkeypatch)
    calls = []
    monkeypatch.setattr(analyzer, '_ollama_embedding',
                        lambda text: calls.append(text) or [1.0, 0.0])

    base = 'how do i configure the antivirus scanner for inbound file transfers'
    questions = [question(base), question(base + ' please')]
    groups = analyzer.group_similar_questions(questions)

    assert calls == []  # merged lexically; single bucket means no embeddings
    assert len(groups) == 1
    assert groups[0]['count'] == 2


def test_borderline_groups_merged_when_verifier_agrees(monkeypatch):
    """Pairs just below the threshold are merged when the LLM says same topic."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    monkeypatch.setenv('LLM_VERIFY_MARGIN', '0.05')

    # Two questions with similarity ~0.83: below 0.85 but inside the margin
    fake = np.array([[1.0, 0.0], [0.83, np.sqrt(1 - 0.83 ** 2)]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('How do I reset my password?'),
                 question('Steps for changing my password?')]

    asked = []

    def verifier(a, b):
        asked.append((a, b))
        return True

    groups = analyzer.group_similar_questions(questions, verifier=verifier)
    assert len(asked) == 1
    assert len(groups) == 1
    assert groups[0]['count'] == 2


def test_borderline_groups_stay_apart_when_verifier_disagrees(monkeypatch):
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    monkeypatch.setenv('LLM_VERIFY_MARGIN', '0.05')

    fake = np.array([[1.0, 0.0], [0.83, np.sqrt(1 - 0.83 ** 2)]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('How do I reset my password?'),
                 question('How do I reset my API key?')]
    groups = analyzer.group_similar_questions(questions, verifier=lambda a, b: False)
    assert len(groups) == 2


def test_clearly_different_groups_skip_the_verifier(monkeypatch):
    """The verifier is only consulted inside the borderline band."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')

    fake = np.array([[1.0, 0.0], [0.0, 1.0]])  # similarity ~0: far below band
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    def never(a, b):
        raise AssertionError('verifier should not be called')

    questions = [question('How do I reset my password?'),
                 question('What is the deploy schedule?')]
    groups = analyzer.group_similar_questions(questions, verifier=never)
    assert len(groups) == 2


def test_average_link_prevents_chaining(monkeypatch):
    """
    Regression for the real-world mega-group: in a domain-homogeneous corpus
    every adjacent pair can clear the threshold (A~B, B~C) while A and C are
    unrelated. Single-link chains them into one group; average-link must not.
    """
    analyzer = make_analyzer(monkeypatch, threshold='0.75')
    matrix = np.array([
        [1.0, 0.80, 0.20],
        [0.80, 1.0, 0.80],
        [0.20, 0.80, 1.0],
    ])
    clusters = analyzer._cluster_buckets(3, matrix)
    # B joins A (0.8). C's average to {A,B} is (0.2+0.8)/2 = 0.5 < 0.75: stays out
    assert clusters == [[0, 1], [2]]


def test_average_link_keeps_two_tight_clusters_apart(monkeypatch):
    """Two tight pairs with elevated cross-similarity stay two groups."""
    analyzer = make_analyzer(monkeypatch, threshold='0.75')
    matrix = np.array([
        [1.0, 0.90, 0.76, 0.60],
        [0.90, 1.0, 0.60, 0.60],
        [0.76, 0.60, 1.0, 0.90],
        [0.60, 0.60, 0.90, 1.0],
    ])
    clusters = analyzer._cluster_buckets(4, matrix)
    # C's avg to {A,B} = (0.76+0.60)/2 = 0.68 < 0.75 despite the 0.76 best link
    assert clusters == [[0, 1], [2, 3]]


def test_large_corpus_uses_leader_clustering(monkeypatch):
    """Above LARGE_CLUSTERING_THRESHOLD, grouping avoids the full n^2 matrix."""
    analyzer = make_analyzer(monkeypatch)
    monkeypatch.setenv('LARGE_CLUSTERING_THRESHOLD', '2')  # force the large path

    fake = np.array([[1.0, 0.0], [1.0, 0.05], [0.0, 1.0]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    def never(a, b):
        raise AssertionError('verifier must be skipped on the large path')

    questions = [question('How do I reset my password?'),
                 question('Steps to reset my password quickly?'),
                 question('What is the deploy schedule?')]
    groups = analyzer.group_similar_questions(questions, verifier=never)

    assert len(groups) == 2
    assert groups[0]['count'] == 2
    assert groups[1]['count'] == 1
    assert 0.0 < groups[0]['avg_similarity'] <= 1.0
    assert groups[1]['avg_similarity'] == 1.0


def test_nomic_model_gets_clustering_prefix(monkeypatch):
    """nomic-embed-text is trained with task prefixes; we must send one."""
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('OLLAMA_MODEL', 'nomic-embed-text')
    analyzer = SimilarityAnalyzer(provider='ollama', use_disk_cache=False)
    assert analyzer.embed_prefix == 'clustering: '

    sent = []
    monkeypatch.setattr(analyzer, '_ollama_embedding',
                        lambda text: sent.append(text) or [1.0, 0.0])
    analyzer.get_embeddings_batch(['how do i reset my password?'])
    assert sent == ['clustering: how do i reset my password?']


def test_other_models_get_no_prefix(monkeypatch):
    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.85')
    monkeypatch.setenv('OLLAMA_MODEL', 'mxbai-embed-large')
    analyzer = SimilarityAnalyzer(provider='ollama', use_disk_cache=False)
    assert analyzer.embed_prefix == ''


def test_similarity_stats_recorded(monkeypatch):
    analyzer = make_analyzer(monkeypatch)
    fake = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('a much longer question one?'),
                 question('another quite different two?'),
                 question('completely unrelated three?')]
    analyzer.group_similar_questions(questions)

    stats = analyzer.last_similarity_stats
    assert stats is not None
    assert stats['max'] == 0.8  # best pair: [0.6,0.8] vs [0,1]
    assert 0.0 <= stats['median'] <= stats['p90'] <= stats['max']


def test_default_threshold_is_model_aware(monkeypatch):
    monkeypatch.delenv('SIMILARITY_THRESHOLD', raising=False)
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')
    analyzer = SimilarityAnalyzer(provider='ollama', use_disk_cache=False)
    # Field-calibrated upward twice: unrelated same-domain questions can
    # score 0.8+ with nomic, so the base default is 0.85 (plus the noise gate)
    assert analyzer.similarity_threshold == 0.85
    assert analyzer.threshold_pinned is False

    monkeypatch.setenv('SIMILARITY_THRESHOLD', '0.9')
    pinned = SimilarityAnalyzer(provider='ollama', use_disk_cache=False)
    assert pinned.similarity_threshold == 0.9
    assert pinned.threshold_pinned is True


def test_threshold_auto_adjusts_when_top_pair_stands_out(monkeypatch):
    monkeypatch.delenv('SIMILARITY_THRESHOLD', raising=False)
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')
    analyzer = SimilarityAnalyzer(provider='ollama', use_disk_cache=False)

    # Best pair 0.78 (below the 0.80 default) but far above the other pairs:
    # a genuine cluster the default just missed
    fake = np.array([[1.0, 0.0, 0.0],
                     [0.78, 0.6247, 0.0],
                     [0.0, 0.0, 1.0]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('a longer question number one?'),
                 question('a different question number two?'),
                 question('an unrelated question number three?')]
    groups = analyzer.group_similar_questions(questions)

    assert analyzer.threshold_auto_adjusted is True
    assert analyzer.effective_threshold == 0.76  # best pair 0.78, minus 0.02
    # The configured threshold must NOT drift: a reused analyzer would carry
    # one corpus's adjustment into the next analysis
    assert analyzer.similarity_threshold == 0.85
    assert any(g['count'] == 2 for g in groups)


def test_no_auto_adjust_into_the_noise_band(monkeypatch):
    """
    Regression for the real-world 70%-average blob: when ALL pairs sit in a
    narrow band (single-domain corpus), relaxing the threshold would merge
    unrelated topics. The analyzer must refuse and keep singletons.
    """
    monkeypatch.delenv('SIMILARITY_THRESHOLD', raising=False)
    monkeypatch.setenv('OLLAMA_MODEL', 'test-embed')
    analyzer = SimilarityAnalyzer(provider='ollama', use_disk_cache=False)

    # Pairwise sims ~0.70-0.72: a dense noise band with no standout pair
    fake = np.array([[1.0, 0.0, 0.0],
                     [0.72, 0.6940, 0.0],
                     [0.70, 0.2824, 0.6559]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('a longer question number one?'),
                 question('a different question number two?'),
                 question('an unrelated question number three?')]
    groups = analyzer.group_similar_questions(questions)

    assert analyzer.threshold_auto_adjusted is False
    assert analyzer.similarity_threshold == 0.85  # unchanged
    assert all(g['count'] == 1 for g in groups)  # honest singletons, no blob


def test_verifier_merge_rejected_when_combined_group_is_loose(monkeypatch):
    """An LLM 'same topic' yes cannot re-create a mixed mega-group."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')

    matrix = np.array([
        [1.00, 0.90, 0.84, 0.20],
        [0.90, 1.00, 0.20, 0.20],
        [0.84, 0.20, 1.00, 0.90],
        [0.20, 0.20, 0.90, 1.00],
    ])

    def never(a, b):
        raise AssertionError('verifier must not be consulted for a loose merge')

    buckets = [[question(f'question number {i}?')] for i in range(4)]
    clusters = analyzer._merge_borderline_clusters(
        [[0, 1], [2, 3]], matrix, buckets, never)
    # Combined avg would be (.9+.9+.84+.2+.2+.2)/6 = 0.54: guard skips it
    assert clusters == [[0, 1], [2, 3]]


def test_pinned_threshold_never_auto_adjusts(monkeypatch):
    analyzer = make_analyzer(monkeypatch, threshold='0.85')  # env-pinned

    fake = np.array([[1.0, 0.0], [0.6, 0.8], [-1.0, 0.0]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('a longer question number one?'),
                 question('a different question number two?'),
                 question('an opposite question number three?')]
    groups = analyzer.group_similar_questions(questions)

    assert analyzer.threshold_auto_adjusted is False
    assert all(g['count'] == 1 for g in groups)


def test_embedding_cache_roundtrip(tmp_path):
    cache = EmbeddingCache('ollama', 'test-model', cache_dir=str(tmp_path))
    assert cache.get('hello') is None

    cache.set('hello', [0.1, 0.2])
    cache.save()

    reloaded = EmbeddingCache('ollama', 'test-model', cache_dir=str(tmp_path))
    assert reloaded.get('hello') == [0.1, 0.2]


def test_embedding_cache_survives_corruption(tmp_path):
    cache = EmbeddingCache('ollama', 'test-model', cache_dir=str(tmp_path))
    cache.set('hello', [0.1])
    cache.save()
    cache.cache_path.write_text('{not valid json')

    reloaded = EmbeddingCache('ollama', 'test-model', cache_dir=str(tmp_path))
    assert reloaded.get('hello') is None  # starts fresh instead of crashing


def test_concurrent_cache_instances_merge_on_save(tmp_path):
    """Two instances saving to the same file keep each other's entries."""
    first = EmbeddingCache('ollama', 'test-model', cache_dir=str(tmp_path))
    second = EmbeddingCache('ollama', 'test-model', cache_dir=str(tmp_path))

    first.set('alpha', [0.1])
    first.save()
    second.set('beta', [0.2])
    second.save()  # must not clobber 'alpha' written after second loaded

    reloaded = EmbeddingCache('ollama', 'test-model', cache_dir=str(tmp_path))
    assert reloaded.get('alpha') == [0.1]
    assert reloaded.get('beta') == [0.2]


def test_cache_evicts_oldest_beyond_max_entries(tmp_path):
    from slack_question_analyzer.disk_cache import JsonDiskCache
    cache = JsonDiskCache('ollama', 'test-model', str(tmp_path), max_entries=2)
    cache.set('one', [1])
    cache.set('two', [2])
    cache.set('three', [3])
    cache.save()

    reloaded = JsonDiskCache('ollama', 'test-model', str(tmp_path))
    assert reloaded.get('one') is None  # oldest evicted
    assert reloaded.get('two') == [2]
    assert reloaded.get('three') == [3]


def test_disabled_cache_does_not_write(tmp_path):
    cache = EmbeddingCache('ollama', 'test-model', cache_dir=str(tmp_path), enabled=False)
    cache.set('hello', [0.1])
    cache.save()
    assert not cache.cache_path.exists()


def test_subject_overlap_folds_inflection(monkeypatch):
    """'timing out' and 'timeout' must count as a shared subject — the
    unstemmed copy of this metric scored the symptom/fix pair at zero and
    the pair never reached the verifier."""
    analyzer = make_analyzer(monkeypatch)
    buckets = [
        [{'normalized_text': 'sftp transfers keep timing out on slow links?'}],
        [{'normalized_text': 'how do i increase the sftp timeout setting?'}],
    ]
    a = analyzer._subject_tokens([0], buckets)
    b = analyzer._subject_tokens([1], buckets)
    assert analyzer._subject_overlap(a, b) >= 0.5


def test_subject_gate_defers_template_merges_to_the_verifier(monkeypatch):
    """Two questions sharing only 'Does wM MFT support X?' scaffolding sit
    just above the bar on template similarity alone. With a verifier
    available, the join is deferred to it (here: verifier says no ->
    separate groups); with MERGE_SUBJECT_MIN=0 the old numeric merge is
    back."""
    import numpy as np
    analyzer = make_analyzer(monkeypatch)
    texts = ['does wm mft support google cloud storage?',
             'does wm mft support a maximum file size limit?']
    questions = [{'text': t, 'normalized_text': t, 'original_message': t,
                  'date': 'Unknown'} for t in texts]
    # Cosine 0.86: above the 0.85 bar, inside the bar+0.03 narrow band
    vectors = {texts[0]: [1.0, 0.0, 0.0],
               texts[1]: [0.86, np.sqrt(1 - 0.86 ** 2), 0.0]}
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda t, progress_callback=None: np.array(
                            [vectors[x] for x in t]))

    groups = analyzer.group_similar_questions(
        questions, verifier=lambda a, b: False)
    assert all(g['count'] == 1 for g in groups)  # verifier declined the merge

    monkeypatch.setenv('MERGE_SUBJECT_MIN', '0')  # gate off: numeric merge
    groups = analyzer.group_similar_questions(
        questions, verifier=lambda a, b: False)
    assert any(g['count'] == 2 for g in groups)


def test_cache_hit_survives_eviction_on_full_cache(tmp_path):
    """An entry READ this run must not be evicted by a mid-run save().

    Regression: eviction trims the front of the insertion-ordered dict, and
    get() used to leave hit entries at the front — so on a full cache, this
    run's own cache hits could vanish between save() and the read-back,
    crashing the analysis with EmbeddingError.
    """
    from slack_question_analyzer.disk_cache import JsonDiskCache
    cache = JsonDiskCache('ollama', 'test-model', str(tmp_path), max_entries=2)
    cache.set('old-but-used', [1])
    cache.set('other', [2])
    cache.save()

    fresh = JsonDiskCache('ollama', 'test-model', str(tmp_path), max_entries=2)
    assert fresh.get('old-but-used') == [1]  # hit refreshes recency
    fresh.set('newcomer', [3])               # pushes the cache over max
    fresh.save()
    assert fresh.get('old-but-used') == [1]  # still there after eviction
    assert fresh.get('other') is None        # the untouched entry went


def test_cache_wrong_shape_file_starts_fresh(tmp_path):
    """Valid JSON of the wrong shape must mean 'start fresh', not crash."""
    from slack_question_analyzer.disk_cache import JsonDiskCache
    probe = JsonDiskCache('ollama', 'test-model', str(tmp_path))
    probe.cache_path.parent.mkdir(parents=True, exist_ok=True)
    probe.cache_path.write_text('[1, 2, 3]', encoding='utf-8')

    cache = JsonDiskCache('ollama', 'test-model', str(tmp_path))
    assert cache.get('anything') is None  # no AttributeError
    cache.set('anything', [0.5])
    cache.save()
    assert JsonDiskCache('ollama', 'test-model', str(tmp_path)).get('anything') == [0.5]


def test_noise_gate_raises_the_bar_on_dense_corpora(monkeypatch):
    """
    Field regression: any FIXED threshold eventually sits inside some corpus's
    noise band. The bar must rise above the measured pairwise bulk (p90).
    """
    analyzer = make_analyzer(monkeypatch, threshold='0.85')

    # Dense corpus: the bulk of pairs sits at ~0.86, above the threshold
    analyzer.last_similarity_stats = {'max': 0.91, 'p90': 0.87, 'median': 0.84}
    bar = analyzer._gated_threshold(12)
    assert bar == 0.92  # p90 + 0.05
    assert analyzer.noise_gate == 0.92

    # Sparse corpus: bulk far below the threshold -> threshold unchanged
    analyzer.last_similarity_stats = {'max': 0.9, 'p90': 0.55, 'median': 0.4}
    assert analyzer._gated_threshold(12) == 0.85
    assert analyzer.noise_gate is None


def test_noise_gate_skipped_on_tiny_corpora(monkeypatch):
    """p90 is meaningless with a handful of pairs; the gate must not engage."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    analyzer.last_similarity_stats = {'max': 0.91, 'p90': 0.9, 'median': 0.9}
    assert analyzer._gated_threshold(4) == 0.85
    assert analyzer.noise_gate is None


def test_noise_gate_capped(monkeypatch):
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    analyzer.last_similarity_stats = {'max': 0.99, 'p90': 0.97, 'median': 0.95}
    assert analyzer._gated_threshold(20) == 0.95  # never above the cap


def test_ai_audit_splits_false_numeric_pairs(monkeypatch):
    """
    Field regression (metering + Azure tokens at 0.81): embeddings score some
    unrelated pairs as high as true pairs. The auditor checks every formed
    group; evicted outliers become unique questions.
    """
    analyzer = make_analyzer(monkeypatch, threshold='0.75')
    fake = np.array([[1.0, 0.0], [0.81, np.sqrt(1 - 0.81 ** 2)]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('Does the metering agent come pre-installed?'),
                 question('Are container-level Azure tokens supported?')]
    groups = analyzer.group_similar_questions(questions,
                                              auditor=lambda texts: [0])
    assert len(groups) == 2  # the false 0.81 pair was split by the AI
    assert all(g['count'] == 1 for g in groups)


def test_uncertain_auditor_keeps_groups(monkeypatch):
    """On auditor failure (None) or a clean audit ([]), the group stands."""
    analyzer = make_analyzer(monkeypatch, threshold='0.75')
    fake = np.array([[1.0, 0.0], [0.81, np.sqrt(1 - 0.81 ** 2)]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('How do I reset my password?'),
                 question('Steps for changing my password?')]
    for verdict in (None, []):
        groups = analyzer.group_similar_questions(
            questions, auditor=lambda texts, v=verdict: v)
        assert len(groups) == 1
        assert groups[0]['count'] == 2


def test_audit_evicts_outlier_from_larger_group(monkeypatch):
    """
    Field regression: a 3-question group (metering glued onto a monitoring
    pair) was never checked because only pairs were confirmed. The audit
    covers any group size and keeps the coherent remainder together.
    """
    analyzer = make_analyzer(monkeypatch, threshold='0.75')

    def auditor(texts):
        return [i for i, t in enumerate(texts) if 'metering' in t]

    clusters = analyzer._audit_clusters(
        [[0, 1, 2], [3]],
        [[question('Configure the on-prem metering server?')],
         [question('Monitor one application for errors?')],
         [question('Change the monitoring mindset for integration products?')],
         [question('Unrelated?')]], auditor)
    assert sorted(map(sorted, clusters)) == [[0], [1, 2], [3]]


def test_audit_respects_call_cap(monkeypatch):
    analyzer = make_analyzer(monkeypatch, threshold='0.75')
    monkeypatch.setenv('LLM_VERIFY_MAX', '1')
    calls = []

    def auditor(texts):
        calls.append(texts)
        return [0]

    clusters = analyzer._audit_clusters(
        [[0, 1, 2], [3, 4]],
        [[question(f'q{i}?')] for i in range(5)], auditor)
    assert len(calls) == 1  # cap enforced; biggest group audited first
    assert sorted(map(sorted, clusters)) == [[0], [1, 2], [3, 4]]


def test_ranking_breaks_count_ties_by_cohesion(monkeypatch):
    """Equal-count groups must rank by avg similarity, not insertion order."""
    analyzer = make_analyzer(monkeypatch, threshold='0.75')
    # Pair A at ~0.78, pair B at ~0.95 (orthogonal across pairs)
    fake = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.78, float(np.sqrt(1 - 0.78 ** 2)), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.95, float(np.sqrt(1 - 0.95 ** 2))],
    ])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('first pair question one?'),
                 question('first pair question two?'),
                 question('second pair question one?'),
                 question('second pair question two?')]
    groups = analyzer.group_similar_questions(questions)

    assert [g['count'] for g in groups] == [2, 2]
    # Tighter group ranks first despite equal counts
    assert groups[0]['avg_similarity'] > groups[1]['avg_similarity']


def test_known_topics_claim_questions_despite_low_pairwise_similarity(monkeypatch):
    """
    Funnel stage 1: two questions whose PAIRWISE similarity is below the bar
    still group when both score high against the same curated category. This
    is the fix for 0-groups runs where obviously-related questions (two Azure
    token questions) sat just under an adaptive bar.
    """
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    # Both questions at ~0.90 to the category centroid (1,0,0), but only
    # ~0.62 to each other
    fake = np.array([[0.9, 0.435, 0.0], [0.9, -0.435, 0.0]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)
    topics = [{'topic': 'Azure Token Support', 'centroid': [1.0, 0.0, 0.0]}]

    questions = [question('Why does a container-level token fail against Azure Blob?'),
                 question('Do we support container-level tokens in Azure Blob?')]
    groups = analyzer.group_similar_questions(questions, known_topics=topics)
    assert len(groups) == 1
    assert groups[0]['count'] == 2


def test_single_known_topic_claim_released_to_clustering(monkeypatch):
    """A category claiming only ONE question must not lock it away from
    normal clustering (it could still pair with an unclaimed question)."""
    analyzer = make_analyzer(monkeypatch, threshold='0.75')
    # q0 matches the category; q1 doesn't, but q0/q1 are 0.81 similar
    fake = np.array([[1.0, 0.0], [0.81, float(np.sqrt(1 - 0.81 ** 2))]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)
    topics = [{'topic': 'Some Category', 'centroid': [1.0, 0.0]}]

    questions = [question('How do I reset my password?'),
                 question('Steps for changing my password?')]
    groups = analyzer.group_similar_questions(questions, known_topics=topics)
    assert len(groups) == 1  # still paired by normal clustering
    assert groups[0]['count'] == 2


def test_known_topics_with_wrong_dimensions_are_ignored(monkeypatch):
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    fake = np.array([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)
    topics = [{'topic': 'Legacy Entry', 'centroid': [1.0, 0.0, 0.0, 0.0]}]

    questions = [question('a question about one thing?'),
                 question('a question about another thing?')]
    groups = analyzer.group_similar_questions(questions, known_topics=topics)
    assert len(groups) == 2  # no crash, no bogus claims


def test_singleton_rescue_absorbs_undergrouped_member(monkeypatch):
    """
    Field regression: a third thread question sat just under the in-bucket
    bar and was stranded in uniques. The rescue pass adjudicates it against
    its NEAREST group only; the verifier's yes absorbs it.
    """
    analyzer = make_analyzer(monkeypatch, threshold='0.8')
    # q0/q1 pair at 0.85; q2 averages ~0.76 to the group (under bar,
    # inside the rescue margin on AVERAGE-link — the same metric
    # clustering uses, so one close member can't pull in a stranger)
    fake = np.array([
        [1.0, 0.0, 0.0],
        [0.85, float(np.sqrt(1 - 0.85 ** 2)), 0.0],
        [0.72, 0.167, float(np.sqrt(1 - 0.72 ** 2 - 0.167 ** 2))],
    ])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)
    questions = [question('Any way to avoid running out of threads?'),
                 question('Can the thread issue be fixed by scaling vertically?'),
                 question('Any way around thread use with many scheduled actions?')]

    asked = []

    def verifier(a, b):
        asked.append((tuple(a), tuple(b)))
        return True

    groups = analyzer.group_similar_questions(questions, verifier=verifier)
    assert len(groups) == 1
    assert groups[0]['count'] == 3
    # Exactly one rescue adjudication, against the nearest group
    rescue_calls = [c for c in asked if len(c[0]) == 1]
    assert len(rescue_calls) == 1


def test_singleton_rescue_abstain_stays_singleton(monkeypatch):
    """Verifier no/uncertain -> the singleton stays a singleton (rare is
    not wrong)."""
    analyzer = make_analyzer(monkeypatch, threshold='0.8')
    fake = np.array([
        [1.0, 0.0, 0.0],
        [0.85, float(np.sqrt(1 - 0.85 ** 2)), 0.0],
        [0.75, 0.0, float(np.sqrt(1 - 0.75 ** 2))],
    ])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)
    questions = [question('Threads question one?'),
                 question('Threads question two?'),
                 question('A genuinely different ask?')]
    for verdict in (False, None):
        groups = analyzer.group_similar_questions(
            questions, verifier=lambda a, b, v=verdict: v if len(a) == 1 else None)
        assert sorted(g['count'] for g in groups) == [1, 2]


def test_singleton_rescue_skips_far_singletons(monkeypatch):
    """A singleton far from every group is rare, not wrong: no LLM call."""
    analyzer = make_analyzer(monkeypatch, threshold='0.8')
    fake = np.array([
        [1.0, 0.0, 0.0],
        [0.85, float(np.sqrt(1 - 0.85 ** 2)), 0.0],
        [0.0, 0.0, 1.0],  # orthogonal: far outside the rescue margin
    ])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    def never_single(a, b):
        assert len(a) != 1, 'far singleton must not reach the verifier'
        return None

    questions = [question('a?'), question('b?'), question('c?')]
    groups = analyzer.group_similar_questions(questions, verifier=never_single)
    assert sorted(g['count'] for g in groups) == [1, 2]


def typed_question(text, qtype=None):
    q = question(text)
    if qtype:
        q['qtype'] = qtype
    return q


def test_cross_type_borderline_merge_vetoed(monkeypatch):
    """The GENERAL rule behind every observed false merge: a capability
    question never LLM-merges with a breakage report on mere VOCABULARY
    overlap — different subjects, the verifier is not even consulted."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    monkeypatch.setenv('LLM_VERIFY_MARGIN', '0.05')
    fake = np.array([[1.0, 0.0], [0.83, float(np.sqrt(1 - 0.83 ** 2))]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    def never(a, b):
        raise AssertionError('cross-type pair must not reach the verifier')

    questions = [typed_question('How do I enable PGP encryption on uploads?', 'how-to'),
                 typed_question('Why does the metering report keep failing?', 'troubleshooting')]
    groups = analyzer.group_similar_questions(questions, verifier=never)
    assert len(groups) == 2


def test_cross_type_merge_allowed_when_subject_is_shared(monkeypatch):
    """The veto's downgrade: a how-to and a breakage report about the SAME
    named subject reach the verifier (one answer often resolves both) — and
    the verifier still makes the call."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    monkeypatch.setenv('LLM_VERIFY_MARGIN', '0.05')
    fake = np.array([[1.0, 0.0], [0.83, float(np.sqrt(1 - 0.83 ** 2))]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [typed_question('How do I merge the content of two files?', 'how-to'),
                 typed_question('Why does the file merge keep failing?', 'troubleshooting')]

    asked = []

    def verifier(a, b):
        asked.append((a, b))
        return True

    groups = analyzer.group_similar_questions(questions, verifier=verifier)
    assert len(asked) == 1  # consulted, not auto-vetoed
    assert len(groups) == 1  # verifier said yes -> merged

    # And a NO still keeps them apart
    groups = analyzer.group_similar_questions(questions, verifier=lambda a, b: False)
    assert len(groups) == 2


def test_same_family_borderline_merge_still_asks_verifier(monkeypatch):
    """how-to and is-it-possible are the same family (capability): the
    verifier still decides those."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    monkeypatch.setenv('LLM_VERIFY_MARGIN', '0.05')
    fake = np.array([[1.0, 0.0], [0.83, float(np.sqrt(1 - 0.83 ** 2))]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [typed_question('How do I trigger a transfer via REST?', 'how-to'),
                 typed_question('Is it possible to trigger transfers via REST?', 'is-it-possible')]
    groups = analyzer.group_similar_questions(questions, verifier=lambda a, b: True)
    assert len(groups) == 1


def test_untyped_questions_never_vetoed(monkeypatch):
    """No type info = no veto (regex-extracted questions carry no qtype)."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    monkeypatch.setenv('LLM_VERIFY_MARGIN', '0.05')
    fake = np.array([[1.0, 0.0], [0.83, float(np.sqrt(1 - 0.83 ** 2))]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)
    questions = [question('How do I reset my password?'),
                 question('Steps for changing my password?')]
    groups = analyzer.group_similar_questions(questions, verifier=lambda a, b: True)
    assert len(groups) == 1


def test_cross_type_singleton_rescue_vetoed(monkeypatch):
    analyzer = make_analyzer(monkeypatch, threshold='0.8')
    fake = np.array([
        [1.0, 0.0, 0.0],
        [0.85, float(np.sqrt(1 - 0.85 ** 2)), 0.0],
        [0.75, 0.0, float(np.sqrt(1 - 0.75 ** 2))],
    ])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    def verifier(a, b):
        if len(a) == 1:
            raise AssertionError('cross-type rescue must not reach the verifier')
        return None

    questions = [typed_question('How can we avoid running out of threads?', 'how-to'),
                 typed_question('Is there a way to scale thread use vertically?', 'is-it-possible'),
                 typed_question('Why did the thread pool crash with an error?', 'troubleshooting')]
    groups = analyzer.group_similar_questions(questions, verifier=verifier)
    assert sorted(g['count'] for g in groups) == [1, 2]


def test_audit_eviction_needs_verifier_agreement(monkeypatch):
    """Fresh-transcript regression: a single audit sample evicted a TRUE
    pair. Eviction is destructive — the auditor nominates, the verifier
    must independently confirm (explicit False) or the nominee stays."""
    analyzer = make_analyzer(monkeypatch, threshold='0.75')

    # Auditor flags index 0; verifier says SAME topic -> eviction overruled
    clusters = analyzer._audit_clusters(
        [[0, 1]],
        [[question('Limit total concurrent transfers per node?')],
         [question('Cap how many transfers run at the same time per node?')]],
        auditor=lambda texts: [0],
        verifier=lambda a, b: True)
    assert sorted(map(sorted, clusters)) == [[0, 1]]  # pair survives

    # Verifier agrees it's different -> eviction proceeds
    clusters = analyzer._audit_clusters(
        [[0, 1]],
        [[question('Does the metering agent come pre-installed?')],
         [question('Are container-level Azure tokens supported?')]],
        auditor=lambda texts: [0],
        verifier=lambda a, b: False)
    assert sorted(map(sorted, clusters)) == [[0], [1]]

    # No verifier available -> auditor verdict stands (old behavior)
    clusters = analyzer._audit_clusters(
        [[0, 1]],
        [[question('a?')], [question('b?')]],
        auditor=lambda texts: [0])
    assert sorted(map(sorted, clusters)) == [[0], [1]]


def test_audit_undoes_a_flagged_rescue_without_verifier_overrule(monkeypatch):
    """Eval rounds 3-5: rescue (verifier YES) -> audit flags the member ->
    verifier overrules its own second opinion -> mega-group. A rescue is
    one yes on a borderline add; an audit flag ties the judges 1-1, and a
    tie reverts: the rescue is undone with NO overrule round."""
    analyzer = make_analyzer(monkeypatch, threshold='0.8')
    fake = np.array([
        [1.0, 0.0, 0.0],
        [0.85, float(np.sqrt(1 - 0.85 ** 2)), 0.0],
        [0.72, 0.167, float(np.sqrt(1 - 0.72 ** 2 - 0.167 ** 2))],
    ])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)
    questions = [question('How do we enforce host key verification?'),
                 question('Can MFT verify the partner host key first?'),
                 question('How do I rotate the SSH keys for partners?')]

    # The merge-happy verifier says yes to EVERYTHING (rescue and overrule
    # alike); the auditor flags the rescued member as an outlier
    auditor_calls = []

    def auditor(texts):
        auditor_calls.append(list(texts))
        return [len(texts) - 1] if len(texts) == 3 else None

    groups = analyzer.group_similar_questions(
        questions, verifier=lambda a, b: True, auditor=auditor)
    assert auditor_calls  # the audit ran on the 3x group
    counts = sorted(g['count'] for g in groups)
    assert counts == [1, 2]  # rescue undone: tie reverts, no mega-group
    singleton = next(g for g in groups if g['count'] == 1)
    assert 'rotate the SSH keys' in singleton['representative_question']


def test_rescue_only_completes_pairs_never_grows_established_groups(monkeypatch):
    """Nine eval rounds of mega-groups grew the same way: a singleton
    rescued into an already-formed 3+ group. Rescue exists to complete an
    under-grouped PAIR; a 3+ group that didn't catch the singleton during
    clustering is evidence the singleton differs."""
    analyzer = make_analyzer(monkeypatch, threshold='0.8')
    # q0/q1/q2: an established trio; q3 sits in the rescue band near them
    fake = np.array([
        [1.0, 0.0, 0.0],
        [0.9, float(np.sqrt(1 - 0.9 ** 2)), 0.0],
        [0.9, -float(np.sqrt(1 - 0.9 ** 2)), 0.0],
        [0.72, 0.167, float(np.sqrt(1 - 0.72 ** 2 - 0.167 ** 2))],
    ])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)
    questions = [question('Enforce host key verification?'),
                 question('Verify the partner host key first?'),
                 question('Require host key checking on outbound?'),
                 question('How do I rotate the SSH keys for partners?')]

    rescue_calls = []

    def verifier(a, b):
        if len(a) == 1:
            rescue_calls.append(a)
        return True  # merge-happy: would approve anything asked

    groups = analyzer.group_similar_questions(questions, verifier=verifier)
    counts = sorted(g['count'] for g in groups)
    assert counts == [1, 3]      # singleton NOT absorbed into the trio
    assert rescue_calls == []    # the rescue never even consulted the LLM


def test_rescue_cap_is_enforced_against_live_group_size(monkeypatch):
    """The pair cap must count members as the group GROWS: with the cap
    checked against a stale snapshot, one pair could absorb every nearby
    singleton in a single pass — the mega-group failure the cap exists
    to prevent."""
    monkeypatch.setenv('LLM_RESCUE_MAX_GROUP', '2')
    analyzer = make_analyzer(monkeypatch)
    analyzer.effective_threshold = 0.85

    buckets = [[question(f'question number {i}?')] for i in range(4)]
    clusters = [[0, 1], [2], [3]]
    # Both singletons sit 0.80 from the pair: inside the rescue margin
    sim = np.array([
        [1.00, 0.90, 0.80, 0.80],
        [0.90, 1.00, 0.80, 0.80],
        [0.80, 0.80, 1.00, 0.10],
        [0.80, 0.80, 0.10, 1.00],
    ])

    result, rescued = analyzer._rescue_singletons(
        clusters, sim, buckets, verifier=lambda a, b: True)

    assert rescued == {2}          # first singleton completes the pair...
    assert [3] in result           # ...second must NOT pile onto the trio
    assert sorted(len(c) for c in result) == [1, 3]


def test_rescue_completes_a_trio_under_the_tighter_margin(monkeypatch):
    """3+ targets are rescueable — the '4th occurrence stranded under the
    bar' case — but only inside the TIGHTER large-group window."""
    monkeypatch.setenv('LLM_RESCUE_MARGIN_LARGE', '0.05')
    analyzer = make_analyzer(monkeypatch)
    analyzer.effective_threshold = 0.85

    buckets = [[question(f'question number {i}?')] for i in range(4)]
    sim = np.array([
        [1.00, 0.90, 0.90, 0.82],
        [0.90, 1.00, 0.90, 0.82],
        [0.90, 0.90, 1.00, 0.82],
        [0.82, 0.82, 0.82, 1.00],
    ])
    result, rescued = analyzer._rescue_singletons(
        [[0, 1, 2], [3]], sim, buckets, verifier=lambda a, b: True)
    assert rescued == {3}
    assert sorted(len(c) for c in result) == [4]

    # 0.78 avg would clear the PAIR margin (0.75) but not the large-group
    # window (0.80): the trio must not absorb it
    sim_far = sim.copy()
    sim_far[3, :3] = sim_far[:3, 3] = 0.78
    result, rescued = analyzer._rescue_singletons(
        [[0, 1, 2], [3]], sim_far, buckets, verifier=lambda a, b: True)
    assert rescued == set()
    assert sorted(len(c) for c in result) == [1, 3]


def test_lexical_overlap_surfaces_borderline_pair_for_verification(monkeypatch):
    """An embedding model can score genuine paraphrases far below the bar.
    Clusters sharing most of their distinctive content words reach the
    verifier even when cosine says they are not close — the verifier still
    decides."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    # Cosine 0.72: far outside the 0.03 verify margin
    fake = np.array([[1.0, 0.0], [0.72, float(np.sqrt(1 - 0.72 ** 2))]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('How do I purge files older than 30 days?'),
                 question('Can we purge files older than 30 days automatically?')]

    asked = []

    def verifier(a, b):
        asked.append((a, b))
        return True

    groups = analyzer.group_similar_questions(questions, verifier=verifier)
    assert len(asked) == 1
    assert len(groups) == 1

    # Verifier says no -> they stay apart (the widening only nominates)
    groups = analyzer.group_similar_questions(questions,
                                              verifier=lambda a, b: False)
    assert len(groups) == 2


def test_lexical_widening_ignores_pairs_with_different_subjects(monkeypatch):
    """Low cosine AND low content-word overlap: the verifier is never
    consulted — cheap pass stays cheap."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    fake = np.array([[1.0, 0.0], [0.72, float(np.sqrt(1 - 0.72 ** 2))]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    def never(a, b):
        raise AssertionError('different-subject pair must not reach the verifier')

    questions = [question('How do I purge files older than 30 days?'),
                 question('Where can I download the audit report for May?')]
    groups = analyzer.group_similar_questions(questions, verifier=never)
    assert len(groups) == 2


def test_lexical_candidates_have_their_own_budget(monkeypatch):
    """Lexical candidates sit below the cosine band by construction —
    ranked inside the shared cosine cap they'd always be truncated. Their
    separate budget must survive even a zeroed cosine cap."""
    monkeypatch.setenv('LLM_VERIFY_MAX', '0')
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    fake = np.array([[1.0, 0.0], [0.72, float(np.sqrt(1 - 0.72 ** 2))]])
    monkeypatch.setattr(analyzer, 'get_embeddings_batch',
                        lambda texts, progress_callback=None: fake)

    questions = [question('How do I purge files older than 30 days?'),
                 question('Can we purge files older than 30 days automatically?')]
    groups = analyzer.group_similar_questions(questions,
                                              verifier=lambda a, b: True)
    assert len(groups) == 1


def test_dedup_is_question_mark_insensitive(monkeypatch):
    """The same ask with and without its '?' is ONE question — tier-1 keys
    must not split on trailing punctuation."""
    analyzer = make_analyzer(monkeypatch)
    calls = []
    monkeypatch.setattr(analyzer, '_ollama_embedding',
                        lambda text: calls.append(text) or [1.0, 0.0])
    questions = [question('How do I disable the scan?'),
                 question('how do i disable the scan')]
    groups = analyzer.group_similar_questions(questions)
    assert len(groups) == 1
    assert groups[0]['count'] == 2
    assert calls == []  # merged at tier 1: zero embeddings


def test_preflight_probes_the_endpoint_and_raises_when_down(monkeypatch):
    """Fail fast BEFORE 30 minutes of LLM extraction, not after."""
    monkeypatch.setenv('EMBEDDING_PREFLIGHT', 'on')
    analyzer = make_analyzer(monkeypatch)
    analyzer.MAX_RETRIES = 1

    probed = []
    monkeypatch.setattr(analyzer, '_ollama_embedding',
                        lambda text: probed.append(text) or [1.0, 0.0])
    analyzer.preflight()
    assert probed  # a REAL round-trip, not a cache read

    def boom(text):
        raise EmbeddingError('connection refused')

    monkeypatch.setattr(analyzer, '_ollama_embedding', boom)
    with pytest.raises(EmbeddingError):
        analyzer.preflight()

    monkeypatch.setenv('EMBEDDING_PREFLIGHT', 'off')
    analyzer.preflight()  # switched off: no probe, no raise


def _template_corpus(a_text, b_text, n_fillers=10):
    """Buckets where 'mft'/'saas' are corpus-common template words and the
    two texts under test share nothing else — the 2026-07-07 field shape."""
    texts = [a_text, b_text] + [
        f'How to configure the mft saas widget{i} feature?'
        for i in range(n_fillers)]
    return [[question(t)] for t in texts]


def test_verifier_yes_cannot_template_merge_disjoint_subjects(monkeypatch):
    """Field regression (2026-07-07 run): a supported-token check and a
    vault-integration ask merged at avg 0.77 on product-name scaffolding,
    with the verifier approving. A cosine pair sharing NOT ONE distinctive
    subject word loses the margin discount — the merge must clear the full
    bar, which template merges by definition cannot."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    monkeypatch.setenv('LLM_VERIFY_MARGIN', '0.05')
    analyzer.effective_threshold = 0.85

    buckets = _template_corpus(
        'Does wm mft saas support container-level azure tokens?',
        'Are there alternative solutions for integrating mft saas with hashicorp vault?')
    n = len(buckets)
    sim = np.full((n, n), 0.1)
    np.fill_diagonal(sim, 1.0)
    sim[0][1] = sim[1][0] = 0.83  # inside the margin band, below the bar

    clusters = [[i] for i in range(n)]
    merged = analyzer._merge_borderline_clusters(
        clusters, sim, buckets, verifier=lambda a, b: True)
    homes = [c for c in merged if 0 in c or 1 in c]
    assert len(homes) == 2  # still two separate groups

    # Control: same numbers, but the pair shares a distinctive subject
    # word ('azure') — the discount stays and the verifier's yes merges
    buckets = _template_corpus(
        'Does wm mft saas support container-level azure tokens?',
        'Why does azure token auth fail for mft saas containers?')
    clusters = [[i] for i in range(n)]
    merged = analyzer._merge_borderline_clusters(
        clusters, sim, buckets, verifier=lambda a, b: True)
    homes = [c for c in merged if 0 in c or 1 in c]
    assert len(homes) == 1  # merged into one group


def test_rescue_requires_a_shared_distinctive_subject_word(monkeypatch):
    """Rescue exists for rewordings, and rewordings share at least one
    distinctive subject word — a singleton sharing only template words
    with its nearest group never reaches the verifier."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')
    analyzer.effective_threshold = 0.85
    monkeypatch.setenv('LLM_RESCUE_MARGIN', '0.1')

    buckets = _template_corpus(
        'Does wm mft saas support container-level azure tokens?',
        'Why is azure container token auth failing on mft saas?') \
        + [[question('Are there alternative solutions for integrating '
                     'mft saas with hashicorp vault?')]]
    n = len(buckets)
    vault = n - 1
    sim = np.full((n, n), 0.1)
    np.fill_diagonal(sim, 1.0)
    sim[0][1] = sim[1][0] = 0.9          # the established pair
    sim[0][vault] = sim[vault][0] = 0.78  # inside the rescue window...
    sim[1][vault] = sim[vault][1] = 0.78  # ...on average too

    asked = []

    def verifier(a, b):
        asked.append((a, b))
        return True

    clusters = [[0, 1]] + [[i] for i in range(2, n)]
    result, rescued = analyzer._rescue_singletons(clusters, sim, buckets, verifier)
    assert rescued == set()
    assert asked == []  # guard fired before spending a verifier call

    # Control: the singleton is a genuine reworded 4th occurrence sharing
    # 'azure'/'token' — the guard passes and the verifier's yes rescues it
    buckets[vault] = [question('Any way around the azure token '
                               'authorization failure in mft saas?')]
    clusters = [[0, 1]] + [[i] for i in range(2, n)]
    result, rescued = analyzer._rescue_singletons(clusters, sim, buckets, verifier)
    assert rescued == {vault}


def test_poisoned_bank_centroid_cannot_reclaim_unrelated_questions(monkeypatch):
    """Field regression (2026-07-08): a bank centroid BLENDED from a past
    over-merge sits between several unrelated asks; both score >=0.85
    against it, so it re-claims them as one group in every future analysis,
    bypassing clustering, verify, and rescue. A claimed member sharing not
    one distinctive subject word with the rest of its claim is released."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')

    buckets = _template_corpus(
        'Does wm mft saas support container-level azure tokens?',
        'Are there alternative solutions for integrating mft saas with hashicorp vault?')
    n = len(buckets)
    # q0 and q1 are dissimilar to each other (0.6) but the poisoned
    # centroid (their blend) scores ~0.89 against BOTH
    embeddings = np.array([[1.0, 0.0], [0.6, 0.8]] + [[0.0, 1.0]] * (n - 2))
    poisoned = {'topic': 'Supported Protocols', 'centroid': [1.6, 0.8]}

    claims = analyzer._claim_known_topics(embeddings, [poisoned], buckets)
    assert claims == []  # both members released back to clustering

    # Control: a HEALTHY topic whose two claimed questions share their
    # subject ('azure token') keeps its claim
    buckets = _template_corpus(
        'Does wm mft saas support container-level azure tokens?',
        'Why does azure token auth fail on mft saas?')
    claims = analyzer._claim_known_topics(embeddings, [poisoned], buckets)
    assert claims == [[0, 1]]

    # Without buckets (legacy callers), claiming works as before
    claims = analyzer._claim_known_topics(embeddings, [poisoned])
    assert claims == [[0, 1]]


def test_two_topic_poisoned_claim_splits_into_components(monkeypatch):
    """Audit regression: a centroid blended from an over-merge of TWO
    topics claims two internally-coherent halves — a member-vs-rest test
    never releases anyone (each member finds its partner). Connected
    components split the claim into its coherent halves instead."""
    analyzer = make_analyzer(monkeypatch, threshold='0.85')

    texts = [
        'How do we rotate the ssh key for the transfer user?',
        'Where is the ssh key configured for transfers?',
        'Why did the timezone shift after the dst change?',
        'Which timezone setting controls the dst change display?',
    ] + [f'How to configure the mft saas widget{i} feature?' for i in range(8)]
    buckets = [[question(t)] for t in texts]
    n = len(buckets)
    # All four score >= 0.85 against the poisoned blended centroid
    embeddings = np.array([[1.0, 0.45], [1.0, 0.5], [0.45, 1.0], [0.5, 1.0]]
                          + [[-1.0, 0.0]] * (n - 4))
    poisoned = {'topic': 'Mixed Topic', 'centroid': [1.0, 1.0]}

    claims = analyzer._claim_known_topics(embeddings, [poisoned], buckets)
    assert sorted(claims) == [[0, 1], [2, 3]]  # split, nothing lost
