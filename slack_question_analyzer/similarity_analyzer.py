"""
Similarity analysis module using AI embeddings.
Local-only by design: Ollama is the sole provider — free, private, and
no transcript ever leaves the machine.
Groups similar questions together using semantic similarity.
"""

import os
import re
import time
import logging
from typing import List, Dict, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from .ollama_http import raise_with_detail, friendly_failure

from .disk_cache import JsonDiskCache
from .textutil import prefix_match, stem

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embeddings could not be retrieved from the provider."""


class EmbeddingCache(JsonDiskCache):
    """
    Persistent embedding cache backed by a JSON file.

    Embeddings never change for a given (model, text) pair, so caching them on
    disk makes repeat analyses near-instant and avoids paying for the same
    API calls twice.
    """

    def __init__(self, provider: str, model: str, cache_dir: Optional[str] = None,
                 enabled: bool = True):
        cache_dir = cache_dir or os.getenv('EMBEDDING_CACHE_DIR', '.embedding_cache')
        super().__init__(provider, model, cache_dir, enabled=enabled,
                         max_entries=int(os.getenv('EMBEDDING_CACHE_MAX', '20000')))


class SimilarityAnalyzer:
    """Analyzes question similarity using AI embeddings."""

    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 1.0
    REQUEST_TIMEOUT_SECONDS = 30

    def __init__(self, provider: Literal['ollama'] = 'ollama',
                 use_disk_cache: bool = True, threshold: Optional[float] = None):
        """
        Initialize the similarity analyzer.

        Args:
            provider: kept for call-site compatibility; only 'ollama' is
                      supported (this tool is local-only by design)
            use_disk_cache: Persist embeddings to disk so repeat runs are fast
            threshold: Similarity threshold (0-1). Overrides the
                       SIMILARITY_THRESHOLD env variable when given.
        """
        load_dotenv()

        if provider != 'ollama':
            raise ValueError(
                f"Unknown provider '{provider}'. This tool runs entirely on "
                f"local Ollama — cloud providers are not supported."
            )

        self.provider = provider
        # 'Pinned' means the user chose a threshold (param or env). Unpinned
        # thresholds start at a model-aware default and may auto-adjust when
        # nothing groups — similarity scales differ between embedding models.
        if threshold is not None:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
            self.similarity_threshold = float(threshold)
            self.threshold_pinned = True
        elif os.getenv('SIMILARITY_THRESHOLD'):
            self.similarity_threshold = self._read_threshold()
            self.threshold_pinned = True
        else:
            # Local models (nomic etc.) score paraphrases lower than ada-002
            # Field-calibrated on real MFT transcripts: in a single-domain
            # channel, nomic scores UNRELATED questions ~0.65-0.72, so the
            # threshold must sit above that noise band
            self.similarity_threshold = 0.85
            self.threshold_pinned = False
        self.threshold_auto_adjusted = False
        # Effective bar actually used for grouping: the threshold, raised above
        # the corpus's measured noise level when the corpus is dense
        self.effective_threshold = self.similarity_threshold
        self.noise_gate = None

        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
        self.embedding_model = os.getenv('OLLAMA_MODEL', 'nomic-embed-text')

        cache_enabled = use_disk_cache and os.getenv('EMBEDDING_CACHE', 'on').lower() not in ('off', '0', 'false')
        self.embeddings_cache = EmbeddingCache(
            provider=provider,
            model=self.embedding_model,
            enabled=cache_enabled
        )

        # nomic-embed-text is trained with task prefixes; embedding without one
        # degrades quality. 'clustering:' matches our use case.
        self.embed_prefix = ('clustering: '
                             if self.embedding_model.startswith('nomic-embed-text')
                             else '')

        # Pairwise similarity stats from the most recent grouping run, used to
        # suggest a better threshold when nothing groups
        self.last_similarity_stats = None

    @staticmethod
    def _read_threshold() -> float:
        raw = os.getenv('SIMILARITY_THRESHOLD', '0.85')
        try:
            threshold = float(raw)
        except ValueError:
            raise ValueError(f"SIMILARITY_THRESHOLD must be a number, got '{raw}'") from None
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"SIMILARITY_THRESHOLD must be between 0 and 1, got {threshold}")
        return threshold

    def _with_retries(self, fn, description: str):
        """Run fn() with retries and exponential backoff on transient errors."""
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    logger.warning("%s failed (attempt %d/%d): %s. Retrying in %.0fs...",
                                   description, attempt, self.MAX_RETRIES, e, delay)
                    time.sleep(delay)
        raise EmbeddingError(
            f"{description} failed after {self.MAX_RETRIES} attempts: "
            f"{friendly_failure(last_error, self.ollama_url)}") from last_error

    def _ollama_embedding(self, text: str) -> List[float]:
        """Fetch a single embedding from Ollama with timeout and retries."""
        def call():
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=self.REQUEST_TIMEOUT_SECONDS
            )
            # Carry Ollama's own error sentence ("model requires more system
            # memory...") instead of a bare "500 Server Error"
            raise_with_detail(response)
            data = response.json()
            if 'embedding' not in data or not data['embedding']:
                raise EmbeddingError(
                    f"Ollama returned no embedding (is model "
                    f"'{self.embedding_model}' pulled? Try: ollama pull {self.embedding_model})"
                )
            return data['embedding']

        return self._with_retries(call, f"Ollama embedding request ({self.ollama_url})")

    def preflight(self):
        """One real embedding round-trip, bypassing the cache: proves the
        endpoint works BEFORE expensive upstream stages spend minutes on
        work that would be lost when grouping can't start. Raises
        EmbeddingError (after the standard retries) when it can't.
        EMBEDDING_PREFLIGHT=off disables the probe."""
        if os.getenv('EMBEDDING_PREFLIGHT', 'on').lower() in ('off', '0', 'false'):
            return
        self._ollama_embedding('preflight probe')

    def _get_ollama_embeddings_parallel(self, texts: List[str], max_workers: int = 5,
                                        on_each=None):
        """
        Get embeddings from Ollama in parallel for better performance.

        Args:
            texts: List of texts to embed
            max_workers: Number of parallel requests (default: 5)
            on_each: Optional callback invoked after each embedding completes

        Raises:
            EmbeddingError: If any embedding could not be retrieved
        """
        errors = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._ollama_embedding, text): text for text in texts}

            for future in as_completed(futures):
                text = futures[future]
                try:
                    self.embeddings_cache.set(text, future.result())
                except Exception as e:
                    errors.append((text, e))
                if on_each:
                    on_each()

        if errors:
            sample_text, sample_error = errors[0]
            raise EmbeddingError(
                f"Failed to embed {len(errors)} of {len(texts)} texts. "
                f"First failure ('{sample_text[:60]}...'): {sample_error}"
            )

    def get_embeddings_batch(self, texts: List[str], progress_callback=None) -> np.ndarray:
        """
        Get embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            progress_callback: Optional fn(completed, total) called as
                               embeddings are fetched

        Returns:
            2D numpy array of embeddings

        Raises:
            EmbeddingError: If embeddings could not be retrieved
        """
        if not texts:
            return np.empty((0, 0))

        # Model-specific task prefix (cache keys include it, so switching
        # prefixes never reuses stale vectors)
        if self.embed_prefix:
            texts = [self.embed_prefix + text for text in texts]

        # Embed each unique text only once
        unique_uncached = []
        seen = set()
        for text in texts:
            if text not in seen and self.embeddings_cache.get(text) is None:
                unique_uncached.append(text)
                seen.add(text)

        if unique_uncached:
            logger.info("Fetching %d new embeddings (%d cached)...",
                        len(unique_uncached), len(texts) - len(unique_uncached))

        total = len(unique_uncached)
        completed = 0

        def report():
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

        if progress_callback:
            progress_callback(0, total)

        # Ollama only supports one prompt per request; small parallel batches
        batch_size = 10
        for i in range(0, len(unique_uncached), batch_size):
            batch = unique_uncached[i:i + batch_size]
            self._get_ollama_embeddings_parallel(batch, on_each=report)

        embeddings = []
        for text in texts:
            embedding = self.embeddings_cache.get(text)
            if embedding is None:
                raise EmbeddingError(f"Missing embedding for text: '{text[:80]}'")
            embeddings.append(embedding)

        # Save AFTER the read-back: save() may evict oldest entries on a full
        # cache, and evicting something this run still needs would turn a
        # cache hit into a crash
        self.embeddings_cache.save()

        return np.array(embeddings)

    @staticmethod
    def _lexical_similarity(text1: str, text2: str) -> float:
        """Token-set Jaccard similarity — a cheap, AI-free first pass."""
        tokens1, tokens2 = set(text1.split()), set(text2.split())
        if not tokens1 or not tokens2:
            return 0.0
        return len(tokens1 & tokens2) / len(tokens1 | tokens2)

    def _dedupe_questions(self, questions: List[Dict]) -> List[List[Dict]]:
        """
        Merge exact and near-duplicate questions WITHOUT any AI calls.

        Tier 1: identical normalized text — '?'-insensitive, because the
                same ask routinely arrives with and without its question
                mark ("can someone help me set up the agent").
        Tier 2: token-set Jaccard similarity >= LEXICAL_DEDUP_THRESHOLD
                (default 0.9 — strict enough that only rewordings merge).

        Returns buckets of questions; only one embedding is needed per bucket.
        """
        # Tier 1: exact duplicates
        buckets: Dict[str, List[Dict]] = {}
        order = []
        for q in questions:
            key = q['normalized_text'].rstrip('?! ').strip() \
                or q['normalized_text']
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(q)

        # Tier 2: lexical near-duplicates. Tokenize each key ONCE (profiled:
        # per-pair re-tokenization made this pass 97% of grouping time at
        # 2500 questions), and skip size-incompatible pairs with the EXACT
        # bound Jaccard(A,B) <= min(|A|,|B|)/max(|A|,|B|) — no candidate
        # above the threshold is ever skipped.
        lexical_threshold = float(os.getenv('LEXICAL_DEDUP_THRESHOLD', '0.9'))
        canonical = []  # (key, token frozenset, token count)
        for key in order:
            tokens = frozenset(key.split())
            n = len(tokens)
            target = None
            for ckey, ctokens, cn in canonical:
                lo, hi = (n, cn) if n <= cn else (cn, n)
                if lo == 0 or lo < lexical_threshold * hi:
                    continue
                if len(tokens & ctokens) / len(tokens | ctokens) >= lexical_threshold:
                    target = ckey
                    break
            if target is not None:
                buckets[target].extend(buckets.pop(key))
            else:
                canonical.append((key, tokens, n))

        return [buckets[key] for key, _, _ in canonical]

    def group_similar_questions(self, questions: List[Dict],
                                progress_callback=None, verifier=None,
                                auditor=None, known_topics=None,
                                fixed_threshold=None) -> List[Dict]:
        """
        Group similar questions together.

        Exact and near-duplicate questions are merged with cheap string
        comparison first, so the AI provider is only called for genuinely
        distinct questions. When a verifier is given, group pairs whose
        similarity falls just below the threshold are double-checked by it.

        Args:
            questions: List of question dictionaries with 'text' and 'normalized_text'
            progress_callback: Optional fn(completed, total) for embedding progress
            verifier: Optional fn(texts_a, texts_b) -> Optional[bool] used to
                      decide borderline merges (e.g. an LLM yes/no check)
            auditor: Optional fn(texts) -> Optional[list] evicting outliers
                     from formed groups
            known_topics: Optional bank entries that claim questions by
                          classification before clustering
            fixed_threshold: Use exactly this bar — no noise gate, no
                             auto-adjust. For clustering WITHIN a routed
                             bucket: the corpus there is topically coherent
                             by construction, so the adaptive gate (built to
                             keep unrelated topics apart) would wrongly push
                             the bar above genuine sub-groups.

        Returns:
            List of question groups with representative questions
        """
        if not questions:
            return []

        logger.info("Analyzing %d questions...", len(questions))
        self.last_similarity_stats = None
        self.threshold_auto_adjusted = False
        self.effective_threshold = fixed_threshold or self.similarity_threshold
        self.noise_gate = None

        # Tiers 1-2: merge duplicates without AI
        buckets = self._dedupe_questions(questions)
        if len(buckets) < len(questions):
            logger.info("Deduplicated to %d distinct questions (%d duplicates merged without AI)",
                        len(buckets), len(questions) - len(buckets))

        # A single distinct question needs no embeddings at all
        if len(buckets) == 1:
            if progress_callback:
                progress_callback(0, 0)
            bucket = buckets[0]
            return [{
                'representative_question': bucket[0]['text'],
                'questions': bucket,
                'count': len(bucket),
                'avg_similarity': 1.0
            }]

        # Tier 3: semantic grouping — embed one representative per bucket
        texts = [bucket[0]['normalized_text'] for bucket in buckets]
        embeddings = self.get_embeddings_batch(texts, progress_callback=progress_callback)

        large_threshold = int(os.getenv('LARGE_CLUSTERING_THRESHOLD', '2000'))
        if len(buckets) > large_threshold:
            # Too many distinct questions for an n x n similarity matrix:
            # use memory-safe leader clustering instead
            logger.info("Large corpus (%d distinct questions): using leader "
                        "clustering (no full similarity matrix)", len(buckets))
            if verifier is not None:
                logger.info("Skipping LLM borderline verification on large corpora")
            groups = self._group_large(buckets, embeddings)
        else:
            # Calculate similarity matrix between distinct questions
            similarity_matrix = cosine_similarity(embeddings)
            self.last_similarity_stats = self._similarity_stats(similarity_matrix)

            # NOTE: the adaptive noise gate and the auto-adjust below only
            # apply when NO taxonomy is configured (the default deployment
            # ships taxonomy.json, which pins a fixed bar) — tuning
            # NOISE_GATE_MARGIN has no effect on taxonomy runs.
            # The grouping bar is RELATIVE to the corpus's measured noise:
            # in a single-domain channel, nomic scores unrelated questions
            # anywhere up to ~0.8+, so any fixed threshold eventually fails.
            # The bar rises above the bulk of pairwise similarities (p90).
            self.effective_threshold = (fixed_threshold if fixed_threshold
                                        else self._gated_threshold(len(buckets)))

            # Funnel stage 1: known categories claim questions directly
            claimed_clusters, claimed = [], set()
            if known_topics:
                claimed_clusters = self._claim_known_topics(embeddings, known_topics)
                claimed = {i for c in claimed_clusters for i in c}
                if claimed_clusters:
                    logger.info("Known topics claimed %d question(s) into %d "
                                "group(s) before clustering",
                                len(claimed), len(claimed_clusters))

            clusters = self._cluster_buckets(
                len(buckets), similarity_matrix, exclude=claimed,
                # The subject gate DEFERS narrow template-merges to the
                # verifier — without a verifier there is nobody to defer
                # to, and the gate would just silently under-merge
                buckets=buckets if verifier is not None else None)

            # Auto-adjust: when the user didn't pin a threshold and nothing
            # merged, re-cluster just below the most similar pair — but ONLY
            # if that pair clearly stands out from the bulk, and never below
            # the noise gate.
            stats = self.last_similarity_stats
            if (not self.threshold_pinned and not claimed_clusters
                    and not fixed_threshold
                    and all(len(c) == 1 for c in clusters)
                    and stats and stats['max'] < self.effective_threshold):
                separation = stats['max'] - stats['p90']
                adjusted = round(stats['max'] - 0.02, 2)
                floor = max(0.5, self.noise_gate or 0.0)
                if separation >= 0.04 and adjusted >= floor:
                    logger.info("No groups at bar %.2f; auto-adjusting to "
                                "%.2f (top pair %.3f stands out from the bulk, "
                                "p90 %.3f)", self.effective_threshold, adjusted,
                                stats['max'], stats['p90'])
                    # Adjust only the per-run effective bar. Mutating the
                    # configured threshold here would leak one corpus's
                    # adjustment into the next analysis on a reused analyzer
                    # (and drift the topic-bank floor with it).
                    self.effective_threshold = adjusted
                    self.threshold_auto_adjusted = True
                    clusters = self._cluster_buckets(
                        len(buckets), similarity_matrix, exclude=claimed,
                        buckets=buckets if verifier is not None else None)
                else:
                    logger.info("No groups at bar %.2f and NOT auto-adjusting: "
                                "top pair (%.3f) sits inside the noise band "
                                "(p90 %.3f) — these questions are about "
                                "genuinely different topics",
                                self.effective_threshold, stats['max'], stats['p90'])

            clusters = claimed_clusters + clusters

            # Tier 4: LLM double-check for merges embeddings couldn't decide
            if verifier is not None and len(clusters) > 1:
                clusters = self._merge_borderline_clusters(clusters, similarity_matrix,
                                                           buckets, verifier)

            # Rescue stranded singletons: a question numerically just under
            # the bar may still belong to an existing group — the verifier
            # decides. Surgical by design: nearest group ONLY, conservative
            # verifier bias, abstain -> stays a singleton.
            rescued = set()
            if verifier is not None:
                clusters, rescued = self._rescue_singletons(
                    clusters, similarity_matrix, buckets, verifier)

            # Final QC: the LLM audits every formed group (any size, however
            # it formed — embeddings OR a borderline merge) and evicts clear
            # outliers. Embeddings not trained on the domain score some
            # unrelated pairs as high as true pairs; the audit is the decider.
            if auditor is not None:
                clusters = self._audit_clusters(clusters, buckets, auditor,
                                                verifier=verifier,
                                                rescued=rescued)

            groups = [self._build_group(indices, buckets, embeddings, similarity_matrix)
                      for indices in clusters]

        # Rank by frequency; break ties by cohesion so equal-count groups
        # have a deterministic, defensible order
        groups.sort(key=lambda x: (-x['count'], -x['avg_similarity']))

        sizes = [(g['count'], round(g['avg_similarity'], 3)) for g in groups[:8]]
        logger.info("Grouping bar %.3f (threshold %.2f%s) -> groups (count, avg): %s",
                    self.effective_threshold, self.similarity_threshold,
                    f", noise gate {self.noise_gate:.3f}" if self.noise_gate else "",
                    sizes)

        return groups

    MIN_BUCKETS_FOR_GATE = 8  # p90 is too noisy on tiny corpora

    def _gated_threshold(self, n_buckets: int) -> float:
        """
        The bar actually used for grouping: the configured threshold, raised
        above the corpus's pairwise-similarity bulk (p90 + margin) when the
        corpus is dense. Self-calibrating across embedding models and domains.
        """
        self.noise_gate = None
        stats = self.last_similarity_stats
        if not stats or n_buckets < self.MIN_BUCKETS_FOR_GATE:
            return self.similarity_threshold

        margin = float(os.getenv('NOISE_GATE_MARGIN', '0.05'))
        gate = round(min(stats['p90'] + margin, 0.95), 3)
        if gate > self.similarity_threshold:
            self.noise_gate = gate
            logger.info("Dense corpus (p90 pairwise similarity %.3f): raising "
                        "the grouping bar from %.2f to %.3f so unrelated "
                        "same-domain questions don't merge",
                        stats['p90'], self.similarity_threshold, gate)
            return gate
        return self.similarity_threshold

    def _group_large(self, buckets: List[List[Dict]], embeddings) -> List[Dict]:
        """
        Leader clustering for large corpora: each question is compared to
        cluster centroids only — O(n*k) time and O(n*d) memory — instead of
        building the full O(n^2) similarity matrix.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = embeddings / norms

        clusters: List[List[int]] = []
        sums = []         # running (un-normalized) sum of unit vectors per cluster
        centroids = None  # normalized centroid matrix, updated incrementally

        for i in range(len(unit)):
            if clusters:
                sims = centroids @ unit[i]
                best = int(np.argmax(sims))
                if sims[best] >= self.effective_threshold:
                    clusters[best].append(i)
                    sums[best] += unit[i]
                    centroids[best] = sums[best] / np.linalg.norm(sums[best])
                    continue
            clusters.append([i])
            sums.append(unit[i].copy())
            centroids = (unit[i].copy().reshape(1, -1) if centroids is None
                         else np.vstack([centroids, unit[i]]))

        groups = []
        for indices in clusters:
            group_questions = [q for idx in indices for q in buckets[idx]]
            vectors = unit[indices]
            centroid = np.mean(vectors, axis=0)

            distances = np.linalg.norm(vectors - centroid, axis=1)
            representative = buckets[indices[int(np.argmin(distances))]][0]['text']

            if len(indices) > 1:
                # Pairwise similarity within the cluster (sampled when huge)
                sample = vectors if len(indices) <= 200 else vectors[:200]
                sims = sample @ sample.T
                upper = sims[np.triu_indices(len(sample), k=1)]
                avg_sim = float(np.mean(upper)) if upper.size else 1.0
            else:
                avg_sim = 1.0

            groups.append({
                'representative_question': representative,
                'questions': group_questions,
                'count': len(group_questions),
                'avg_similarity': avg_sim
            })

        return groups

    @staticmethod
    def _similarity_stats(similarity_matrix) -> Dict:
        """
        Distribution of pairwise similarities between distinct questions.
        Lets users (and the UI) see whether the threshold fits their
        embedding model — similarity scales vary a lot between models.
        """
        n = similarity_matrix.shape[0]
        pairs = similarity_matrix[np.triu_indices(n, k=1)]
        if pairs.size == 0:
            return None
        return {
            'max': round(float(np.max(pairs)), 3),
            'p90': round(float(np.percentile(pairs, 90)), 3),
            'median': round(float(np.median(pairs)), 3),
        }

    def _claim_known_topics(self, embeddings, known_topics: List[Dict]) -> List[List[int]]:
        """
        Funnel stage 1: known categories (the learned topic bank, seeded with
        curated domain topics) claim questions by classification. Two domain
        questions can score anywhere against EACH OTHER in a generic embedding
        space, but both scoring high against the same curated category is a
        much stronger signal — so those become a group directly.

        Only categories claiming 2+ questions form groups; single claims are
        released back to normal clustering so they can still pair up there.
        """
        threshold = float(os.getenv('BANK_MATCH_THRESHOLD', '0.85'))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = embeddings / norms

        centroids = []
        for topic in known_topics:
            c = np.asarray(topic.get('centroid'), dtype=float)
            norm = np.linalg.norm(c)
            ok = norm > 0 and c.shape == unit[0].shape
            centroids.append(c / norm if ok else None)

        by_topic: Dict[int, List[int]] = {}
        for i in range(len(unit)):
            best, best_sim = None, threshold
            for k, c in enumerate(centroids):
                if c is None:
                    continue
                sim = float(unit[i] @ c)
                if sim >= best_sim:
                    best, best_sim = k, sim
            if best is not None:
                by_topic.setdefault(best, []).append(i)

        return [indices for indices in by_topic.values() if len(indices) >= 2]

    def _cluster_buckets(self, n: int, similarity_matrix,
                         exclude=frozenset(),
                         buckets: Optional[List[List[Dict]]] = None) -> List[List[int]]:
        """
        Greedy average-link clustering of bucket indices.

        A question joins a group only when its AVERAGE similarity to every
        existing member clears the threshold — not just its best match.
        Single-link (max) chaining merges everything in domain-homogeneous
        corpora (every question mentions the same product), producing one
        giant mixed group; average-link keeps clusters tight.

        Subject gate (when `buckets` is provided): a join that clears the
        bar only NARROWLY (within LLM_VERIFY_MARGIN above it) while sharing
        almost no subject words with the group (< MERGE_SUBJECT_MIN) is
        DEFERRED, not taken — shared template scaffolding ('Does wM MFT
        support X?') inflates cosine between questions about different
        things. Deferred pairs stay within the borderline-verify window
        (best-sim ≥ bar ≥ bar − margin), so the verifier makes the call
        instead of the template. MERGE_SUBJECT_MIN=0 disables.
        """
        # Vectorized greedy scan (profiled: the per-candidate list-build +
        # np.mean cost 2s at ~2200 buckets; a running similarity SUM per
        # cluster turns each scan step into one numpy comparison — same
        # clusters, ~70x faster). avg >= threshold is tested as
        # sum >= threshold * count to avoid a divide per candidate.
        subject_min = float(os.getenv('MERGE_SUBJECT_MIN', '0.25'))
        verify_margin = float(os.getenv('LLM_VERIFY_MARGIN', '0.03'))
        gate_on = subject_min > 0 and buckets is not None
        member_tokens = None
        if gate_on:
            member_tokens = [self._subject_tokens([b], buckets)
                             for b in range(n)]
            # Corpus-common tokens ARE the template ('mft', 'support', the
            # product name) — overlap on them is what inflates the cosine
            # in the first place, so they carry zero subject evidence for
            # the gate. Same principle as the keyword scorer: corpus-wide
            # words score nothing.
            df: Dict[str, int] = {}
            for toks in member_tokens:
                for t in toks:
                    df[t] = df.get(t, 0) + 1
            common_bar = max(2, round(0.25 * n))
            common = {t for t, c in df.items() if c >= common_bar}
            member_tokens = [toks - common for toks in member_tokens]

        sim = np.asarray(similarity_matrix)
        clusters = []
        assigned = np.zeros(n, dtype=bool)
        for i in exclude:
            assigned[i] = True

        for i in range(n):
            if assigned[i]:
                continue
            group_indices = [i]
            assigned[i] = True
            group_tokens = set(member_tokens[i]) if gate_on else None
            sums = sim[i].copy()
            count = 1
            j = i + 1
            while j < n:
                window = (sums[j:] >= self.effective_threshold * count) \
                    & ~assigned[j:]
                hits = np.flatnonzero(window)
                if hits.size == 0:
                    break
                k = j + int(hits[0])
                if gate_on:
                    avg = sums[k] / count
                    if avg < self.effective_threshold + verify_margin \
                            and self._subject_overlap(
                                group_tokens, member_tokens[k]) < subject_min:
                        # Narrow numeric margin + disjoint subjects: defer
                        # to the verifier instead of template-merging
                        j = k + 1
                        continue
                group_indices.append(k)
                assigned[k] = True
                if gate_on:
                    group_tokens |= member_tokens[k]
                sums += sim[k]
                count += 1
                j = k + 1
            clusters.append(group_indices)

        return clusters

    def _rescue_singletons(self, clusters: List[List[int]], similarity_matrix,
                           buckets: List[List[Dict]], verifier):
        """
        A singleton near (but under) the bar is sometimes an under-grouped
        member of an existing cluster and sometimes genuinely unique — the
        two look identical numerically. The tell is whether the NEAREST
        group passes the one-doc-page test, so each candidate singleton is
        adjudicated by the verifier against its nearest group only:
        - never looped against all groups (that would quietly dissolve the
          uniques bucket — over-merge wearing a disguise)
        - nearness is AVERAGE similarity to the group's members (the same
          metric clustering uses) — a single close member must not pull in
          a singleton the rest of the group is far from
        - verifier keeps its conservative doubt-means-no bias
        - abstain/failure -> stays a singleton

        Returns (clusters, rescued_bucket_indices). The rescued set matters
        downstream: a rescue is one verifier YES on a borderline add, and
        if the audit then flags that same member, the judges are tied —
        the audit may undo a rescue without a second overrule round.
        """
        margin = float(os.getenv('LLM_RESCUE_MARGIN', '0.1'))
        max_checks = int(os.getenv('LLM_RESCUE_MAX', '10'))
        # Rescue was built for under-grouped PAIRS (a third thread question
        # stranded just under the bar), and pairs keep the generous margin.
        # 3+ targets are allowed too — a 4x recurrence whose 4th sits just
        # under the bar is the most common real under-count — but under a
        # TIGHTER margin: an established group that didn't catch the
        # singleton during clustering is itself evidence it differs, and
        # unbounded rescue into big groups is how mega-groups grew. Every
        # rescue stays undoable by the audit (judges tied -> undo).
        max_target = int(os.getenv('LLM_RESCUE_MAX_GROUP', '4'))
        margin_large = float(os.getenv('LLM_RESCUE_MARGIN_LARGE', '0.05'))
        multi = [i for i, c in enumerate(clusters) if 2 <= len(c) <= max_target]
        rescued = set()
        if not multi:
            return clusters, rescued

        checked = 0
        absorbed = set()
        for si, cluster in enumerate(clusters):
            if len(cluster) != 1 or checked >= max_checks:
                continue
            s = cluster[0]
            best_group, best_sim = None, -1.0
            for gi in multi:
                members = clusters[gi]
                # Live size check: an earlier rescue may have grown this
                # group past the cap already — eligibility computed once
                # up front would let a pair absorb singletons without limit
                if len(members) > max_target:
                    continue
                sim = sum(similarity_matrix[s][j] for j in members) / len(members)
                bar = self.effective_threshold - (margin if len(members) <= 2
                                                  else margin_large)
                if sim < bar:
                    continue
                if sim > best_sim:
                    best_group, best_sim = gi, sim
            if best_group is None:
                continue  # genuinely far from everything: rare, not wrong
            if not self._types_compatible(cluster, clusters[best_group], buckets) \
                    and not self._shared_subject(cluster, clusters[best_group],
                                                 buckets):
                continue  # different question kind AND different subject
            checked += 1
            group_texts = [buckets[j][0]['text'] for j in clusters[best_group][:3]]
            if verifier([buckets[s][0]['text']], group_texts) is True:
                logger.info("AI rescued a stranded question into a group: %.80r",
                            buckets[s][0]['text'])
                clusters[best_group].append(s)
                absorbed.add(si)
                rescued.add(s)
        if absorbed:
            return [c for i, c in enumerate(clusters) if i not in absorbed], rescued
        return clusters, rescued

    def _audit_clusters(self, clusters: List[List[int]],
                        buckets: List[List[Dict]], auditor,
                        verifier=None, rescued=None) -> List[List[int]]:
        """
        Final LLM audit of every formed group: evict clear outliers.

        Field finding: an embedding model that wasn't trained on the domain
        scores some unrelated pairs (metering vs. cloud tokens: 0.81) as high
        as genuine pairs (0.78-0.83) — no numeric bar can separate them. The
        auditor's domain knowledge is the decider; on uncertainty (None or
        an empty list), the numeric grouping stands. Biggest groups are
        audited first (most damage if wrong).

        Second field finding: a single model sample evicted a TRUE pair
        ("limit concurrent transfers" / "cap transfers per node"). Eviction
        is destructive, so it needs TWO judges: the auditor only nominates;
        the verifier must independently confirm the nominee is a different
        topic (explicit False). True/uncertain -> the nominee stays.

        Exception — RESCUED members: the verifier already cast its YES when
        it approved the rescue, so an audit flag makes the judges 1-1 on a
        borderline add. The merge-happy verifier must not break that tie in
        its own favor (that loop built three mega-groups across eval
        rounds): a flagged rescue is simply undone, back to a singleton.
        """
        # The audit has its OWN budget: it silently shared LLM_VERIFY_MAX,
        # so raising verify spend doubled audit spend without anyone asking
        max_checks = int(os.getenv('LLM_AUDIT_MAX',
                                   os.getenv('LLM_VERIFY_MAX', '10')))
        # Each nominee costs one verifier call OUTSIDE every documented
        # budget — cap the fan-out per group; past it, nominees STAY (the
        # safe, non-destructive default: eviction needs two judges)
        nominee_cap = int(os.getenv('LLM_AUDIT_NOMINEE_MAX', '3'))
        rescued = rescued or set()
        checked = 0
        audited = []
        for cluster in sorted(clusters, key=len, reverse=True):
            if len(cluster) >= 2 and checked < max_checks:
                checked += 1
                texts = [buckets[idx][0]['text'] for idx in cluster]
                outliers = auditor(texts)
                if outliers:
                    nominated = {cluster[i] for i in outliers}
                    evicted = set()
                    for nominee_n, idx in enumerate(sorted(nominated)):
                        if nominee_n >= nominee_cap:
                            logger.info("Audit nominee cap reached — keeping "
                                        "remaining nominee(s): %.60r",
                                        buckets[idx][0]['text'])
                            break
                        rest_texts = [buckets[j][0]['text']
                                      for j in cluster if j not in nominated][:3]
                        if idx in rescued:
                            evicted.add(idx)
                            logger.info("Audit undid a rescue (judges tied "
                                        "on a borderline add): %.90r",
                                        buckets[idx][0]['text'])
                            audited.append([idx])
                            continue
                        if verifier is not None and rest_texts:
                            verdict = verifier([buckets[idx][0]['text']], rest_texts)
                            if verdict is not False:
                                logger.info("Audit eviction overruled by the "
                                            "verifier (same topic): %.80r",
                                            buckets[idx][0]['text'])
                                # Judges tied 1-1 on this member: leave a
                                # mark so routing can act as the third
                                # judge downstream (internal, stripped
                                # before results render)
                                buckets[idx][0]['_audit_flagged'] = True
                                continue
                        evicted.add(idx)
                        logger.info("AI evicted an outlier from a group "
                                    "(both judges agree: different topic): %.90r",
                                    buckets[idx][0]['text'])
                        audited.append([idx])
                    rest = [idx for idx in cluster if idx not in evicted]
                    if rest:
                        audited.append(rest)
                    continue
            audited.append(cluster)
        return audited

    # Question types collapse into three families: asking about a capability,
    # reporting a breakage, or giving product feedback. A capability question
    # and a breakage report about the same feature are different items needing
    # different handling — LLM-assisted merges across families are vetoed.
    # This is the GENERAL rule behind every observed false merge (shared
    # vocabulary, different intent), so it needs no per-instance examples.
    TYPE_FAMILIES = {'how-to': 'capability', 'is-it-possible': 'capability',
                     'feature-request': 'feedback',
                     'troubleshooting': 'breakage', 'defect-report': 'breakage'}

    def _cluster_type_families(self, cluster: List[int],
                               buckets: List[List[Dict]]) -> set:
        families = set()
        for idx in cluster:
            for q in buckets[idx]:
                family = self.TYPE_FAMILIES.get(q.get('qtype'))
                if family:
                    families.add(family)
        return families

    def _types_compatible(self, cluster_a: List[int], cluster_b: List[int],
                          buckets: List[List[Dict]]) -> bool:
        """Untyped questions are always compatible (no information, no veto)."""
        families_a = self._cluster_type_families(cluster_a, buckets)
        families_b = self._cluster_type_families(cluster_b, buckets)
        if not families_a or not families_b:
            return True
        return bool(families_a & families_b)

    # Words that appear in nearly every support question — overlap on these
    # says nothing about a shared subject
    _SUBJECT_STOP_WORDS = frozenset("""
        a an the and or but if is are was were be been being do does did done
        can could would should will shall may might must have has had having
        get got gets getting go goes going how what when where why who whom
        which i we you it its my our your their me us them this that these
        those there here any some no not none to too for of in on at with
        without from by as into onto about over under between after before
        need needs want wants like know see set use using used make makes
        way please thanks help anyone someone possible currently team
        """.split())

    def _subject_tokens(self, cluster: List[int],
                        buckets: List[List[Dict]]) -> set:
        """Distinctive content words a cluster is about — STEMMED, like
        every other content-word metric in the pipeline (drifted copies of
        the folding are how 'timing' and 'timeout' scored zero overlap and
        the symptom/fix pair never reached the verifier)."""
        tokens = set()
        for idx in cluster:
            for q in buckets[idx]:
                text = (q.get('normalized_text') or q.get('text') or '').lower()
                for t in re.findall(r"[a-z0-9][a-z0-9\-_./]{2,}", text):
                    # Pure numbers ('2024', '500') name nothing by themselves
                    if t not in self._SUBJECT_STOP_WORDS \
                            and not t.replace('.', '').replace('/', '').isdigit():
                        tokens.add(stem(t))
        return tokens

    @staticmethod
    def _subject_overlap(a: set, b: set) -> float:
        """Prefix-tolerant containment of the smaller token set in the
        larger: 'transfers' matches 'transfer', 'timeouts' matches
        'timeout' — like _topic_grounded's stems. One metric everywhere:
        the veto override and the lexical candidate gate must not disagree
        about what "shared subject" means."""
        if not a or not b:
            return 0.0
        small, large = (a, b) if len(a) <= len(b) else (b, a)

        def matches(t: str) -> bool:
            # Guard at 3, not 4: crude stems can be 3 chars ('timing' ->
            # 'tim') and must still prefix-match their unstemmed kin
            # ('timeout'); shorter than 3 never reaches here (token regex)
            return any(t == m or (len(t) >= 3 and len(m) >= 3
                                  and prefix_match(t, m))
                       for m in large)

        return sum(1 for t in small if matches(t)) / len(small)

    def _shared_subject(self, cluster_a: List[int], cluster_b: List[int],
                        buckets: List[List[Dict]]) -> bool:
        """
        Do two clusters talk about the same named thing(s)?

        The type-family veto exists for shared-VOCABULARY false merges
        (different subjects, same product words). But a how-to and a
        breakage report about the SAME specific subject ("raise the sftp
        timeout" / "sftp transfers keep timing out") are frequently one
        topic — one answer resolves both. High overlap of distinctive
        content words is the tell that separates the two cases, and it
        downgrades the veto from "never" to "ask the verifier".
        Set SHARED_SUBJECT_MIN above 1.0 to disable the override.
        """
        min_overlap = float(os.getenv('SHARED_SUBJECT_MIN', '0.5'))
        return self._subject_overlap(
            self._subject_tokens(cluster_a, buckets),
            self._subject_tokens(cluster_b, buckets)) >= min_overlap

    def _merge_borderline_clusters(self, clusters: List[List[int]], similarity_matrix,
                                   buckets: List[List[Dict]], verifier) -> List[List[int]]:
        """
        Ask the verifier about cluster pairs whose best cross-similarity is
        near or above the threshold — pairs that average-link kept apart but
        might genuinely belong together (the LLM decides).
        """
        margin = float(os.getenv('LLM_VERIFY_MARGIN', '0.03'))
        max_checks = int(os.getenv('LLM_VERIFY_MAX', '10'))
        # Second candidate source: an embedding model scores some genuine
        # paraphrases well below the bar (field finding: same-subject
        # rewordings at 0.6-0.7), so cosine nearness alone misses them. A
        # pair of clusters sharing most of their distinctive content words
        # (same prefix-tolerant metric as the veto override) is ALSO a
        # candidate — the verifier still makes every decision. Lexical
        # candidates get their OWN budget and ranking: they sit below the
        # cosine band by construction, so ranking them by cosine inside the
        # shared cap would truncate them all whenever the corpus is busy.
        # Set LLM_VERIFY_LEXICAL_MIN above 1.0 to disable.
        lexical_min = float(os.getenv('LLM_VERIFY_LEXICAL_MIN', '0.7'))
        lexical_max = int(os.getenv('LLM_VERIFY_LEXICAL_MAX', '5'))
        # ...and their own (looser) numeric tightness floor
        lexical_slack = float(os.getenv('LLM_VERIFY_LEXICAL_SLACK', '0.15'))

        # Tokenize each cluster ONCE — computing per pair turns this loop
        # into a quadratic re-tokenization of the whole corpus
        lexical_on = lexical_min <= 1.0
        tokens = ([self._subject_tokens(c, buckets) for c in clusters]
                  if lexical_on else None)

        cosine_candidates = []
        lexical_candidates = []
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                best = max(similarity_matrix[i][j]
                           for i in clusters[a] for j in clusters[b])
                if best >= self.effective_threshold - margin:
                    cosine_candidates.append((best, a, b, 'cosine'))
                elif lexical_on:
                    overlap = self._subject_overlap(tokens[a], tokens[b])
                    if overlap >= lexical_min:
                        lexical_candidates.append((overlap, a, b, 'lexical'))
        cosine_candidates.sort(reverse=True)
        lexical_candidates.sort(reverse=True)  # by subject overlap
        candidates = (cosine_candidates[:max_checks]
                      + lexical_candidates[:lexical_max])

        if not candidates:
            return clusters

        logger.info("Verifying %d borderline group pair(s) with the LLM...", len(candidates))
        parent = list(range(len(clusters)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged_count = 0
        for _, a, b, kind in candidates:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue

            # Guard: even with an LLM yes, the merged cluster must stay
            # numerically tight — otherwise a liberal model re-creates the
            # mixed mega-groups average-link exists to prevent. Lexical
            # candidates were surfaced BECAUSE cosine underscores them, so
            # their floor is looser (their safety comes from the subject
            # overlap plus the verifier plus the audit downstream).
            combined = clusters[ra] + clusters[rb]
            pair_sims = [similarity_matrix[x][y]
                         for ix, x in enumerate(combined) for y in combined[ix + 1:]]
            floor = self.effective_threshold - (margin if kind == 'cosine'
                                                else lexical_slack)
            if float(np.mean(pair_sims)) < floor:
                continue

            # Type-family veto: a capability question never LLM-merges with
            # a breakage report on mere vocabulary overlap — but when both
            # clusters are about the SAME named subject, the veto downgrades
            # to "ask the verifier" (one answer often resolves both)
            if not self._types_compatible(clusters[ra], clusters[rb], buckets):
                if not self._shared_subject(clusters[ra], clusters[rb], buckets):
                    logger.info("Skipped a cross-type merge (different question "
                                "kinds): %.60r / %.60r",
                                buckets[clusters[ra][0]][0]['text'],
                                buckets[clusters[rb][0]][0]['text'])
                    continue
                logger.info("Cross-type pair shares its subject — letting the "
                            "verifier decide: %.60r / %.60r",
                            buckets[clusters[ra][0]][0]['text'],
                            buckets[clusters[rb][0]][0]['text'])

            texts_a = [buckets[idx][0]['text'] for idx in clusters[ra][:3]]
            texts_b = [buckets[idx][0]['text'] for idx in clusters[rb][:3]]
            if verifier(texts_a, texts_b) is True:
                parent[rb] = ra
                clusters[ra] = combined
                merged_count += 1

        if merged_count:
            logger.info("LLM verification merged %d borderline group pair(s)", merged_count)
        return [clusters[i] for i in range(len(clusters)) if find(i) == i]

    def _build_group(self, group_indices: List[int], buckets: List[List[Dict]],
                     embeddings, similarity_matrix) -> Dict:
        """Construct the result dict for one cluster of buckets."""
        # Expand buckets back into their member questions
        group_questions = [q for idx in group_indices for q in buckets[idx]]

        # Find the most representative bucket (closest to centroid)
        if len(group_indices) > 1:
            group_embeddings = embeddings[group_indices]
            centroid = np.mean(group_embeddings, axis=0)

            distances = [np.linalg.norm(emb - centroid) for emb in group_embeddings]
            representative_idx = group_indices[int(np.argmin(distances))]
            representative = buckets[representative_idx][0]['text']
        else:
            representative = buckets[group_indices[0]][0]['text']

        # Average pairwise similarity between the distinct questions
        if len(group_indices) > 1:
            similarities = []
            for a in range(len(group_indices)):
                for b in range(a + 1, len(group_indices)):
                    similarities.append(similarity_matrix[group_indices[a]][group_indices[b]])
            avg_sim = float(np.mean(similarities)) if similarities else 1.0
        else:
            avg_sim = 1.0

        return {
            'representative_question': representative,
            'questions': group_questions,
            'count': len(group_questions),
            'avg_similarity': avg_sim
        }
