"""
Similarity analysis module using AI embeddings.
Supports Azure OpenAI, OpenAI, and local Ollama.
Groups similar questions together using semantic similarity.
"""

import os
import time
import logging
from typing import List, Dict, Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

from .disk_cache import JsonDiskCache

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

    def __init__(self, provider: Literal['azure', 'openai', 'ollama'] = 'ollama',
                 use_disk_cache: bool = True, threshold: Optional[float] = None):
        """
        Initialize the similarity analyzer.

        Args:
            provider: AI provider to use ('azure', 'openai', or 'ollama')
            use_disk_cache: Persist embeddings to disk so repeat runs are fast
            threshold: Similarity threshold (0-1). Overrides the
                       SIMILARITY_THRESHOLD env variable when given.
        """
        load_dotenv()

        if provider not in ('azure', 'openai', 'ollama'):
            raise ValueError(
                f"Unknown provider '{provider}'. Expected 'azure', 'openai', or 'ollama'."
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

        if provider == 'azure':
            # Azure OpenAI configuration
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            if not api_key or not endpoint:
                raise EmbeddingError(
                    "Azure provider requires AZURE_OPENAI_API_KEY and "
                    "AZURE_OPENAI_ENDPOINT to be set (see 'setup' command)."
                )
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
                azure_endpoint=endpoint
            )
            self.deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
            self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
        elif provider == 'openai':
            # Standard OpenAI configuration
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise EmbeddingError(
                    "OpenAI provider requires OPENAI_API_KEY to be set "
                    "(see 'setup' command)."
                )
            self.client = OpenAI(api_key=api_key)
            self.deployment_name = None
            self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
        else:  # ollama
            # Local Ollama configuration
            self.client = None
            self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
            self.embedding_model = os.getenv('OLLAMA_MODEL', 'nomic-embed-text')
            self.deployment_name = None

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
        raise EmbeddingError(f"{description} failed after {self.MAX_RETRIES} attempts: {last_error}") from last_error

    def _ollama_embedding(self, text: str) -> List[float]:
        """Fetch a single embedding from Ollama with timeout and retries."""
        def call():
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=self.REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            data = response.json()
            if 'embedding' not in data or not data['embedding']:
                raise EmbeddingError(
                    f"Ollama returned no embedding (is model "
                    f"'{self.embedding_model}' pulled? Try: ollama pull {self.embedding_model})"
                )
            return data['embedding']

        return self._with_retries(call, f"Ollama embedding request ({self.ollama_url})")

    def _openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Fetch a batch of embeddings from OpenAI/Azure with retries."""
        model = self.deployment_name if self.provider == 'azure' else self.embedding_model

        def call():
            response = self.client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]

        return self._with_retries(call, f"{self.provider} embeddings request")

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

        # Process in batches to avoid rate limits
        batch_size = 100 if self.provider != 'ollama' else 10  # Smaller batches for Ollama

        for i in range(0, len(unique_uncached), batch_size):
            batch = unique_uncached[i:i + batch_size]

            if self.provider == 'ollama':
                # Ollama only supports one prompt per request; parallelize for speed
                self._get_ollama_embeddings_parallel(batch, on_each=report)
            else:
                for text, embedding in zip(batch, self._openai_embeddings(batch), strict=False):
                    self.embeddings_cache.set(text, embedding)
                    report()

        self.embeddings_cache.save()

        embeddings = []
        for text in texts:
            embedding = self.embeddings_cache.get(text)
            if embedding is None:
                raise EmbeddingError(f"Missing embedding for text: '{text[:80]}'")
            embeddings.append(embedding)

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

        Tier 1: identical normalized text.
        Tier 2: token-set Jaccard similarity >= LEXICAL_DEDUP_THRESHOLD
                (default 0.9 — strict enough that only rewordings merge).

        Returns buckets of questions; only one embedding is needed per bucket.
        """
        # Tier 1: exact duplicates
        buckets: Dict[str, List[Dict]] = {}
        order = []
        for q in questions:
            key = q['normalized_text']
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(q)

        # Tier 2: lexical near-duplicates
        lexical_threshold = float(os.getenv('LEXICAL_DEDUP_THRESHOLD', '0.9'))
        canonical = []
        for key in order:
            target = None
            for ckey in canonical:
                if self._lexical_similarity(key, ckey) >= lexical_threshold:
                    target = ckey
                    break
            if target is not None:
                buckets[target].extend(buckets.pop(key))
            else:
                canonical.append(key)

        return [buckets[key] for key in canonical]

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

            clusters = self._cluster_buckets(len(buckets), similarity_matrix,
                                             exclude=claimed)

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
                    self.similarity_threshold = adjusted
                    self.effective_threshold = adjusted
                    self.threshold_auto_adjusted = True
                    clusters = self._cluster_buckets(len(buckets), similarity_matrix,
                                                     exclude=claimed)
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
                         exclude=frozenset()) -> List[List[int]]:
        """
        Greedy average-link clustering of bucket indices.

        A question joins a group only when its AVERAGE similarity to every
        existing member clears the threshold — not just its best match.
        Single-link (max) chaining merges everything in domain-homogeneous
        corpora (every question mentions the same product), producing one
        giant mixed group; average-link keeps clusters tight.
        """
        clusters = []
        assigned = set(exclude)

        for i in range(n):
            if i in assigned:
                continue

            group_indices = [i]
            assigned.add(i)

            for j in range(i + 1, n):
                if j in assigned:
                    continue

                avg_similarity = float(np.mean(
                    [similarity_matrix[j][idx] for idx in group_indices]))
                if avg_similarity >= self.effective_threshold:
                    group_indices.append(j)
                    assigned.add(j)

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
        # Rescue completes under-grouped PAIRS (the field case it was built
        # for: a third thread question stranded just under the bar). An
        # ESTABLISHED 3+ group that didn't catch the singleton during
        # clustering is itself evidence the singleton differs — rescuing
        # into those is how every mega-group across nine eval rounds grew
        max_target = int(os.getenv('LLM_RESCUE_MAX_GROUP', '2'))
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
                sim = sum(similarity_matrix[s][j] for j in members) / len(members)
                if sim > best_sim:
                    best_group, best_sim = gi, sim
            if best_sim < self.effective_threshold - margin:
                continue  # genuinely far from everything: rare, not wrong
            if not self._types_compatible(cluster, clusters[best_group], buckets):
                continue  # different question kind than its nearest group
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
        max_checks = int(os.getenv('LLM_VERIFY_MAX', '10'))
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
                    for idx in sorted(nominated):
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

    def _merge_borderline_clusters(self, clusters: List[List[int]], similarity_matrix,
                                   buckets: List[List[Dict]], verifier) -> List[List[int]]:
        """
        Ask the verifier about cluster pairs whose best cross-similarity is
        near or above the threshold — pairs that average-link kept apart but
        might genuinely belong together (the LLM decides).
        """
        margin = float(os.getenv('LLM_VERIFY_MARGIN', '0.03'))
        max_checks = int(os.getenv('LLM_VERIFY_MAX', '10'))

        candidates = []
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                best = max(similarity_matrix[i][j]
                           for i in clusters[a] for j in clusters[b])
                if best >= self.effective_threshold - margin:
                    candidates.append((best, a, b))
        candidates.sort(reverse=True)
        candidates = candidates[:max_checks]

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
        for _, a, b in candidates:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue

            # Guard: even with an LLM yes, the merged cluster must stay
            # numerically tight — otherwise a liberal model re-creates the
            # mixed mega-groups average-link exists to prevent
            combined = clusters[ra] + clusters[rb]
            pair_sims = [similarity_matrix[x][y]
                         for ix, x in enumerate(combined) for y in combined[ix + 1:]]
            if float(np.mean(pair_sims)) < self.effective_threshold - margin:
                continue

            # Type-family veto: a capability question never LLM-merges with
            # a breakage report, however similar the vocabulary
            if not self._types_compatible(clusters[ra], clusters[rb], buckets):
                logger.info("Skipped a cross-type merge (different question "
                            "kinds): %.60r / %.60r",
                            buckets[clusters[ra][0]][0]['text'],
                            buckets[clusters[rb][0]][0]['text'])
                continue

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
