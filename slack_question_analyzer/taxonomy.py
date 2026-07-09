"""
Versioned routing taxonomy: broad buckets that questions are sorted into
BEFORE any clustering, plus the fixed merge map that collapses buckets into
the final dashboard categories.

Design (living pipeline):
- Buckets are DATA (taxonomy.json), not code: edit/add buckets without
  touching the pipeline, bump 'version' when you do. Results are stamped
  with the taxonomy version that classified them.
- Routing is embeddings-first: each bucket has an 'anchor' description;
  questions go to their nearest anchor. The LLM is only consulted for
  genuinely ambiguous questions (top-2 anchors within a margin) and only
  ever answers a closed single-number choice — never open-ended clustering.
- Questions near no anchor are NOT forced into a wrong home: they are
  flagged for review and the rate is reported as a health metric.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .inputs import default_data_path

logger = logging.getLogger(__name__)

# A question must clear this similarity to its best anchor to be routed;
# below it, the question is an outlier (kept, but flagged for review)
DEFAULT_OUTLIER_FLOOR = 0.4
# Top-2 anchors closer than this -> ambiguous -> LLM adjudicates
DEFAULT_AMBIGUITY_MARGIN = 0.05
# Between the outlier floor and this, the best anchor is a WEAK match —
# off-topic and too-vague questions land here ("is the wiki down?" still
# scores ~0.5 against a connectivity anchor on shared outage vocabulary).
# Embeddings alone must not force these into the closest wrong home: the
# LLM adjudicates the closed choice, and its abstain (0) -> review pile
DEFAULT_CONFIDENCE_FLOOR = 0.55


class Taxonomy:
    """The bucket definitions loaded from taxonomy.json."""

    def __init__(self, path: Optional[str] = None, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = os.getenv('TAXONOMY', 'on').lower() not in ('off', '0', 'false')
        env_path = path or os.getenv('TAXONOMY_PATH')
        self.path = Path(env_path) if env_path else default_data_path('taxonomy.json')
        self.version = None
        self.buckets: List[Dict] = []

        if not enabled:
            return
        if not self.path.is_file():
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Wrong SHAPES (a list at top level, string buckets) must
            # disable routing with a warning like parse errors do — not
            # crash the analysis with an AttributeError
            buckets = data.get('buckets', []) if isinstance(data, dict) else []
            if buckets and isinstance(buckets, list) and all(
                    isinstance(b, dict) and b.get('id') and b.get('name')
                    and b.get('anchor') for b in buckets):
                self.buckets = buckets
                self.version = data.get('version')
            else:
                logger.warning("Taxonomy at %s is malformed (need an object "
                               "with a 'buckets' list of {id, name, anchor}); "
                               "routing disabled", self.path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Taxonomy at %s is unreadable (%s); routing disabled",
                           self.path, e)

    @property
    def enabled(self) -> bool:
        return bool(self.buckets)

    def anchor_texts(self) -> List[str]:
        return [b['anchor'] for b in self.buckets]

    def bucket_name(self, index: int) -> str:
        return self.buckets[index]['name']

    def final_category(self, index: int) -> str:
        """The broad dashboard category this bucket collapses into."""
        return self.buckets[index].get('category') or self.buckets[index]['name']

    def distinctive_tokens(self) -> List[set]:
        """Per-bucket stemmed content tokens that appear in EXACTLY ONE
        anchor. Nine long same-product anchor paragraphs embed into a
        narrow cosine band (margin-ambiguity fires constantly), yet each
        anchor carries dozens of tokens unique to it — 'quarantine',
        'metering', 'onboarding'. Those tokens are cheap deterministic
        evidence for breaking embedding ties. Computed from whatever
        taxonomy the user loaded — data-driven, never hardcoded."""
        if getattr(self, '_distinctive', None) is None:
            from .textutil import stem_tokens
            per_bucket = [stem_tokens((b.get('anchor') or '')
                                      + ' ' + (b.get('name') or ''))
                          for b in self.buckets]
            counts: Dict[str, int] = {}
            for toks in per_bucket:
                for t in toks:
                    counts[t] = counts.get(t, 0) + 1
            self._distinctive = [{t for t in toks if counts[t] == 1}
                                 for toks in per_bucket]
        return self._distinctive


def route_questions(question_embeddings, anchor_embeddings,
                    outlier_floor: Optional[float] = None,
                    ambiguity_margin: Optional[float] = None,
                    confidence_floor: Optional[float] = None):
    """
    Pure-code routing: nearest anchor per question, with TWO humility gates.
    A question is only embedding-routed when its best anchor is both strong
    (>= confidence floor) and clearly ahead of the runner-up (>= margin);
    anything weaker goes to the LLM's closed choice, where abstain -> review.

    Returns (assignments, ambiguous, outliers):
    - assignments: question index -> bucket index (confident routes only)
    - ambiguous:   [(question index, [candidate bucket indices], reason)]
                   for the LLM; reason is 'margin' (top-2 anchors too close —
                   both candidates are strong) or 'floor' (weak best anchor —
                   possibly off-topic). The reason matters when the LLM
                   budget runs out: a margin case can safely take the
                   embedding favorite, but force-routing a floor case is
                   exactly what the confidence floor exists to prevent — it
                   belongs in review.
    - outliers:    [question indices] near no anchor
    """
    if outlier_floor is None:
        outlier_floor = float(os.getenv('ROUTE_OUTLIER_FLOOR',
                                        str(DEFAULT_OUTLIER_FLOOR)))
    if ambiguity_margin is None:
        ambiguity_margin = float(os.getenv('ROUTE_AMBIGUITY_MARGIN',
                                           str(DEFAULT_AMBIGUITY_MARGIN)))
    if confidence_floor is None:
        confidence_floor = float(os.getenv('ROUTE_CONFIDENCE_FLOOR',
                                           str(DEFAULT_CONFIDENCE_FLOOR)))

    def unit(matrix):
        m = np.asarray(matrix, dtype=float)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    q = unit(question_embeddings)
    a = unit(anchor_embeddings)
    sims = q @ a.T  # questions x anchors

    assignments: Dict[int, int] = {}
    ambiguous = []
    outliers = []
    for i in range(sims.shape[0]):
        order = np.argsort(sims[i])[::-1]
        best, best_sim = int(order[0]), float(sims[i][order[0]])
        if best_sim < outlier_floor:
            outliers.append(i)
            continue
        candidates = [best]
        if sims.shape[1] > 1:
            second, second_sim = int(order[1]), float(sims[i][order[1]])
            candidates.append(second)
            if best_sim - second_sim < ambiguity_margin:
                # A floor-weak best anchor outranks mere margin ambiguity:
                # the fallback semantics differ (review vs embedding favorite)
                reason = 'floor' if best_sim < confidence_floor else 'margin'
                ambiguous.append((i, candidates, reason))
                continue
        if best_sim < confidence_floor:
            # Weak best match: off-topic or too vague to embed near any
            # anchor — the LLM decides, with abstain available
            ambiguous.append((i, candidates, 'floor'))
            continue
        assignments[i] = best
    return assignments, ambiguous, outliers
