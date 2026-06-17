"""
Main analyzer module that orchestrates question extraction, grouping, and ranking.
"""

import os
import re
import json
import math
import logging
from pathlib import Path
from typing import List, Dict, Optional, Literal
from datetime import datetime, timezone
import numpy as np
from dotenv import load_dotenv
from .question_extractor import QuestionExtractor
from .similarity_analyzer import SimilarityAnalyzer
from .group_labeler import GroupLabeler, PROMPT_PACK_VERSION
from .topic_bank import TopicBank
from .taxonomy import Taxonomy, route_questions
from .exporters import to_csv, to_markdown

logger = logging.getLogger(__name__)


def _app_version() -> str:
    from slack_question_analyzer import __version__  # lazy: avoids circular import
    return __version__

# Caps keep the optional LLM passes fast on huge transcripts
MAX_LABELED_GROUPS = 20
MAX_DETECTED_MESSAGES = 40
DETECT_BATCH_SIZE = 8
MAX_ANSWER_CHECKS = 20
MIN_DETECT_MESSAGE_CHARS = 20
# 'auto' extraction uses the LLM for everything up to this many messages
MAX_FULL_EXTRACT_MESSAGES = 150


class QuestionAnalyzer:
    """Main analyzer class that coordinates the analysis pipeline."""

    def __init__(self, provider: Optional[Literal['azure', 'openai', 'ollama']] = None,
                 use_disk_cache: bool = True, threshold: Optional[float] = None,
                 label_groups: Optional[bool] = None):
        """
        Initialize the analyzer.

        Args:
            provider: AI provider to use ('azure', 'openai', or 'ollama').
                     If None, reads from AI_PROVIDER env variable (defaults to 'ollama')
            use_disk_cache: Persist embeddings to disk so repeat runs are fast
            threshold: Similarity threshold (0-1). Overrides the
                       SIMILARITY_THRESHOLD env variable when given.
            label_groups: Generate LLM topic labels/summaries per group.
                          None (default) reads GROUP_LABELS env: 'auto' labels
                          when a generation model is available, 'on'/'off' force it.
        """
        load_dotenv()

        if provider is None:
            provider = os.getenv('AI_PROVIDER', 'ollama')

        self.extractor = QuestionExtractor()
        self.similarity_analyzer = SimilarityAnalyzer(provider=provider,
                                                      use_disk_cache=use_disk_cache,
                                                      threshold=threshold)

        if label_groups is None:
            mode = os.getenv('GROUP_LABELS', 'auto').lower()
        else:
            mode = 'on' if label_groups else 'off'
        # GROUP_LABELS=off disables ALL LLM features (labels, verification,
        # detection, answers, executive summary)
        self.labeler = GroupLabeler(provider) if mode in ('auto', 'on') else None
        self._labels_forced = (mode == 'on')

        # Per-feature switches; each is 'auto' (use when model available),
        # 'on', or 'off'
        self._verify_mode = os.getenv('LLM_VERIFY_GROUPS', 'auto').lower()
        self._detect_mode = os.getenv('LLM_EXTRACTION', 'auto').lower()
        self._answers_mode = os.getenv('LLM_ANSWER_DETECTION', 'auto').lower()
        self._summary_mode = os.getenv('EXECUTIVE_SUMMARY', 'auto').lower()
        self._themes_mode = os.getenv('THEMES', 'auto').lower()
        self._last_routing = None  # taxonomy routing health stats, per run
        self._dropped: List[Dict] = []  # provenance trail, reset per run

    def _llm_enabled(self, mode: str) -> bool:
        """Whether an optional LLM feature should run."""
        if self.labeler is None or mode == 'off':
            return False
        if mode == 'on':
            return True
        return self._labels_forced or self.labeler.available()  # 'auto'

    def analyze_slack_content(self, content: str, progress_callback=None) -> Dict:
        """
        Analyze Slack content and return grouped questions.

        Args:
            content: Raw Slack content string
            progress_callback: Optional fn(stage, completed, total) reporting
                progress. Stages: 'extracting', 'embedding', 'grouping',
                'keywords', 'complete'. For 'embedding', completed/total count
                individual embeddings; other stages report (0, 1) then (1, 1).

        Returns:
            Dictionary containing analysis results
        """
        return self.analyze_contents([content], progress_callback=progress_callback)

    def analyze_contents(self, contents: List[str], progress_callback=None,
                         cancel_check=None) -> Dict:
        """
        Analyze one or more Slack content strings as a single corpus.

        Used for multi-file uploads (e.g. a zipped Slack export with one JSON
        file per day): messages from every file are merged before analysis.
        """
        def report(stage, completed=0, total=1):
            if progress_callback:
                progress_callback(stage, completed, total)

        report('extracting')
        self._dropped = []  # provenance: every removed question, with reason
        if self.labeler is not None:
            self.labeler.stats = {}  # fresh abstain/verdict counters per run
            # Cancel takes effect before each LLM call, not just at stage
            # boundaries — a single call can run minutes on CPU hardware
            self.labeler.cancel_check = cancel_check
        logger.info("Step 1: Extracting questions from %d file(s)...", len(contents))
        messages = []
        for content in contents:
            messages.extend(self.extractor.extract_messages(content))

        # LLM-first extraction: best quality, so it's the default ('auto') for
        # normal-sized transcripts; 'full' forces it regardless of size
        use_full = (self._detect_mode == 'full'
                    or (self._detect_mode == 'auto'
                        and len(messages) <= MAX_FULL_EXTRACT_MESSAGES))
        did_full = False
        if use_full and self._llm_enabled('auto'):
            # Load the model BEFORE the first timed call: an 8B model's load
            # time alone can blow a per-call timeout and silently downgrade
            # the whole extraction to regex
            self.labeler.warm_up()
            # Extraction is the hardest open-ended job in the pipeline, and
            # seven eval rounds of fast-model wobble (fabricated questions,
            # invented subjects, rhetorical leaks) say so. On a SMALL
            # transcript the token cost is affordable: give the whole job
            # to the quality model
            quality = len(messages) <= int(os.getenv('EXTRACT_QUALITY_MAX', '30'))
            if quality:
                logger.info("Small transcript (%d messages): the quality "
                            "model handles extraction directly", len(messages))
            questions = self._extract_questions_llm(messages, report,
                                                    thorough=quality)
            did_full = True
        else:
            questions = self.extractor.questions_from_messages(messages)
        questions = self._drop_rhetorical_filler(questions)
        questions = self._collapse_same_message_rephrasings(questions)
        questions = self._enforce_single_ask_cap(questions)
        questions = self._enforce_date_integrity(questions)
        questions = self._consolidate_same_ask(questions)

        # Feature requests are product feedback, not support questions a doc
        # page resolves — they leave the support funnel entirely. Diversion
        # is gated on DETERMINISTIC linguistic evidence in the source
        # message, because both models have proven unreliable here in both
        # directions (the 3B tag misses explicitly-labeled items; the 8B
        # confirm has diverted plain capability questions):
        #   - no wish-phrasing in the source -> stays in support, period.
        #     "Can we cap transfers per node?" is a support ask no matter
        #     what any model thinks (a misrouted support question is
        #     invisible to support; feedback noise is merely noise)
        #   - wish-phrasing + an explicit label ("feature request",
        #     "product feedback") -> feedback; the asker classified it
        #   - wish-phrasing alone -> the quality model decides (doubt ->
        #     support). The 3B's tag no longer gates anything: it missed
        #     items sitting in messages literally headed "Feature request:"
        confirm = (self.labeler.confirm_feature_request
                   if self.labeler is not None and self._llm_enabled('auto')
                   else None)
        feature_requests, kept = [], []
        for q in questions:
            source = q.get('original_message') or ''
            wishful = bool(self._WISH_RE.search(source))
            labeled = bool(self._FEEDBACK_LABEL_RE.search(source))
            # The purest build-it signal: a wish PLUS the asker's own
            # statement that the capability doesn't exist — no model
            # opinion needed (the 8B once kept 'would like cron support -
            # doesn't look possible today' in the support funnel)
            says_missing = bool(self._IMPOSSIBLE_RE.search(source))
            if wishful and (labeled or says_missing
                            or (confirm is not None
                                and confirm(q['text'], source) is True)):
                q['qtype'] = 'feature-request'
                feature_requests.append(q)
                continue
            if q.get('qtype') == 'feature-request':
                logger.info("Kept in support (%s): %.80r",
                            'no wish-phrasing in the source message'
                            if not wishful else 'not confirmed', q['text'])
            kept.append(q)
        questions = kept
        if feature_requests:
            logger.info("Routed %d confirmed feature request(s) out of the "
                        "support funnel to product feedback", len(feature_requests))
        logger.info("Found %d questions", len(questions))
        report('extracting', 1, 1)

        # Optional LLM pass: catch implicit help requests the regex missed
        # (already covered when the LLM did the whole extraction)
        if not did_full and self._detect_mode in ('auto', 'on') \
                and self._llm_enabled(self._detect_mode):
            questions += self._detect_missed_questions(messages, questions, report)

        if not questions:
            report('complete', 1, 1)
            return {
                'total_questions': 0,
                'total_groups': 0,
                'groups': [],
                'ungrouped_questions': [],
                'feature_requests': feature_requests,
                'metadata': self._metadata()
            }

        verify_on = self._llm_enabled(self._verify_mode)
        verifier = self.labeler.verify_same_topic if verify_on else None
        auditor = self.labeler.audit_group if verify_on else None

        # First run ever: pre-load the bank with curated starter topics so
        # groups get good names from day one
        self._seed_topic_bank_if_empty()

        # The learned topic bank's categories claim questions by
        # classification before any pairwise clustering
        known_topics = None
        bank = TopicBank(model=self.similarity_analyzer.embedding_model)
        if bank.enabled and bank.entries:
            model = self.similarity_analyzer.embedding_model
            known_topics = [e for e in bank.entries
                            if not e.get('model') or e['model'] == model]

        logger.info("Step 2: Grouping similar questions using AI...")
        self._last_routing = None
        taxonomy = Taxonomy()
        if taxonomy.enabled:
            groups = self._group_with_taxonomy(questions, taxonomy, verifier,
                                               auditor, known_topics, report)
        else:
            groups = self.similarity_analyzer.group_similar_questions(
                questions,
                progress_callback=(lambda done, total: report('embedding', done, total)),
                verifier=verifier,
                auditor=auditor,
                known_topics=known_topics
            )
        # Invariant: a recurrence requires occurrences from DISTINCT source
        # messages. Same-message rephrases that slip past consolidation can
        # cluster into a phantom 'asked 2x'; collapse them deterministically
        # — whatever upstream pass leaked them.
        self._collapse_same_source_occurrences(groups)
        # Exit assertion: a group may only render a count it can PROVE with
        # rows. Violations are repaired (never shipped) and counted loudly.
        groups = self._enforce_render_integrity(groups)
        logger.info("Created %d question groups", len(groups))
        report('grouping', 1, 1)

        # Add keywords and date ranges to each group. Keywords are scored
        # against the whole corpus: a word common to every group ("just",
        # "need") characterizes nothing.
        logger.info("Step 3: Extracting keywords from groups...")
        corpus_df = self._corpus_doc_freq(questions)
        for group in groups:
            group['keywords'] = self._extract_keywords(
                group['questions'], corpus_df, len(questions))
            group['date_range'] = self._date_range(group['questions'])
        report('keywords', 1, 1)

        # Optional LLM pass: name each group and summarize what's being asked
        self._label_groups(groups, report)

        # Optional LLM pass: did thread replies answer the question?
        answered_total = self._detect_answers(questions, groups, report)

        # Separate single-question groups
        multi_question_groups = [g for g in groups if g['count'] > 1]
        single_questions = [g for g in groups if g['count'] == 1]

        result = {
            # Derived from rendered rows, never counted earlier: the exit
            # invariant can collapse occurrences inside groups AFTER the
            # question list was built, and a total the page can't prove
            # with rows is the same lie as a 2x with empty slots
            'total_questions': sum(g['count'] for g in groups),
            'total_groups': len(multi_question_groups),
            'groups': multi_question_groups,
            'ungrouped_questions': [q['questions'][0] for q in single_questions],
            'feature_requests': feature_requests,
            # Provenance: every question any stage removed, with its reason —
            # nothing is ever silently consumed
            'dropped_questions': list(self._dropped),
            # Answered=0 with no replies in the export means "unmeasurable",
            # not "everything went unanswered" — the UI shows the difference
            'threads_present': any(q.get('replies') for q in questions),
            'answered_questions': answered_total,
            'metadata': self._metadata()
        }

        # Funnel stage 2: roll everything up into a handful of broad themes
        result['themes'] = self._assign_themes(multi_question_groups,
                                               result['ungrouped_questions'])

        # Optional LLM pass: 2-3 sentence executive summary
        if self._llm_enabled(self._summary_mode) and multi_question_groups:
            report('summarizing', 0, 1)
            logger.info("Generating executive summary...")
            result['executive_summary'] = self.labeler.summarize_analysis(
                multi_question_groups, len(questions), themes=result.get('themes'))
            report('summarizing', 1, 1)
        else:
            result['executive_summary'] = None

        report('complete', 1, 1)
        return result

    def _record_drop(self, question: Dict, reason: str):
        """Provenance: a removed question becomes an auditable record in the
        results, never a silent mutation."""
        self._dropped.append({'text': question.get('text'),
                              'date': question.get('date'),
                              'source': question.get('original_message'),
                              'reason': reason})

    # Content-free rhetorical filler the extraction prompt already names —
    # built only from pronouns and pleasantries, so there is nothing to
    # answer no matter the context. The models keep leaking these (and the
    # two-judge consolidation once PROTECTED one), so the prompt's own list
    # is enforced in code.
    _RHETORICAL_FILLER_RE = re.compile(
        r'^(?:has |have )?(?:anyone|anybody)(?: else)? (?:seen|hit|run into)'
        r'(?: this| that| it)?(?: before)?$'
        r'|^(?:anyone|anybody) else (?:ready|excited|hyped|looking forward)\b.*$'
        r'|^any (?:thoughts|ideas|luck|advice)(?: on this| on that)?$'
        r'|^(?:right|thoughts|make sense|does that make sense|makes sense)$',
        re.IGNORECASE)

    def _drop_rhetorical_filler(self, questions: List[Dict]) -> List[Dict]:
        kept = []
        for q in questions:
            text = re.sub(r'[?!.\s]+$', '', (q.get('text') or '').strip())
            if self._RHETORICAL_FILLER_RE.match(text):
                logger.info("Dropped content-free rhetorical filler: %.80r",
                            q['text'])
                self._record_drop(q, 'rhetorical filler (content-free)')
                continue
            kept.append(q)
        return kept

    @staticmethod
    def _ask_sentence(source: str) -> str:
        """The sentence ending at the source's FIRST '?' — when a message
        has exactly one question mark, that sentence is the asker's actual
        question, and survivor-ranking should prefer its rewrite."""
        qmark = source.find('?')
        if qmark == -1 or source.count('?') > 1:
            return source
        start = max(source.rfind('.', 0, qmark), source.rfind('!', 0, qmark))
        return source[start + 1:qmark + 1].strip() or source

    # A question that LEADS with a restatement marker is by its own words a
    # rewording of the message's other ask — the marker is deterministic
    # evidence the lexical-overlap test can't see ('I mean is there a
    # built-in way to gzip the payload?' shares no content words with 'Can
    # we compress files before sending?')
    _RESTATE_MARKER_RE = re.compile(
        r'^(?:basically|i mean|in other words|that is|so basically'
        r'|put differently)\b', re.IGNORECASE)

    # 'Is that the right approach, or is there a cleaner pattern?' — a
    # question whose subject is pure deixis ('that'/'this') and whose
    # content words are all META-vocabulary about asking (approach, way,
    # pattern...) cannot stand alone: it is a continuation of the message's
    # other ask, even with zero domain-word overlap
    _DEICTIC_LEAD_RE = re.compile(r'^(?:is|would|does) (?:that|this)\b',
                                  re.IGNORECASE)
    _META_QUESTION_WORDS = frozenset({
        'that', 'this', 'there', 'right', 'correct', 'best', 'better',
        'clean', 'cleaner', 'approach', 'approaches', 'pattern', 'patterns',
        'idea', 'ideas', 'sense', 'option', 'options', 'work', 'possible'})

    def _collapse_same_message_rephrasings(self, questions: List[Dict]) -> List[Dict]:
        """
        Two extractions of the same ask from ONE message are one question,
        not an 'asked 2x' topic. The extractor sometimes rewrites a single
        complaint from two angles ('What is the antivirus scanning error...?'
        / 'Why does the Copy Task fail due to an antivirus scanning error?');
        within one message, a moderate content-word overlap means same ask.
        Distinct multi-questions in a message (REST triggers vs bulk-disable)
        share almost no content words and are kept.

        CONTENT words only (>3 chars, same bar as source support): two
        distinct asks rewritten onto one template ('Can we X in wM MFT
        (SaaS)?' / 'Can we Y in wM MFT (SaaS)?') share plenty of filler and
        boilerplate, and that carries zero same-ask evidence. Anything in
        the gray zone falls through to the LLM consolidation pass, where
        dropping takes two judges.
        """
        threshold = float(os.getenv('SAME_MESSAGE_REPHRASE_OVERLAP', '0.5'))

        def stem(t):
            # Light suffix-folding so 'failed transfers' and 'transfer
            # fails' count as shared content — a real rephrase pair scored
            # ZERO overlap on exact tokens and survived as a fake 2x
            for suffix in ('ing', 'ed', 'es', 's'):
                if t.endswith(suffix) and len(t) - len(suffix) >= 3:
                    return t[:len(t) - len(suffix)]
            return t

        def tokens(q):
            return {stem(t) for t in re.findall(r'[a-z0-9]+',
                                                q['normalized_text'].lower())
                    if len(t) > 3}

        def rank(q, toks):
            # Which phrasing survives a collapse: the one the source message
            # actually vouches for. An extraction that borrowed vocabulary
            # (prompt examples, neighbor messages) loses to the rewrite drawn
            # from the message itself. (Ask-sentence-first ranking was tried
            # and reverted: substring matches let contaminated rewrites
            # outscore genuine ones that import the context's subject.)
            support = self._source_support(q['normalized_text'],
                                           q.get('original_message') or '')
            return (support, len(toks), len(q.get('text') or ''))

        kept: List[Dict] = []
        by_message: Dict[str, List[Dict]] = {}
        for q in questions:
            source = q.get('original_message') or ''
            toks = tokens(q)
            text = (q.get('text') or '').strip()
            if source and self._ENUMERATED_ASKS_RE.search(source):
                # Enumerated-split siblings are locked separate: the asker
                # declared them distinct, and that outranks any overlap or
                # marker heuristic (identical-text dups were already removed
                # at extraction)
                by_message.setdefault(source, [])
                kept.append(q)
                continue
            marked = bool(self._RESTATE_MARKER_RE.match(text)) or (
                bool(self._DEICTIC_LEAD_RE.match(text))
                and toks <= self._META_QUESTION_WORDS)
            match = None
            for seen in by_message.get(source, []):
                # IDENTICAL text is a genuine repeat (someone asked the exact
                # question again) — occurrence counting handles it. Only a
                # DIFFERENT rewrite with heavy overlap is a rephrasing.
                if seen['norm'] == q['normalized_text']:
                    continue
                # Either side leading with a restatement marker IS the
                # overlap evidence — the asker said so themselves
                if marked or seen['marked']:
                    match = seen
                    break
                overlap = len(toks & seen['tokens']) / max(1, min(len(toks), len(seen['tokens'])))
                if overlap >= threshold:
                    match = seen
                    break
            if match is not None:
                if rank(q, toks) > match['rank']:
                    logger.info("Collapsed a same-message rephrasing (kept a "
                                "better-supported one): %.80r", match['q']['text'])
                    self._record_drop(match['q'], 'same-message rephrasing (lexical)')
                    kept[match['pos']] = q
                    match.update(norm=q['normalized_text'], tokens=toks,
                                 q=q, rank=rank(q, toks), marked=marked)
                else:
                    logger.info("Collapsed a same-message rephrasing: %.80r", q['text'])
                    self._record_drop(q, 'same-message rephrasing (lexical)')
                continue
            by_message.setdefault(source, []).append(
                {'norm': q['normalized_text'], 'tokens': toks, 'q': q,
                 'rank': rank(q, toks), 'pos': len(kept), 'marked': marked})
            kept.append(q)
        return kept

    @staticmethod
    def _source_support(question_norm: str, message_text: str) -> float:
        """
        Fraction of the question's content words present in a message.
        Rewrites draw their vocabulary from their source, so a genuine
        extraction scores high; a misattributed one scores near zero.
        """
        tokens = [t for t in re.findall(r'[a-z0-9]+', question_norm.lower())
                  if len(t) > 3]
        if not tokens:
            return 1.0
        msg = message_text.lower()
        msg_tokens = [t for t in re.findall(r'[a-z0-9]+', msg) if len(t) > 3]
        matched = sum(1 for t in tokens
                      if t in msg or any(mt in t for mt in msg_tokens))
        return matched / len(tokens)

    SOURCE_SUPPORT_MIN = 0.35

    # Deterministic feedback-lane gates (see analyze_contents). WISH:
    # capability-wish phrasing — deliberately excludes task phrasing
    # ("customer wants to bulk-deactivate...") and polite support openers
    # ("I would like to know how..."). LABEL: the asker classified the
    # item themselves.
    _WISH_RE = re.compile(
        r"would (really )?(love|like)(?! to (know|understand|ask))"
        r"|would be (great|nice|helpful)|would help"
        r"|\bwish (the|we|it|there)\b|please add|any plans to|love it if"
        r"|wants? to be able to|doesn'?t exist today|isn'?t possible today"
        r"|(doesn'?t|does not) (look|seem|appear)( to be)? possible",
        re.IGNORECASE)
    _IMPOSSIBLE_RE = re.compile(
        r"(doesn'?t|does not) (exist|(look|seem|appear)( to be)? possible)"
        r"( today)?|isn'?t possible today|not (currently )?supported today",
        re.IGNORECASE)
    _FEEDBACK_LABEL_RE = re.compile(
        r'\b(feature requests?|product feedback|enhancement requests?'
        r'|customer feedback)\b', re.IGNORECASE)

    def _verify_source(self, question: Dict, hit: Dict, batch) -> Optional[Dict]:
        """
        Invariant: an extracted question must be textually supported by its
        claimed source message. If not, reassign it to the batch message
        that supports it best; if none does, drop it (and let the safety
        net's regex pass recover whatever the true source actually said).
        """
        min_support = float(os.getenv('EXTRACT_SUPPORT_MIN',
                                      str(self.SOURCE_SUPPORT_MIN)))
        claimed_text = batch[hit['index']][0]
        if self._source_support(question['normalized_text'], claimed_text) >= min_support:
            return question

        best_index, best_support = None, min_support
        for i, (text, _) in enumerate(batch):
            if i == hit['index']:
                continue
            support = self._source_support(question['normalized_text'], text)
            if support > best_support:
                best_index, best_support = i, support
        if best_index is not None:
            text, message = batch[best_index]
            logger.info("Reassigned an extraction to its true source message "
                        "(claimed message doesn't contain it): %.80r",
                        question['text'])
            if self.labeler is not None:
                self.labeler._count('extract_reassigned')
            return self._llm_question(hit['question'], text, message,
                                      hit.get('type'))
        logger.info("Dropped an unsupported extraction (no message in the "
                    "batch contains it): %.80r", question['text'])
        self._record_drop(question, 'unsupported extraction (no source contains it)')
        if self.labeler is not None:
            self.labeler._count('extract_dropped_unsupported')
        return None

    def _enforce_single_ask_cap(self, questions: List[Dict]) -> List[Dict]:
        """
        Invariant: an UNENUMERATED message containing at most one question
        mark asks at most one question. Every genuine multi-ask message
        signals itself — numbered lists, 'and separately', 'two things' —
        or simply contains several '?'s; without any such signal, a second
        extraction is the model rewriting the message's context into an
        extra question ('Is there a max size?' + 'Can it handle very large
        payloads?' from one sentence). Keep the best-supported phrasing,
        drop the rest with provenance.

        Only applies when original_message is untruncated (< its 200-char
        cap): a clipped source can hide '?'s and enumeration markers.
        """
        by_source: Dict[str, List[Dict]] = {}
        for q in questions:
            by_source.setdefault(q.get('original_message') or '', []).append(q)

        drop: set = set()
        for source, qs in by_source.items():
            if not source or len(source) >= 200:
                continue
            # Identical-text entries are genuine repeats (distinct short
            # messages can share their whole text) — occurrence counting
            # owns those. The cap only adjudicates DISTINCT rewrites.
            if len({q['normalized_text'] for q in qs}) < 2:
                continue
            if source.count('?') > 1 or self._ENUMERATED_ASKS_RE.search(source):
                continue

            # The lone '?' marks the asker's ACTUAL question: the survivor
            # must be the rewrite of THAT sentence, not of the surrounding
            # context (both can be verbatim-supported by the whole message,
            # and 'why are transfers timing out' once outranked 'how do I
            # increase the timeout' purely on length)
            ask_sentence = self._ask_sentence(source)

            def rank(q):
                return (self._source_support(q['normalized_text'], ask_sentence),
                        self._source_support(q['normalized_text'], source),
                        len(q.get('text') or ''))

            best_norm = max(qs, key=rank)['normalized_text']
            for q in qs:
                if q['normalized_text'] != best_norm:
                    logger.info("Dropped an extra extraction (unenumerated "
                                "single-'?' message asks one question): %.80r",
                                q['text'])
                    self._record_drop(q, "extra extraction (single-'?' "
                                         'message, no enumerated asks)')
                    drop.add(id(q))
        return [q for q in questions if id(q) not in drop]

    def _consolidate_same_ask(self, questions: List[Dict]) -> List[Dict]:
        """
        Lexical collapse catches near-verbatim restatement; this catches
        REPHRASED restatement: a message whose one ask was extracted as two
        differently-worded questions ("wrong timezone after DST?" / "is the
        DST issue timezone-related?"). For every message that produced 2+
        questions, the quality model picks the distinct asks (closed choice,
        abstain = keep all).

        Dropping a question is destructive, so it follows the two-judge
        rule: the consolidator nominates, and the verifier must confirm the
        dropped question is the SAME ask as a kept one (explicit True).
        Different/uncertain -> the question stays. This is what protects a
        genuine two-part message (IP ranges + maintenance window) from
        losing its second half.
        """
        if self.labeler is None or not self._llm_enabled('auto'):
            return questions
        cap = int(os.getenv('LLM_CONSOLIDATE_MAX', '15'))
        calls = 0

        by_message: Dict[str, List[int]] = {}
        for i, q in enumerate(questions):
            by_message.setdefault(q.get('original_message') or '', []).append(i)

        drop = set()
        for source, indices in by_message.items():
            if len(indices) < 2 or calls >= cap or not source:
                continue
            # PRECEDENCE RULE: a message that EXPLICITLY enumerates separate
            # asks ('1. ... 2. ...', 'and separately', 'two things') had its
            # split decided at extraction, on the asker's own words — the
            # split decision outranks every collapse decision, always.
            # Consolidation once deleted 'max retry count' as a 'rephrasing'
            # of its enumerated sibling 'custom transfer label'.
            if self._ENUMERATED_ASKS_RE.search(source):
                continue
            calls += 1
            keep = self.labeler.consolidate_same_ask(
                source, [questions[i]['text'] for i in indices])
            if keep is None:
                continue
            keep_set = {indices[k - 1] for k in keep}
            kept_texts = [questions[i]['text'] for i in indices if i in keep_set][:3]
            for i in indices:
                if i in keep_set:
                    continue
                verdict = self.labeler.verify_same_topic(
                    [questions[i]['text']], kept_texts)
                if verdict is not True:
                    logger.info("Consolidation overruled by the verifier "
                                "(distinct ask kept): %.80r", questions[i]['text'])
                    if self.labeler is not None:
                        self.labeler._count('consolidation_overruled')
                    continue
                logger.info("Consolidated a same-ask rewrite: %.80r",
                            questions[i]['text'])
                self.labeler._count('same_ask_collapsed')
                self._record_drop(questions[i], 'same-ask consolidation (two judges)')
                drop.add(i)
        if drop:
            return [q for i, q in enumerate(questions) if i not in drop]
        return questions

    def _enforce_date_integrity(self, questions: List[Dict]) -> List[Dict]:
        """
        Invariant: identical question text on two different dates is illegal
        unless that text genuinely appears at both dates. A date-collision
        copy whose own source message doesn't contain the question is a
        backfilled phantom — it gets dropped, never emitted as a fake
        'asked 2x' recurrence.
        """
        min_support = float(os.getenv('EXTRACT_SUPPORT_MIN',
                                      str(self.SOURCE_SUPPORT_MIN)))
        by_text: Dict[str, List[Dict]] = {}
        for q in questions:
            by_text.setdefault(q['normalized_text'], []).append(q)

        dropped = set()
        for norm, copies in by_text.items():
            dates = {q.get('date') for q in copies}
            if len(copies) < 2 or len(dates) < 2:
                continue
            for q in copies:
                support = self._source_support(norm, q.get('original_message') or '')
                if support < min_support:
                    logger.info("Dropped a date-collision phantom (%s copy of "
                                "a question its source doesn't contain): %.80r",
                                q.get('date'), q['text'])
                    self._record_drop(q, 'date-collision phantom')
                    if self.labeler is not None:
                        self.labeler._count('date_collisions_dropped')
                    dropped.add(id(q))
        if dropped:
            return [q for q in questions if id(q) not in dropped]
        return questions

    def _extract_questions_llm(self, messages: List[Dict], report,
                                thorough: bool = False) -> List[Dict]:
        """
        LLM-first extraction (LLM_EXTRACTION=full): every message goes to the
        LLM, which extracts and cleanly rewrites each question. Batches where
        the LLM fails fall back to the regex extractor, so a flaky model never
        loses questions.
        """
        candidates = []
        for message in messages:
            text = ' '.join(self.extractor.clean_slack_markup(message['text']).split())
            if text:
                candidates.append((text, message))

        batches = [candidates[i:i + DETECT_BATCH_SIZE]
                   for i in range(0, len(candidates), DETECT_BATCH_SIZE)]
        logger.info("LLM-first extraction over %d message(s) in %d batch(es)...",
                    len(candidates), len(batches))

        questions = []
        seen_in_message = set()  # (normalized_text, original_message)
        report('detecting', 0, len(batches))
        for batch_num, batch in enumerate(batches, 1):
            hits = self.labeler.extract_questions(
                [text for text, _ in batch], thorough=thorough)
            if hits is not None:  # [] is a real answer: "no questions here"
                for hit in hits:
                    text, message = batch[hit['index']]
                    question = self._llm_question(hit['question'], text,
                                                  message, hit.get('type'))
                    # Field finding (ground-truth audit): the fast model can
                    # attribute a question to the WRONG message in its batch.
                    # The question then inherits the wrong date, the true
                    # source's questions never get extracted, and the
                    # duplicate becomes a phantom "asked 2x". Every extraction
                    # must be textually supported by its claimed source —
                    # otherwise reassign it to the batch message that does
                    # support it, or drop it with a trace.
                    question = self._verify_source(question, hit, batch)
                    if question is None:
                        continue
                    key = (question['normalized_text'], question['original_message'])
                    if key in seen_in_message:
                        logger.info("Dropped a duplicate extraction from one "
                                    "message: %.80r", question['text'])
                        continue
                    seen_in_message.add(key)
                    questions.append(question)
            else:
                # LLM failed for this batch: regex keeps us from losing questions
                questions.extend(self.extractor.questions_from_messages(
                    [message for _, message in batch]))
            report('detecting', batch_num, len(batches))

        # Safety net: a fast model can wrongly return "no questions" for a
        # whole batch — or extract only ONE ask from a genuine two-part
        # message, silently losing the second half. Any message that
        # produced FEWER questions than the regex extractor can see in it
        # gets one second look from the quality model; if that also fails,
        # the regex version is kept — losing real questions is worse than
        # keeping a clumsy one.
        produced_count: Dict[str, int] = {}
        for q in questions:
            key = q.get('original_message') or ''
            produced_count[key] = produced_count.get(key, 0) + 1
        suspicious = []
        for text, message in candidates:
            produced = produced_count.get(text[:200], 0)
            if len(self.extractor.extract_questions(text)) > produced:
                suspicious.append((text, message))
            elif produced == 0 and len(text.split()) >= 8:
                # A wordy message with NO extracted ask is exactly where
                # implicit help requests ('been stuck on this all morning')
                # and relayed wishes ('customer would like X') die silently
                # — regex can't see those, so the count check above never
                # fires for them
                suspicious.append((text, message))
        if suspicious:
            logger.info("Double-checking %d message(s) that produced fewer "
                        "questions than they look like they contain...",
                        len(suspicious))
            # Field finding: the fast model sometimes attributes a question to
            # the WRONG message in its batch, leaving the true source looking
            # skipped — re-extracting it here then duplicated the question and
            # created a phantom "asked 2x" group. Recoveries that match an
            # already-extracted question are dropped.
            seen = {q['normalized_text'] for q in questions}

            def add_unless_duplicate(question):
                if question['normalized_text'] in seen:
                    logger.info("Skipping recovered duplicate: %.80r",
                                question['text'])
                    return
                seen.add(question['normalized_text'])
                questions.append(question)

            for start in range(0, len(suspicious), DETECT_BATCH_SIZE):
                batch = suspicious[start:start + DETECT_BATCH_SIZE]
                hits = self.labeler.extract_questions([t for t, _ in batch],
                                                      thorough=True)
                for hit in hits or []:
                    text, message = batch[hit['index']]
                    question = self._llm_question(hit['question'], text,
                                                  message, hit.get('type'))
                    question = self._verify_source(question, hit, batch)
                    if question is None:
                        continue
                    add_unless_duplicate(question)
                # The regex fallback considers EVERY batch message, not just
                # unrecovered ones: a message recovered with fewer questions
                # than regex sees (one half of a two-part ask extracted, the
                # Kafka half lost) deserves its missing parts back — the
                # duplicate guard and the same-message collapse absorb the
                # parts the LLM already produced.
                for q in self.extractor.questions_from_messages(
                        [m for _, m in batch]):
                    # When the quality model SAW the message and said 'no
                    # questions here' (hits succeeded), only an explicit '?'
                    # can overrule it. Question-shaped statements ('Will
                    # post here when it's back up') fabricated asks from
                    # announcements when restored on shape alone. A FAILED
                    # call (None) still restores everything — losing real
                    # questions stays worse than keeping a clumsy one.
                    if hits is not None and '?' not in q['text']:
                        logger.info("Not restoring a question-shaped "
                                    "statement (two models said no ask, no "
                                    "'?'): %.80r", q['text'])
                        continue
                    add_unless_duplicate(q)

        # Reconciliation: questions must never vanish silently. Every message
        # that produced zero questions is named in the log so a dropped real
        # question leaves a trace.
        produced = {q.get('original_message') for q in questions}
        silent = [text for text, _ in candidates if text[:200] not in produced]
        if silent:
            logger.info("%d of %d message(s) produced no questions:",
                        len(silent), len(candidates))
            for text in silent:
                logger.info("  (no questions) %.90r", text)
            if self.labeler is not None:
                self.labeler._count('messages_without_questions', len(silent))
        return questions

    # A message that EXPLICITLY enumerates separate asks ('1. ... 2. ...',
    # 'two unrelated questions', 'and separately') may legitimately put two
    # rows in one cluster; a message that doesn't enumerate cannot — its
    # same-cluster second row is a rephrase. This is the deterministic line
    # between the T6-class (eject: real distinct asks wrongly merged) and
    # the rephrase class (drop: 'I mean...' variants the judges leaked).
    # Digit enumeration must appear in enumeration POSITION (message start
    # or after a colon/semicolon): a bare \d[.)] matched '(test set 2)' in a
    # transcript header glued to the first message, faking an asker-declared
    # multi-ask split for every question in that message
    _ENUMERATED_ASKS_RE = re.compile(
        r'(?:^|[:;]\s)\s*\d{1,2}[.)]\s|and,? separately'
        r'|two (unrelated )?(questions|things)'
        r'|couple of (questions|things)|second question', re.IGNORECASE)

    def _collapse_same_source_occurrences(self, groups: List[Dict]) -> None:
        """
        Enforce, after ALL grouping passes: within one group, one occurrence
        per source message. Two rephrases of one message's ask are one
        asking — counting them as 'asked 2x' is the recurrence lie this
        pipeline has now produced three different ways; this kills the
        whole class regardless of entry point. Cross-message occurrences
        (genuine repeats) are untouched.

        Disposition of the extra row depends on the SOURCE MESSAGE:
        - it explicitly enumerates separate asks -> EJECT to its own
          singleton row (a distinct ask the clusterer wrongly merged, like
          retention + auto-purge from a 'two things' message);
        - no enumeration -> the same-cluster, same-source second row is a
          rephrase that slipped every upstream pass: DROP it, recorded in
          provenance. Same message + same topic cluster + no claim of
          multiple asks = one asking.
        """
        ejected: List[Dict] = []
        for group in groups:
            first_norm: Dict[str, str] = {}
            kept = []
            for q in group['questions']:
                source = q.get('original_message')
                norm = q.get('normalized_text')
                if source and source in first_norm and norm != first_norm[source]:
                    # A DIFFERENT rewrite from an already-counted source
                    # can't count as a second asking of THIS topic.
                    # (Identical text from an identical source stays
                    # countable — distinct short messages can share text.)
                    if self._ENUMERATED_ASKS_RE.search(source):
                        logger.info("Ejected a same-source occurrence into "
                                    "its own row (the message enumerates "
                                    "separate asks): %.80r", q['text'])
                        single = {'representative_question': q['text'],
                                  'questions': [q], 'count': 1,
                                  'avg_similarity': 1.0}
                        if group.get('bucket'):
                            single['bucket'] = group['bucket']
                        ejected.append(single)
                    else:
                        logger.info("Dropped a same-source rephrase inside "
                                    "a group (message claims no separate "
                                    "asks): %.80r", q['text'])
                        self._record_drop(q, 'same-source rephrase in a '
                                             'group (no enumerated asks)')
                    continue
                if source:
                    first_norm.setdefault(source, norm)
                kept.append(q)
            if len(kept) < len(group['questions']):
                group['questions'] = kept
                group['count'] = len(kept)
        groups.extend(ejected)
    def _enforce_render_integrity(self, groups: List[Dict]) -> List[Dict]:
        """
        'Occurrence' is defined ONCE, here, at the exit: a non-empty kept
        question row. A group's count must equal its rows, and a 2+ count
        must be provable — either 2+ distinct source messages, or identical
        text throughout (distinct short messages can share the same text).
        Any group that can't prove its count is repaired on the spot
        (empty rows stripped, unprovable recurrences demoted to singletons)
        and the repair is counted — a '2x' that can't name its two sources
        can never render again, regardless of which upstream stage misbehaved.
        """
        repaired = 0
        result: List[Dict] = []
        for group in groups:
            rows = [q for q in group['questions'] if (q.get('text') or '').strip()]
            if len(rows) != len(group['questions']):
                repaired += 1
                logger.warning("Integrity repair: stripped %d empty row(s) "
                               "from a group", len(group['questions']) - len(rows))
            if not rows:
                repaired += 1
                continue  # a group with no rows does not exist
            group['questions'] = rows
            group['count'] = len(rows)
            if group['count'] >= 2:
                sources = {q.get('original_message') for q in rows}
                texts = {q.get('normalized_text') for q in rows}
                if len(sources) < 2 and len(texts) > 1:
                    repaired += 1
                    logger.warning("Integrity repair: demoted a %dx group "
                                   "that cannot prove distinct sources: %.70r",
                                   group['count'], rows[0]['text'])
                    for q in rows:
                        result.append({**group, 'questions': [q], 'count': 1,
                                       'representative_question': q['text'],
                                       'avg_similarity': 1.0})
                    continue
            result.append(group)
        if repaired and self.labeler is not None:
            self.labeler._count('integrity_repairs', repaired)
        return result

    def _routing_tiebreak(self, groups: List[Dict],
                          anchor_embeddings) -> List[Dict]:
        """
        Third-judge rule for contested merges. When the audit nominated a
        member for eviction and the verifier overruled it, the judges are
        1-1 — and under GLOBAL grouping that stalemate is what builds
        cross-category false merges. Routing breaks the tie: if the
        contested member and the rest of its group CONFIDENTLY route to
        different buckets (both unambiguous, neither abstaining), the
        member is split out to its own row. Routing is too noisy to wall
        grouping — but a clean two-sided disagreement is real evidence.
        """
        sa = self.similarity_analyzer
        out: List[Dict] = []
        split_out: List[Dict] = []
        for group in groups:
            flagged = [q for q in group['questions']
                       if q.pop('_audit_flagged', False)]
            if not flagged or len(group['questions']) < 2:
                out.append(group)
                continue
            for q in flagged:
                rest = [m for m in group['questions'] if m is not q]
                if not rest:
                    break
                rep = next((m for m in rest
                            if m['text'] == group['representative_question']),
                           rest[0])
                embeds = sa.get_embeddings_batch(
                    [q['normalized_text'], rep['normalized_text']])
                assignments, _, _ = route_questions(embeds, anchor_embeddings)
                if (0 in assignments and 1 in assignments
                        and assignments[0] != assignments[1]):
                    logger.info("Routing broke the judges' tie (audit said "
                                "different, verifier said same, categories "
                                "disagree): %.80r split from %.60r",
                                q['text'], group['representative_question'])
                    group['questions'] = rest
                    group['count'] = len(rest)
                    if group['representative_question'] == q['text']:
                        group['representative_question'] = rest[0]['text']
                    split_out.append({'representative_question': q['text'],
                                      'questions': [q], 'count': 1,
                                      'avg_similarity': 1.0})
            out.append(group)
        return out + split_out

    def _group_with_taxonomy(self, questions: List[Dict], taxonomy: Taxonomy,
                             verifier, auditor, known_topics,
                             report) -> List[Dict]:
        """
        The category funnel — MEANING FIRST, CATEGORY SECOND:

        1. Cluster ALL questions globally by meaning (lexical dedup, bank
           claims, average-link at a fixed bar, borderline verify, pair-only
           rescue, two-judge audit — the full machinery, applied once).
        2. Route each resulting CLUSTER to a bucket by its representative.
           Top-2 anchors too close -> the LLM adjudicates a closed choice.
           Near no anchor, or LLM abstains -> the WHOLE cluster is held for
           review.
        3. Each bucket's fixed 'category' becomes the group's theme — the
           final merge map is deterministic code, not a model.

        Why this order (field finding, fixture 7): grouping used to live
        INSIDE buckets, downstream of routing — so identical asks that
        routed to different buckets could never merge. A 4x recurrence
        caught 3 (the fourth routed elsewhere), a 2x never fired (its halves
        landed in two buckets), and an emerging topic could not surface as a
        cluster because routing scattered it first. Recurrence is a fact
        about MEANING; the bucket is presentation. Bonus: an unroutable
        CLUSTER abstains as a unit — a multi-question review pile is the
        'a category is missing' radar, which per-question routing made
        structurally invisible.
        """
        sa = self.similarity_analyzer
        if sa.threshold_pinned:
            bar = sa.similarity_threshold
        else:
            bar = float(os.getenv('IN_BUCKET_THRESHOLD', '0.8'))

        logger.info("Grouping %d question(s) globally at bar %.2f "
                    "(meaning first, category second)...", len(questions), bar)
        groups = sa.group_similar_questions(
            questions,
            progress_callback=(lambda done, total: report('embedding', done, total)),
            verifier=verifier, auditor=auditor,
            known_topics=known_topics, fixed_threshold=bar)

        # Route each cluster by its representative question
        anchor_embeddings = sa.get_embeddings_batch(taxonomy.anchor_texts())
        # Contested merges first: routing is the third judge on any member
        # the audit flagged and the verifier kept
        groups = self._routing_tiebreak(groups, anchor_embeddings)
        rep_embeddings = sa.get_embeddings_batch(
            [self.extractor.normalize_question(g['representative_question'])
             for g in groups])
        assignments, ambiguous, outliers = route_questions(
            rep_embeddings, anchor_embeddings)

        # LLM adjudication for the genuinely ambiguous routes (closed choice)
        adjudicate = (self.labeler.choose_bucket
                      if self.labeler is not None
                      and self._llm_enabled(self._verify_mode) else None)
        cap = int(os.getenv('ROUTE_LLM_MAX', '20'))
        adjudicated = 0
        for gi, candidate_indices in ambiguous:
            choice = None
            if adjudicate and adjudicated < cap:
                adjudicated += 1
                candidates = [{'id': taxonomy.buckets[k]['id'],
                               'name': taxonomy.bucket_name(k)}
                              for k in candidate_indices]
                chosen_id = adjudicate(groups[gi]['representative_question'],
                                       candidates)
                if chosen_id == 0:
                    # The model abstained (fits both/neither): quarantine —
                    # a pooling review pile beats scattered wrong guesses,
                    # and a growing pile means a category is missing
                    outliers.append(gi)
                    continue
                if chosen_id is not None:
                    choice = next((k for k in candidate_indices
                                   if taxonomy.buckets[k]['id'] == chosen_id), None)
            # Fallback: the embedding favorite (first candidate)
            assignments[gi] = choice if choice is not None else candidate_indices[0]

        review_questions = sum(groups[gi]['count'] for gi in outliers)
        self._last_routing = {
            'taxonomy_version': taxonomy.version,
            'routed': len(assignments),
            'ambiguous': len(ambiguous),
            'llm_adjudicated': adjudicated,
            'needs_review': review_questions,
        }
        logger.info("Routed %d cluster(s) into buckets (%d ambiguous, %d "
                    "AI-adjudicated); %d cluster(s) / %d question(s) kept "
                    "for review (no category fits)",
                    len(assignments), len(ambiguous), adjudicated,
                    len(outliers), review_questions)

        for gi, b in assignments.items():
            group = groups[gi]
            category = taxonomy.final_category(b)
            group['bucket'] = taxonomy.bucket_name(b)
            group['theme'] = category
            for q in group['questions']:
                q['theme'] = category
                # Routing provenance on the row itself: survives the
                # group/ungrouped split, so exports and the eval can
                # check where any individual question landed
                q['bucket'] = taxonomy.bucket_name(b)

        # Unroutable clusters are kept whole and visibly flagged — a funnel
        # that quarantines its own uncertainty beats one that forces every
        # item into the closest wrong home. A MULTI-question review cluster
        # is the emerging-category signal.
        for gi in outliers:
            group = groups[gi]
            for q in group['questions']:
                q['needs_review'] = True
            if group['count'] > 1:
                logger.info("EMERGING TOPIC? A coherent %d-question cluster "
                            "fits no category: %.80r — consider adding a "
                            "bucket for it", group['count'],
                            group['representative_question'])
            else:
                logger.info("Needs review (no category fits): %.80r",
                            group['representative_question'])
        return groups

    def _assign_themes(self, groups: List[Dict],
                       unique_questions: List[Dict]) -> Optional[List[Dict]]:
        """
        Funnel stage 2: one LLM call organizes every topic and unique question
        into 3-6 broad themes. Each group/question gets a 'theme'; returns the
        ordered theme list [{'name', 'count'}] for the dashboard funnel.
        """
        # Taxonomy runs already themed everything via the deterministic merge
        # map — just count, no LLM call
        if any(g.get('theme') for g in groups) or any(q.get('theme') for q in unique_questions):
            counts: Dict[str, int] = {}
            for g in groups:
                if g.get('theme'):
                    counts[g['theme']] = counts.get(g['theme'], 0) + g['count']
            for q in unique_questions:
                if q.get('theme'):
                    counts[q['theme']] = counts.get(q['theme'], 0) + 1
            return [{'name': name, 'count': count}
                    for name, count in sorted(counts.items(), key=lambda x: -x[1])]

        items = ([g.get('topic') or g['representative_question'] for g in groups]
                 + [q['text'] for q in unique_questions])
        if len(items) < 4 or not self._llm_enabled(self._themes_mode):
            return None

        logger.info("Organizing %d topic(s) into broad themes...", len(items))
        assigned = self.labeler.assign_themes(items)
        if not assigned:
            return None

        counts: Dict[str, int] = {}
        for g, theme in zip(groups, assigned):
            if theme:
                g['theme'] = theme
                counts[theme] = counts.get(theme, 0) + g['count']
        for q, theme in zip(unique_questions, assigned[len(groups):]):
            if theme:
                q['theme'] = theme
                counts[theme] = counts.get(theme, 0) + 1
        return [{'name': name, 'count': count}
                for name, count in sorted(counts.items(), key=lambda x: -x[1])]

    def _llm_question(self, raw_question: str, original_text: str,
                      message: Dict, qtype: Optional[str] = None) -> Dict:
        """Build a question dict from an LLM-extracted/rewritten question."""
        cleaned = self.extractor.strip_greeting(raw_question)
        question = {
            'text': cleaned,
            'normalized_text': self.extractor.normalize_question(cleaned),
            'date': message.get('date') or 'Unknown',
            'original_message': original_text[:200],
            'llm_extracted': True,
        }
        if qtype:
            question['qtype'] = qtype
        if message.get('replies'):
            question['replies'] = message['replies']
        return question

    def _detect_missed_questions(self, messages: List[Dict], questions: List[Dict],
                                 report) -> List[Dict]:
        """LLM pass over messages where the regex extractor found nothing."""
        matched = {q['original_message'] for q in questions}
        unmatched = []
        for message in messages:
            text = ' '.join(self.extractor.clean_slack_markup(message['text']).split())
            if len(text) >= MIN_DETECT_MESSAGE_CHARS and text[:200] not in matched:
                unmatched.append({'text': text, 'date': message.get('date'),
                                  'replies': message.get('replies')})
        unmatched = unmatched[:MAX_DETECTED_MESSAGES]
        if not unmatched:
            return []

        logger.info("Checking %d unmatched messages for implicit questions...", len(unmatched))
        batches = [unmatched[i:i + DETECT_BATCH_SIZE]
                   for i in range(0, len(unmatched), DETECT_BATCH_SIZE)]
        found = []
        report('detecting', 0, len(batches))
        for batch_num, batch in enumerate(batches, 1):
            for hit in self.labeler.detect_questions([m['text'] for m in batch]):
                message = batch[hit['index']]
                question = {
                    'text': hit['question'],
                    'normalized_text': self.extractor.normalize_question(hit['question']),
                    'date': message.get('date') or 'Unknown',
                    # Canonical source key (cleaned + collapsed, same as
                    # every other path): the source string IS the message's
                    # identity for rephrase-collapse, ejection, and the
                    # render-integrity invariant — a normalization mismatch
                    # makes one message look like two sources
                    'original_message': ' '.join(self.extractor
                                                 .clean_slack_markup(message['text'])
                                                 .split())[:200],
                    'llm_detected': True,
                }
                if message.get('replies'):
                    question['replies'] = message['replies']
                found.append(question)
            report('detecting', batch_num, len(batches))

        if found:
            logger.info("LLM found %d additional question(s)", len(found))
        return found

    def _detect_answers(self, questions: List[Dict], groups: List[Dict], report) -> int:
        """LLM pass: mark questions whose thread replies actually answered them."""
        if not self._llm_enabled(self._answers_mode):
            return 0
        candidates = [q for q in questions if q.get('replies')][:MAX_ANSWER_CHECKS]
        if not candidates:
            return 0

        logger.info("Checking %d threads for answers...", len(candidates))
        report('answers', 0, len(candidates))
        answered_total = 0
        for i, question in enumerate(candidates, 1):
            verdict = self.labeler.is_answered(
                question['text'], question['replies'],
                context=question.get('original_message') or '')
            if verdict is not None:
                question['answered'] = verdict
                if verdict:
                    answered_total += 1
            report('answers', i, len(candidates))

        # Question dicts are shared with groups, so per-group counts are free
        for group in groups:
            group['answered'] = sum(1 for q in group['questions'] if q.get('answered'))
        return answered_total

    def _metadata(self) -> Dict:
        return {
            'analyzed_at': datetime.now(timezone.utc).isoformat(),
            # The app version that PRODUCED these results — the dashboard
            # warns when it differs from the running backend, ending the
            # "nothing changed" confusion when an old saved analysis loads
            'app_version': _app_version(),
            'similarity_threshold': self.similarity_analyzer.similarity_threshold,
            'model': self.similarity_analyzer.embedding_model,
            'provider': self.similarity_analyzer.provider,
            # Pairwise similarity distribution — similarity scales differ
            # between embedding models, so this is how users tune the threshold
            'similarity_stats': self.similarity_analyzer.last_similarity_stats,
            'threshold_auto_adjusted': self.similarity_analyzer.threshold_auto_adjusted,
            # The bar actually used: threshold raised above corpus noise (p90)
            'effective_threshold': self.similarity_analyzer.effective_threshold,
            'noise_gate': self.similarity_analyzer.noise_gate,
            # Routing health (taxonomy runs): rising 'needs_review' over time
            # means the taxonomy is drifting out of sync with real traffic
            'routing': self._last_routing,
            # Abstain/verdict rates: if the rescue pass makes verify fire
            # constantly, in-bucket clustering is under-forming upstream
            'llm_stats': dict(self.labeler.stats) if self.labeler else None,
            'prompt_pack': PROMPT_PACK_VERSION,
        }

    @staticmethod
    def suggested_threshold(results: Dict) -> Optional[float]:
        """
        When nothing grouped, suggest a threshold just below the most similar
        pair so the next run produces at least one group. None when the
        current threshold already groups things (or there's nothing to group).
        """
        metadata = results.get('metadata', {})
        stats = metadata.get('similarity_stats')
        if not stats or results.get('total_groups', 0) > 0:
            return None
        threshold = metadata.get('effective_threshold') or metadata['similarity_threshold']
        if stats['max'] >= threshold:
            return None
        suggestion = round(stats['max'] - 0.02, 2)
        return suggestion if 0 < suggestion < threshold else None

    @staticmethod
    def _topic_grounded(topic: str, group: Dict) -> bool:
        """Every content word of a topic label (stemmed, >3 chars) must
        occur in the group's question text — labels describe, never invent."""
        def stems(text):
            out = set()
            for t in re.findall(r'[a-z0-9]+', text.lower()):
                if len(t) <= 3:
                    continue
                for suffix in ('ing', 'ed', 'es', 's'):
                    if t.endswith(suffix) and len(t) - len(suffix) >= 3:
                        t = t[:len(t) - len(suffix)]
                        break
                out.add(t)
            return out

        topic_stems = stems(topic)
        if not topic_stems:
            return False
        member_stems = stems(' '.join(q.get('text') or ''
                                      for q in group['questions']))
        # Prefix-tolerant match: the crude suffix folding is asymmetric
        # ('failures' -> 'failur' but 'failure' -> 'failure'), so a stem
        # counts as present when either side is a prefix of the other
        return all(any(t.startswith(m) or m.startswith(t)
                       for m in member_stems)
                   for t in topic_stems)

    def _label_groups(self, groups: List[Dict], report):
        """
        Give each group a 'topic' (and, when an LLM is available, a 'summary').

        The topic bank is consulted first: groups matching a known topic keep
        its established name (stable labels across analyses, no LLM call).
        Remaining multi-question groups get LLM-generated labels when a
        generation model is reachable; everything else falls back to keywords.
        """
        candidates = [g for g in groups if g['count'] > 1][:MAX_LABELED_GROUPS]

        # Pass 1: the bank labels topics it has seen before. Bank matching has
        # its own strict floor: auto-threshold may relax grouping, but a loose
        # match inheriting a curated name would be worse than no name.
        bank = TopicBank(model=self.similarity_analyzer.embedding_model)
        bank_matches = {}  # id(group) -> (centroid, matched entry or None)
        labeled_by = {}    # id(group) -> 'bank' | 'llm' | 'keywords'
        if bank.enabled:
            threshold = max(self.similarity_analyzer.similarity_threshold,
                            float(os.getenv('BANK_MATCH_THRESHOLD', '0.85')))
            for group in candidates:
                centroid = self._group_centroid(group)
                matched = bank.match(centroid, threshold)
                bank_matches[id(group)] = (centroid, matched)
                if matched and matched.get('topic'):
                    group['topic'] = matched['topic']
                    group['summary'] = matched.get('summary')
                    labeled_by[id(group)] = 'bank'
            known = sum(1 for _, m in bank_matches.values() if m)
            if known:
                logger.info("Topic bank recognized %d of %d group(s)",
                            known, len(candidates))

        # Pass 2: LLM labels for topics the bank didn't know
        unlabeled = [g for g in candidates if not g.get('topic')]
        use_llm = self.labeler is not None and (self._labels_forced or self.labeler.available())
        if use_llm and unlabeled:
            logger.info("Step 4: Generating topic labels with %s...", self.labeler.model)
            report('labeling', 0, len(unlabeled))
            for i, group in enumerate(unlabeled, 1):
                sample = self._diverse_sample(group['questions'])
                label = self.labeler.label_group([q['text'] for q in sample],
                                                 keywords=group.get('keywords'))
                # Grounding invariant: every content word of the label must
                # appear in the group's own question text. A group of
                # failure-alert questions was once labeled 'Transfer
                # Retries' — a name the members never said. Ungrounded
                # labels fall back to keywords (which are extracted from
                # the members by construction).
                if label and not self._topic_grounded(label['topic'], group):
                    logger.info("Discarded an ungrounded label (words not in "
                                "the group's own questions): %r", label['topic'])
                    label = None
                if label:
                    group['topic'] = label['topic']
                    group['summary'] = label['summary']
                    labeled_by[id(group)] = 'llm'
                report('labeling', i, len(unlabeled))

        # Keyword fallback for anything the LLM didn't (or couldn't) label
        for group in groups:
            if not group.get('topic'):
                group['topic'] = self._keyword_topic(group)
                group['summary'] = None

        # Pass 3: teach the bank — but only quality names. Keyword-fallback
        # topics are never banked: a junk name that sticks is worse than
        # relabeling next time.
        if bank.enabled:
            for group in candidates:
                centroid, matched = bank_matches.get(id(group), (None, None))
                if matched is None and labeled_by.get(id(group)) != 'llm':
                    continue
                entry = bank.record(group, centroid, matched)
                if entry:
                    group['seen_in_analyses'] = entry['analysis_count']
                    group['topic_id'] = entry['id']  # enables renaming in the UI
            bank.save()

    def _seed_topic_bank_if_empty(self):
        """
        Pre-load an empty topic bank from seed_topics.json (curated
        {topic, question} pairs). Embeddings are computed locally on first
        use and cached; failures are non-fatal (the bank just starts empty).
        """
        bank = TopicBank(model=self.similarity_analyzer.embedding_model)
        if not bank.enabled or bank.entries:
            return
        seed_path = Path(os.getenv('SEED_TOPICS_PATH', 'seed_topics.json'))
        if not seed_path.is_file():
            return

        try:
            with open(seed_path, 'r', encoding='utf-8') as f:
                seeds = json.load(f)
            texts = [self.extractor.normalize_question(s['question']) for s in seeds]
            logger.info("Seeding topic bank with %d starter topics "
                        "(embedding them now; one time only)...", len(seeds))
            embeddings = self.similarity_analyzer.get_embeddings_batch(texts)
            for seed, vector in zip(seeds, embeddings, strict=True):
                v = np.asarray(vector, dtype=float)
                norm = np.linalg.norm(v)
                if not norm:
                    continue
                entry = bank.record({'topic': seed['topic'],
                                     'summary': seed.get('summary'),
                                     'representative_question': seed['question'],
                                     'keywords': seed.get('keywords', []),
                                     'count': 0},
                                    (v / norm).tolist())
                if entry:
                    entry['analysis_count'] = 0  # seeds aren't sightings yet
            bank.save()
        except Exception as e:
            logger.warning("Topic bank seeding skipped: %s", e)

    def _group_centroid(self, group: Dict) -> Optional[List[float]]:
        """Mean unit vector of the group's distinct questions (from cache)."""
        cache = self.similarity_analyzer.embeddings_cache
        prefix = self.similarity_analyzer.embed_prefix
        vectors = []
        seen = set()
        for q in group['questions']:
            text = q['normalized_text']
            if text in seen:
                continue
            seen.add(text)
            vector = cache.get(prefix + text)
            if vector is not None:
                v = np.asarray(vector, dtype=float)
                norm = np.linalg.norm(v)
                if norm:
                    vectors.append(v / norm)
        if not vectors:
            return None
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        return (centroid / norm).tolist() if norm else None

    def _diverse_sample(self, questions: List[Dict], k: int = 8) -> List[Dict]:
        """
        Pick up to k questions that span the group's breadth, so the labeling
        prompt sees different phrasings instead of the same one repeated.
        Uses cached embeddings via greedy farthest-point selection; falls back
        to the first k distinct phrasings when embeddings aren't cached.
        """
        # One entry per distinct phrasing, preserving order
        seen = set()
        distinct = []
        for q in questions:
            if q['normalized_text'] not in seen:
                seen.add(q['normalized_text'])
                distinct.append(q)
        if len(distinct) <= k:
            return distinct

        cache = self.similarity_analyzer.embeddings_cache
        prefix = self.similarity_analyzer.embed_prefix
        vectors = [cache.get(prefix + q['normalized_text']) for q in distinct]
        if any(v is None for v in vectors):
            return distinct[:k]

        matrix = np.array(vectors)
        chosen = [0]  # the first (often most common) phrasing seeds the sample
        while len(chosen) < k:
            chosen_matrix = matrix[chosen]
            # For each candidate, distance to its nearest already-chosen point
            distances = np.min(
                np.linalg.norm(matrix[:, None, :] - chosen_matrix[None, :, :], axis=2),
                axis=1)
            distances[chosen] = -1
            chosen.append(int(np.argmax(distances)))
        return [distinct[i] for i in sorted(chosen)]

    @staticmethod
    def _keyword_topic(group: Dict) -> str:
        keywords = group.get('keywords') or []
        if keywords:
            return ' / '.join(k.capitalize() for k in keywords[:2])
        words = group['representative_question'].split()[:4]
        return ' '.join(words)

    @staticmethod
    def _date_range(questions: List[Dict]) -> Dict:
        """First and last date a question in this group was asked."""
        dates = sorted(q['date'] for q in questions if q.get('date') and q['date'] != 'Unknown')
        return {
            'first_asked': dates[0] if dates else None,
            'last_asked': dates[-1] if dates else None
        }

    def save_results(self, results: Dict, output_path: str):
        """Save results in the format implied by the file extension."""
        lower = output_path.lower()
        if lower.endswith('.csv'):
            self.export_csv(results, output_path)
        elif lower.endswith('.md') or lower.endswith('.markdown'):
            self.export_markdown(results, output_path)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    def export_csv(self, results: Dict, output_path: str):
        """Export grouped questions as a flat CSV (one row per question)."""
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            f.write(to_csv(results))

    def export_markdown(self, results: Dict, output_path: str):
        """Export results as a readable Markdown report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(to_markdown(results))

    # Words that characterize nothing in a support channel
    KEYWORD_STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'there', 'here',
        'just', 'need', 'needs', 'needed', 'want', 'wants', 'wanted', 'like',
        'also', 'some', 'any', 'anyone', 'anybody', 'someone', 'something',
        'anything', 'thoughts', 'idea', 'ideas', 'way', 'ways', 'good', 'best',
        'really', 'please', 'thanks', 'know', 'knows', 'one', 'use', 'using',
        'used', 'get', 'gets', 'getting', 'make', 'makes', 'possible', 'able',
        'support', 'supports', 'supported', 'work', 'works', 'working',
        'question', 'questions', 'help', 'team', 'guys', 'mean', 'means',
        'instead', 'either', 'other', 'following', 'right', 'currently',
    }

    @classmethod
    def _question_words(cls, question: Dict) -> set:
        """Distinct meaningful words in one question."""
        words = set()
        for word in question['normalized_text'].lower().split():
            word = ''.join(c for c in word if c.isalnum())
            if len(word) > 3 and word not in cls.KEYWORD_STOP_WORDS:
                words.add(word)
        return words

    def _corpus_doc_freq(self, questions: List[Dict]) -> Dict[str, int]:
        """How many questions in the whole corpus contain each word."""
        df: Dict[str, int] = {}
        for q in questions:
            for word in self._question_words(q):
                df[word] = df.get(word, 0) + 1
        return df

    def _extract_keywords(self, questions: List[Dict],
                          corpus_df: Optional[Dict[str, int]] = None,
                          n_corpus: int = 0) -> List[str]:
        """
        Keywords that characterize THIS group: frequent within the group,
        rare across the REST of the corpus (the group's own questions must
        not penalize its defining word).
        """
        word_freq: Dict[str, int] = {}
        for q in questions:
            for word in self._question_words(q):
                word_freq[word] = word_freq.get(word, 0) + 1

        n_outside = max(0, n_corpus - len(questions)) if corpus_df else 0

        def score(item):
            word, freq = item
            if corpus_df and n_outside:
                df_outside = max(0, corpus_df.get(word, 0) - freq)
                idf = math.log((n_outside + 1) / (df_outside + 1))
                return (freq * idf, freq)
            return (float(freq), freq)

        ranked = sorted(word_freq.items(), key=score, reverse=True)
        return [word for word, _ in ranked[:5]]

    def print_summary(self, results: Dict):
        """
        Print a human-readable summary of the analysis.

        Args:
            results: Analysis results dictionary
        """
        print("\n" + "="*80)
        print("QUESTION ANALYSIS SUMMARY")
        print("="*80)
        print(f"\nTotal Questions Analyzed: {results['total_questions']}")
        print(f"Question Groups Found: {results['total_groups']}")
        print(f"Ungrouped Questions: {len(results['ungrouped_questions'])}")
        if results.get('answered_questions'):
            print(f"Answered (via thread replies): {results['answered_questions']}")
        print(f"Similarity Threshold: {results['metadata']['similarity_threshold']}")
        print(f"Model Used: {results['metadata']['model']}")

        suggestion = self.suggested_threshold(results)
        if suggestion is not None:
            stats = results['metadata']['similarity_stats']
            print(f"\nTIP: No questions grouped at threshold "
                  f"{results['metadata']['similarity_threshold']}. Your most "
                  f"similar pair scored {stats['max']} — similarity scales vary "
                  f"by embedding model. Try: --threshold {suggestion}")

        if results.get('executive_summary'):
            print("\n" + "-"*80)
            print("EXECUTIVE SUMMARY")
            print("-"*80)
            print(results['executive_summary'])

        if results['groups']:
            print("\n" + "-"*80)
            print("TOP QUESTION GROUPS (Ranked by Frequency)")
            print("-"*80)

            for i, group in enumerate(results['groups'][:10], 1):  # Show top 10
                topic = f" [{group['topic']}]" if group.get('topic') else ''
                print(f"\n#{i}{topic} - Occurrences: {group['count']}")
                print(f"Representative Question: {group['representative_question']}")
                if group.get('summary'):
                    print(f"Summary: {group['summary']}")
                print(f"Keywords: {', '.join(group['keywords'])}")
                print(f"Average Similarity: {group['avg_similarity']:.2%}")
                if group.get('answered'):
                    print(f"Answered: {group['answered']} of {group['count']}")

                if group['count'] <= 5:  # Show all questions if 5 or fewer
                    print("All questions in this group:")
                    for q in group['questions']:
                        print(f"  - {q['text'][:100]}")

        if results['ungrouped_questions']:
            print("\n" + "-"*80)
            print(f"UNIQUE QUESTIONS ({len(results['ungrouped_questions'])})")
            print("-"*80)
            for q in results['ungrouped_questions'][:5]:  # Show first 5
                print(f"  - {q['text'][:100]}")

            if len(results['ungrouped_questions']) > 5:
                print(f"  ... and {len(results['ungrouped_questions']) - 5} more")

        print("\n" + "="*80)
