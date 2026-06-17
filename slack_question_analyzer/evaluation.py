"""
Regression evaluation against labeled fixtures.

Two fixture types, both run by `slack-analyzer eval`:
- question-level: frozen questions with their CORRECT buckets/groupings;
  scores routing accuracy and grouping precision/recall
- transcript-level ("type": "transcript"): a raw transcript plus an answer
  key; runs the FULL pipeline (extraction included) and asserts headline
  numbers — counts, recurrence sizes, feedback membership, abstentions,
  noise rejection, occurrence integrity

Re-running after every prompt, anchor, or threshold change turns "seems
better" into "measurably better" — without it, every tuning change is a
guess that may fix this run while silently breaking the last.

The topic bank is excluded on purpose: bank state differs per machine, and
the fixtures measure the pipeline, not learned history.
"""

import json
import os
import re
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, FrozenSet

from .taxonomy import Taxonomy


def load_fixture(path: str) -> Dict:
    """Load and validate a fixture file (question-level or transcript-level)."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if data.get('type') == 'transcript':
        if not data.get('transcript') or not isinstance(data.get('expect'), dict):
            raise ValueError('Transcript fixture needs "transcript" (file) and "expect" (dict)')
        data['_dir'] = str(Path(path).parent)
        return data
    questions = data.get('questions')
    if not isinstance(questions, list) or not questions:
        raise ValueError('Fixture must contain a non-empty "questions" list')
    for q in questions:
        if not q.get('text') or not q.get('bucket'):
            raise ValueError(f"Every fixture question needs 'text' and 'bucket': {q}")
    return data


def _same_group_pairs(assignment: Dict[str, Optional[object]]) -> Set[FrozenSet]:
    """All unordered pairs of texts sharing a non-null group key."""
    items = [(text, key) for text, key in assignment.items() if key is not None]
    return {frozenset((a, b)) for (a, ka), (b, kb)
            in combinations(items, 2) if ka == kb}


def evaluate(analyzer, fixture: Dict) -> Dict:
    """
    Route + group the fixture questions through the real pipeline and score
    against the labels. Requires a taxonomy and a reachable embedding
    provider; uses the LLM verifier/auditor when available.
    """
    taxonomy = Taxonomy()
    if not taxonomy.enabled:
        raise ValueError('Evaluation needs taxonomy.json (routing taxonomy)')

    questions = [{
        'text': q['text'],
        'normalized_text': analyzer.extractor.normalize_question(q['text']),
        'date': q.get('date', 'Unknown'),
        'original_message': q['text'][:200],
    } for q in fixture['questions']]

    verify_on = analyzer._llm_enabled(analyzer._verify_mode)
    verifier = analyzer.labeler.verify_same_topic if verify_on else None
    auditor = analyzer.labeler.audit_group if verify_on else None

    groups = analyzer._group_with_taxonomy(
        questions, taxonomy, verifier, auditor,
        known_topics=None, report=lambda *args: None)

    predicted_bucket: Dict[str, Optional[str]] = {}
    predicted_group: Dict[str, Optional[int]] = {}
    for gi, group in enumerate(groups):
        for q in group['questions']:
            predicted_bucket[q['text']] = ('review' if q.get('needs_review')
                                           else group.get('bucket'))
            predicted_group[q['text']] = gi if group['count'] > 1 else None

    routing_mismatches: List[Dict] = []
    correct = 0
    for q in fixture['questions']:
        got = predicted_bucket.get(q['text'])
        if got == q['bucket']:
            correct += 1
        else:
            routing_mismatches.append({'text': q['text'],
                                       'expected': q['bucket'], 'got': got})

    # Render-integrity assertions: independent of labels, ANY fixture run
    # fails on a group that can't prove its count
    integrity_violations: List[str] = []
    for group in groups:
        rows = [q for q in group['questions'] if (q.get('text') or '').strip()]
        if len(rows) != group['count']:
            integrity_violations.append(
                f"count {group['count']} != {len(rows)} non-empty rows: "
                f"{group['representative_question'][:60]}")
        if group['count'] >= 2:
            sources = {q.get('original_message') for q in rows}
            texts = {q.get('normalized_text') for q in rows}
            if len(sources) < 2 and len(texts) > 1:
                integrity_violations.append(
                    f"{group['count']}x group without distinct sources: "
                    f"{group['representative_question'][:60]}")

    expected_pairs = _same_group_pairs(
        {q['text']: q.get('group') for q in fixture['questions']})
    got_pairs = _same_group_pairs(predicted_group)
    true_pairs = expected_pairs & got_pairs

    return {
        'questions': len(fixture['questions']),
        'routing_correct': correct,
        'routing_accuracy': correct / len(fixture['questions']),
        'routing_mismatches': routing_mismatches,
        'pairs_expected': len(expected_pairs),
        'pairs_found': len(got_pairs),
        'pairs_correct': len(true_pairs),
        'pair_precision': (len(true_pairs) / len(got_pairs)) if got_pairs else 1.0,
        'pair_recall': (len(true_pairs) / len(expected_pairs)) if expected_pairs else 1.0,
        'missed_pairs': [sorted(p) for p in sorted(expected_pairs - got_pairs,
                                                   key=sorted)],
        'wrong_pairs': [sorted(p) for p in sorted(got_pairs - expected_pairs,
                                                  key=sorted)],
        'integrity_violations': integrity_violations,
        'taxonomy_version': taxonomy.version,
        'fixture': str(Path(getattr(fixture, 'path', '') or '')),
    }


def format_report(result: Dict) -> str:
    """Human-readable evaluation report for the console."""
    lines = [
        f"Fixture: {result['questions']} questions · taxonomy v{result['taxonomy_version']}",
        f"Routing:  {result['routing_correct']}/{result['questions']} correct "
        f"({result['routing_accuracy']:.0%})",
        f"Grouping: {result['pairs_correct']}/{result['pairs_expected']} expected pairs found "
        f"(precision {result['pair_precision']:.0%}, recall {result['pair_recall']:.0%})",
    ]
    if result['routing_mismatches']:
        lines.append('\nRouting mismatches:')
        for m in result['routing_mismatches']:
            lines.append(f"  expected [{m['expected']}] got [{m['got']}]: {m['text'][:90]}")
    if result['missed_pairs']:
        lines.append('\nMissed pairs (should group, did not):')
        for a, b in result['missed_pairs']:
            lines.append(f"  - {a[:70]}\n    + {b[:70]}")
    if result['wrong_pairs']:
        lines.append('\nWrong pairs (grouped, should not):')
        for a, b in result['wrong_pairs']:
            lines.append(f"  - {a[:70]}\n    + {b[:70]}")
    if result.get('integrity_violations'):
        lines.append('\nINTEGRITY VIOLATIONS (a count that cannot prove its rows):')
        for v in result['integrity_violations']:
            lines.append(f"  ! {v}")
    if not (result['routing_mismatches'] or result['missed_pairs']
            or result['wrong_pairs'] or result.get('integrity_violations')):
        lines.append('\nPerfect score.')
    return '\n'.join(lines)


def evaluate_transcript(analyzer, fixture: Dict) -> Dict:
    """
    End-to-end evaluation: run the FULL pipeline (extraction included) on a
    raw transcript and assert the answer key's headline numbers. The
    question-level fixture can't see extraction bugs — silent drops,
    over-splitting, verb drift, feedback misrouting all happen before
    routing — so this fixture type starts from the transcript itself.

    The topic bank is pointed at an empty temp file for the run: learned
    state differs per machine, and a fixture must measure the pipeline,
    not one machine's history.
    """
    path = Path(fixture.get('_dir', '.')) / fixture['transcript']
    content = path.read_text(encoding='utf-8')
    expect = fixture['expect']

    saved = {k: os.environ.get(k) for k in ('TOPIC_BANK_PATH', 'SEED_TOPICS_PATH')}
    with tempfile.TemporaryDirectory() as td:
        os.environ['TOPIC_BANK_PATH'] = str(Path(td) / 'bank.json')
        os.environ['SEED_TOPICS_PATH'] = str(Path(td) / 'no_seeds.json')
        try:
            results = analyzer.analyze_contents([content])
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    groups = results.get('groups', [])  # recurring (count >= 2) only
    support = ([q for g in groups for q in g['questions']]
               + list(results.get('ungrouped_questions', [])))
    feedback = list(results.get('feature_requests', []))
    all_rows = support + feedback

    checks: List[Dict] = []

    def check(name: str, ok: bool, detail: str = ''):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})

    def texts(rows) -> List[str]:
        return [(r.get('text') or '') for r in rows]

    def any_match(pattern: str, rows) -> bool:
        rx = re.compile(pattern, re.IGNORECASE)
        return any(rx.search(t) for t in texts(rows))

    if 'total_asks' in expect:
        got = results.get('total_questions', 0) + len(feedback)
        check(f"total asks = {expect['total_asks']}", got == expect['total_asks'],
              f"got {got} ({results.get('total_questions', 0)} support + "
              f"{len(feedback)} feedback)")

    if 'recurring_topics' in expect:
        check(f"recurring topics = {expect['recurring_topics']}",
              len(groups) == expect['recurring_topics'],
              'got ' + (', '.join(f"{g['count']}x {g['representative_question'][:60]}"
                                  for g in groups) or 'none'))

    # The expected recurring group: ONE group must satisfy every pattern
    # (each matching at least one member question)
    if expect.get('recurring_must_match'):
        patterns = [re.compile(p, re.IGNORECASE)
                    for p in expect['recurring_must_match']]
        hit = any(all(any(rx.search(t) for t in texts(g['questions']))
                      for rx in patterns) for g in groups)
        check('expected recurrence fired '
              f"({' & '.join(expect['recurring_must_match'])})", hit,
              'recurring groups: ' + (', '.join(g['representative_question'][:60]
                                                for g in groups) or 'none'))

    # Named recurrences with exact occurrence counts: each spec must find a
    # recurring group matching every pattern AND showing exactly that count
    for spec in expect.get('recurring_groups', []):
        patterns = [re.compile(p, re.IGNORECASE) for p in spec['must_match']]
        matching = [g for g in groups
                    if all(any(rx.search(t) for t in texts(g['questions']))
                           for rx in patterns)]
        ok = any(g['count'] == spec['count'] for g in matching)
        check(f"recurrence /{' & '.join(spec['must_match'])}/ = "
              f"{spec['count']}x", ok,
              'matching groups: ' + (', '.join(
                  f"{g['count']}x {g['representative_question'][:50]}"
                  for g in matching) or 'none'))

    # Genuine singletons the rescue pass must leave alone: the question must
    # survive (matches a support row) but sit in NO recurring group
    for pattern in expect.get('must_stay_singleton', []):
        rx = re.compile(pattern, re.IGNORECASE)
        survived = any_match(pattern, support)
        grouped = [g for g in groups
                   if any(rx.search(t) for t in texts(g['questions']))]
        check(f'/{pattern}/ stays a singleton', survived and not grouped,
              ('absorbed into: ' + '; '.join(g['representative_question'][:60]
                                             for g in grouped))
              if grouped else 'no surviving support question matches')

    # Answered metric (threads): count plus per-question membership via the
    # 'answered' flag answer detection writes onto each question
    if 'answered_count' in expect:
        got = results.get('answered_questions', 0)
        check(f"answered = {expect['answered_count']}",
              got == expect['answered_count'], f'got {got}')
    answered_rows = [r for r in support if r.get('answered') is True]
    for pattern in expect.get('answered_must_match', []):
        check(f'answered includes /{pattern}/', any_match(pattern, answered_rows),
              'answered: ' + (', '.join(t[:60] for t in texts(answered_rows))
                              or 'none'))
    for pattern in expect.get('answered_must_not_match', []):
        bad = [t for t in texts(answered_rows)
               if re.search(pattern, t, re.IGNORECASE)]
        check(f'answered excludes /{pattern}/', not bad,
              '; '.join(t[:70] for t in bad))

    if 'feedback_count' in expect:
        check(f"product feedback = {expect['feedback_count']}",
              len(feedback) == expect['feedback_count'],
              'got: ' + (', '.join(t[:60] for t in texts(feedback)) or 'none'))
    for pattern in expect.get('feedback_must_match', []):
        check(f'feedback contains /{pattern}/', any_match(pattern, feedback),
              'feedback: ' + (', '.join(t[:60] for t in texts(feedback)) or 'none'))
    for pattern in expect.get('feedback_must_not_match', []):
        bad = [t for t in texts(feedback) if re.search(pattern, t, re.IGNORECASE)]
        check(f'feedback free of /{pattern}/ (support question misrouted '
              'into feedback)', not bad, '; '.join(t[:70] for t in bad))

    # Per-message ask counts: the calibration core. 'contains' must appear in
    # the first 200 chars of the source message (original_message is capped)
    for spec in expect.get('message_asks', []):
        marker = spec['contains'].lower()
        rows = [r for r in all_rows
                if marker in (r.get('original_message') or '').lower()]
        check(f"message '{spec['contains']}' -> {spec['asks']} ask(s)",
              len(rows) == spec['asks'],
              f"got {len(rows)}: " + ('; '.join(
                  f"{(r.get('text') or '')[:55]} "
                  f"[src: {(r.get('original_message') or '')[:40]}]"
                  for r in rows) or 'NONE — silently dropped?'))

    for pattern in expect.get('support_must_match', []):
        check(f'support contains /{pattern}/', any_match(pattern, support),
              'no support question matches — dropped or misrouted')
    for pattern in expect.get('must_not_match', []):
        bad = [t for t in texts(all_rows) if re.search(pattern, t, re.IGNORECASE)]
        check(f'no ask matches /{pattern}/ (verb drift)', not bad,
              '; '.join(t[:70] for t in bad))

    # Routing humility: these must sit in the review pile, not be
    # force-fitted into the closest wrong bucket
    review_rows = [r for r in support if r.get('needs_review')]
    for pattern in expect.get('review_must_match', []):
        rx = re.compile(pattern, re.IGNORECASE)
        in_review = any(rx.search(t) for t in texts(review_rows))
        placed = [r for r in support
                  if rx.search(r.get('text') or '') and not r.get('needs_review')]
        check(f'/{pattern}/ held for review, not force-bucketed', in_review,
              '; '.join(f"routed to [{r.get('bucket')}]: {(r.get('text') or '')[:55]}"
                        for r in placed) or 'no surviving question matches')
    # The inverse control: a clear question must route confidently even
    # when it looks hard (error strings), never over-abstain into review
    for spec in expect.get('routed_must_match', []):
        rx = re.compile(spec['match'], re.IGNORECASE)
        rows = [r for r in support if rx.search(r.get('text') or '')]
        ok = any(not r.get('needs_review')
                 and re.search(spec['bucket'], r.get('bucket') or '',
                               re.IGNORECASE) for r in rows)
        check(f"/{spec['match']}/ routed to /{spec['bucket']}/", ok,
              '; '.join(('review pile' if r.get('needs_review')
                         else f"[{r.get('bucket')}]") + f": {(r.get('text') or '')[:55]}"
                        for r in rows) or 'no surviving question matches')

    # False-merge twins: no group may contain a member matching A and
    # another member matching B
    for a, b in expect.get('must_not_group', []):
        rx_a, rx_b = re.compile(a, re.IGNORECASE), re.compile(b, re.IGNORECASE)
        merged = [g for g in groups
                  if any(rx_a.search(t) for t in texts(g['questions']))
                  and any(rx_b.search(t) for t in texts(g['questions']))]
        check(f'/{a}/ and /{b}/ not merged', not merged,
              '; '.join(g['representative_question'][:70] for g in merged))

    # Occurrence integrity, asserted on EVERY transcript fixture: a group's
    # count must equal its populated rows, and a 2+ count needs rows from
    # distinct source messages (identical forwarded text exempt)
    bad_groups = []
    for g in groups:
        rows = [q for q in g['questions'] if (q.get('text') or '').strip()]
        sources = {q.get('original_message') for q in rows}
        distinct_texts = {q.get('normalized_text') or q.get('text') for q in rows}
        if len(rows) != g['count'] or (len(sources) < 2
                                       and len(distinct_texts) > 1):
            bad_groups.append(f"{g['count']}x {g['representative_question'][:50]}"
                              f" ({len(rows)} rows, {len(sources)} sources)")
    check('every count provable (rows populated, sources distinct)',
          not bad_groups, '; '.join(bad_groups))

    # A repair firing means a count-without-rows leaked to the exit
    # invariant — defensive code saved the render, but the leak is a bug
    repairs = (results.get('metadata', {}).get('llm_stats', {})
               or {}).get('integrity_repairs', 0)
    check('no render-integrity repairs needed', not repairs,
          f'{repairs} repair(s) — an upstream stage leaked an unprovable count')

    failed = [c for c in checks if not c['ok']]
    return {'type': 'transcript', 'checks': checks, 'failed': len(failed),
            'passed': len(checks) - len(failed), 'results': results}


def format_transcript_report(result: Dict) -> str:
    """Human-readable transcript-fixture report for the console."""
    lines = []
    for c in result['checks']:
        mark = 'PASS' if c['ok'] else 'FAIL'
        lines.append(f"  [{mark}] {c['name']}")
        if not c['ok'] and c['detail']:
            lines.append(f"         {c['detail']}")
    lines.append(f"\n{result['passed']}/{len(result['checks'])} checks passed"
                 + ('' if not result['failed'] else
                    f" — {result['failed']} FAILED"))
    # On failure, the provenance trail turns a mystery into data: every
    # removed question with its reason and source
    dropped = (result.get('results') or {}).get('dropped_questions') or []
    if result['failed'] and dropped:
        lines.append('\nProvenance (removed during analysis):')
        for d in dropped:
            lines.append(f"  - {(d.get('text') or '')[:60]} — "
                         f"{d.get('reason', '')} "
                         f"[src: {(d.get('source') or '')[:40]}]")
    return '\n'.join(lines)
