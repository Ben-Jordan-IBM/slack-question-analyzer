"""
Export analysis results as CSV or Markdown strings.

Used by both the CLI (writing files) and the API server (download endpoint).
"""

import csv
import io
from typing import Dict, Optional

from .textutil import rank_replies
from .weekly_stats import parse_question_date


def _defuse(value):
    """Neutralize spreadsheet formula injection: transcript text is
    attacker-controlled, and Excel/Sheets execute cells starting with
    = + - @ (or a control char) as formulas when the CSV is opened."""
    if isinstance(value, str) and value and value[0] in ('=', '+', '-', '@', '\t', '\r'):
        return "'" + value
    return value


def to_csv(results: Dict) -> str:
    """Flat CSV: one row per question (grouped, unique, and feedback)."""
    buffer = io.StringIO()
    raw_writer = csv.writer(buffer)

    class _SafeWriter:
        @staticmethod
        def writerow(row):
            raw_writer.writerow([_defuse(cell) for cell in row])

    writer = _SafeWriter()
    writer.writerow(['group_rank', 'group_count', 'representative_question',
                     'keywords', 'avg_similarity', 'question', 'date',
                     'kind', 'theme', 'type', 'answered'])

    def answered_cell(q):
        if q.get('answered') is True:
            return 'yes'
        if q.get('answered') is False:
            return 'no'
        return ''

    for rank, group in enumerate(results['groups'], 1):
        for q in group['questions']:
            writer.writerow([
                rank, group['count'], group['representative_question'],
                '; '.join(group['keywords']), f"{group['avg_similarity']:.4f}",
                q['text'], q.get('date', 'Unknown'),
                'grouped', group.get('theme', ''), q.get('qtype', ''),
                answered_cell(q),
            ])
    for q in results.get('ungrouped_questions', []):
        writer.writerow(['', 1, q['text'], '', '', q['text'],
                         q.get('date', 'Unknown'),
                         'needs review' if q.get('needs_review') else 'unique',
                         q.get('theme', ''), q.get('qtype', ''),
                         answered_cell(q)])
    for q in results.get('feature_requests', []):
        writer.writerow(['', 1, q['text'], '', '', q['text'],
                         q.get('date', 'Unknown'), 'feedback', '',
                         q.get('qtype', ''), ''])
    return buffer.getvalue()


def to_faq_markdown(results: Dict, top_n: int = 15,
                    published: Optional[Dict[str, str]] = None,
                    curated: Optional[Dict[str, str]] = None) -> str:
    """
    Draft FAQ — the identify→act bridge. The analysis already knows the
    most-asked topics, every real phrasing, the replies that actually
    answered them, and (when the LLM drafted one) a grounded synthesis of
    those replies; this assembles all of it into a document a human can
    edit into a real FAQ instead of starting from a blank page. Topics
    with no captured answer are listed as needing an owner rather than
    silently omitted. `published` maps topic_id -> date for topics the
    user marked as already covered by live documentation; `curated` maps
    topic_id -> the human-approved answer saved on the bank, which always
    wins over a fresh draft (so curation is retroactive: exporting an old
    analysis still uses the canonical wording).
    """
    published = published or {}
    curated = curated or {}
    total = results.get('total_questions', 0)
    lines = [
        '# Draft FAQ',
        '',
        f'Generated from an analysis of {total} question(s). AI-drafted '
        'answers are grounded summaries of your own thread replies (the '
        'raw replies are quoted underneath as receipts) — verify before '
        'publishing.',
        '',
    ]
    groups = list(results.get('groups', []))[:top_n]
    if not groups:
        lines += ['_No recurring topics in this analysis yet — upload more '
                  'history to build an FAQ._', '']

    needs_owner = []
    for i, group in enumerate(groups, 1):
        title = group.get('topic') or group['representative_question']
        pub_date = published.get(group.get('topic_id') or '')
        lines.append(f'## {i}. {title}'
                     + (f' _(FAQ published {pub_date})_' if pub_date else ''))
        lines.append('')
        meta_bits = [f"asked {group['count']} time(s)"]
        date_range = group.get('date_range') or {}
        if date_range.get('first_asked'):
            meta_bits.append(f"first {date_range['first_asked']}")
            if date_range.get('last_asked') and \
                    date_range['last_asked'] != date_range['first_asked']:
                meta_bits.append(f"last {date_range['last_asked']}")
        lines.append(f"_{' · '.join(meta_bits)}_")
        lines.append('')
        if group.get('summary'):
            lines += [group['summary'], '']

        lines.append('**As actually asked:**')
        lines.append('')
        seen_texts = []
        for q in group.get('questions', []):
            text = (q.get('text') or '').strip()
            if text and text not in seen_texts:
                seen_texts.append(text)
        for text in seen_texts[:5]:
            lines.append(f'- {text}')
        lines.append('')

        # Confirmed replies, best-first (the reply before "thanks, that
        # worked" is almost certainly the fix)
        replies = []
        for q in group.get('questions', []):
            if q.get('answered') is True:
                for r in (q.get('replies') or []):
                    r = (r or '').strip()
                    if r and r not in replies:
                        replies.append(r)
        replies = rank_replies(replies)

        cur = curated.get(group.get('topic_id') or '')
        if isinstance(cur, dict):
            curated_answer = (cur.get('answer') or '').strip()
            answer_updated = cur.get('updated')
        else:
            curated_answer = (cur or '').strip()
            answer_updated = None
        if not curated_answer:
            curated_answer = (group.get('curated_answer') or '').strip()
            answer_updated = group.get('answer_updated')
        draft = (group.get('draft_answer') or '').strip()
        if curated_answer:
            lines.append('**Answer** _(curated — approved wording saved on '
                         'this topic)_:')
            lines.append('')
            lines.append(curated_answer)
            lines.append('')
            # Staleness nudge: confirmed replies that arrived AFTER the
            # answer was approved may contain fixes the doc lacks
            updated_day = parse_question_date(answer_updated)
            if updated_day:
                newer = sum(
                    1 for q in group.get('questions', [])
                    if q.get('answered') is True
                    and (d := parse_question_date(q.get('date'))) is not None
                    and d > updated_day)
                if newer:
                    lines.append(f'_{newer} newly answered ask(s) since this '
                                 f'answer was saved ({answer_updated}) — '
                                 'the source replies below may contain '
                                 'updates worth folding in._')
                    lines.append('')
        elif draft:
            lines.append('**Draft answer** _(AI-condensed from the replies '
                         'below — every fact checked against them; edit '
                         'before publishing)_:')
            lines.append('')
            lines.append(draft)
            lines.append('')
        if replies:
            lines.append('**Source replies:**' if (curated_answer or draft) else
                         '**Draft answer material (from thread replies):**')
            lines.append('')
            for r in replies[:6]:
                lines.append(f'> {r}')
                lines.append('>')
            lines[-1] = ''
        if not replies and not draft and not curated_answer:
            lines.append('_No captured answer yet — needs an owner._')
            lines.append('')
            needs_owner.append(group)

    # Answered singletons: asked once, but the thread contains a real
    # answer — free FAQ entries the recurring list would skip
    answered_uniques = [q for q in results.get('ungrouped_questions', [])
                        if q.get('answered') is True and q.get('replies')]
    if answered_uniques:
        lines.append('## Also answered (asked once)')
        lines.append('')
        lines.append('Not recurring yet, but the thread already contains '
                     'the answer — cheap entries while the knowledge is '
                     'fresh:')
        lines.append('')
        for q in answered_uniques[:10]:
            lines.append(f"**{q['text']}**")
            lines.append('')
            for r in rank_replies(q['replies'])[:2]:
                lines.append(f'> {r}')
                lines.append('>')
            lines[-1] = ''
        lines.append('')

    if needs_owner:
        # Write-first ranking: frequent AND recent is where a doc saves the
        # most support time. Recency is measured against the newest date in
        # the DATA (not the calendar), so historical exports rank fairly.
        def _days(group):
            out = []
            for q in group.get('questions', []):
                day = parse_question_date(q.get('date'))
                if day:
                    out.append(day)
            return out
        all_days = [d for g in groups for d in _days(g)]
        anchor = max(all_days) if all_days else None

        def _priority(group):
            days = _days(group)
            recent = sum(1 for d in days
                         if anchor and (anchor - d).days <= 30)
            newest = max(days).toordinal() if days else 0
            return (recent, group.get('count', 0), newest)

        lines.append('## Topics still needing an answer')
        lines.append('')
        lines.append('These recur but no thread reply resolved them — '
                     'ranked by frequency and recency, the highest-value '
                     'FAQ entries to write first:')
        lines.append('')
        for group in sorted(needs_owner, key=_priority, reverse=True):
            title = group.get('topic') or group.get('representative_question', '')
            recent, count, _ = _priority(group)
            evidence = [f"asked {count} time(s)"]
            if anchor and recent:
                evidence.append(f"{recent} in the newest 30 days of data")
            date_range = group.get('date_range') or {}
            if date_range.get('last_asked'):
                evidence.append(f"last {date_range['last_asked']}")
            lines.append(f"- **{title}** — {', '.join(evidence)}")
        lines.append('')

    return '\n'.join(lines)


def to_markdown(results: Dict) -> str:
    """Readable Markdown report."""
    # Analyses saved by old versions can miss newer metadata keys — a
    # legacy file must still export, not 500
    meta = results.get('metadata') or {}
    lines = [
        '# Question Analysis Report',
        '',
        f"- **Analyzed at:** {meta.get('analyzed_at', 'Unknown')}",
        f"- **Provider / model:** {meta.get('provider', 'Unknown')} / "
        f"{meta.get('model', 'Unknown')}",
        f"- **Similarity threshold:** {meta.get('similarity_threshold', 'Unknown')}",
        f"- **Total questions:** {results.get('total_questions', 0)}",
        f"- **Question groups:** {results.get('total_groups', 0)}",
        f"- **Unique (ungrouped) questions:** {len(results.get('ungrouped_questions', []))}",
    ]
    if results.get('feature_requests'):
        lines.append(f"- **Product feedback:** {len(results['feature_requests'])}")
    # Answered is only measurable when the export contained thread replies;
    # without them the honest value is "no data", not zero
    if results.get('threads_present'):
        lines.append(f"- **Answered (via thread replies):** "
                     f"{results.get('answered_questions', 0)}")
    lines.append('')
    if results.get('executive_summary'):
        lines += ['## Executive Summary', '', results['executive_summary'], '']

    themes = results.get('themes') or []
    if themes:
        lines += ['## Themes', '']
        for t in themes:
            lines.append(f"- **{t['name']}** — {t['count']} question(s)")
        lines.append('')

    lines += ['## Top Question Groups', '']

    for rank, group in enumerate(results.get('groups') or [], 1):
        title = group.get('topic') or ''
        lines.append(f"### #{rank} — {title + ' — ' if title else ''}"
                     f"asked {group.get('count', '?')} times")
        lines.append('')
        lines.append(f"**{group.get('representative_question', '')}**")
        lines.append('')
        if group.get('summary'):
            lines.append(group['summary'])
            lines.append('')
        if group.get('keywords'):
            lines.append(f"Keywords: {', '.join(group['keywords'])}")
        if group.get('theme'):
            lines.append(f"Theme: {group['theme']}")
        date_range = group.get('date_range') or {}
        if date_range.get('first_asked'):
            lines.append(f"First asked: {date_range['first_asked']} — "
                         f"Last asked: {date_range['last_asked']}")
        if group.get('avg_similarity') is not None:
            lines.append(f"Average similarity: {group['avg_similarity']:.2%}")
        if group.get('answered'):
            lines.append(f"Answered occurrences: {group['answered']}")
        lines.append('')
        lines.append('<details><summary>All questions in this group</summary>')
        lines.append('')
        for q in (group.get('questions') or []):
            lines.append(f"- {q.get('text', '')} _({q.get('date', 'Unknown')})_")
        lines.append('')
        lines.append('</details>')
        lines.append('')

    ungrouped = results.get('ungrouped_questions', [])
    if ungrouped:
        lines.append(f"## Unique Questions ({len(ungrouped)})")
        lines.append('')
        for q in ungrouped:
            markers = []
            if q.get('needs_review'):
                markers.append('needs review')
            if q.get('answered') is True:
                markers.append('answered')
            suffix = f" — _{', '.join(markers)}_" if markers else ''
            lines.append(f"- {q.get('text', '')} _({q.get('date', 'Unknown')})_{suffix}")
        lines.append('')

    feedback = results.get('feature_requests', [])
    if feedback:
        lines.append(f"## Product Feedback ({len(feedback)})")
        lines.append('')
        lines.append('Feature requests routed out of the support funnel:')
        lines.append('')
        for q in feedback:
            lines.append(f"- {q.get('text', '')} _({q.get('date', 'Unknown')})_")
        lines.append('')

    dropped = results.get('dropped_questions', [])
    if dropped:
        lines.append(f"## Removed During Analysis ({len(dropped)})")
        lines.append('')
        lines.append('Provenance trail — duplicates and phantoms, each with '
                     'its reason; nothing is ever silently consumed:')
        lines.append('')
        for q in dropped:
            lines.append(f"- ~~{q.get('text', '')}~~ — {q.get('reason', '')}")
        lines.append('')

    return '\n'.join(lines)
