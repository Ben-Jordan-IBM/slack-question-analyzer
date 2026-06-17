"""
Export analysis results as CSV or Markdown strings.

Used by both the CLI (writing files) and the API server (download endpoint).
"""

import csv
import io
from typing import Dict


def to_csv(results: Dict) -> str:
    """Flat CSV: one row per question (grouped, unique, and feedback)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
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


def to_markdown(results: Dict) -> str:
    """Readable Markdown report."""
    meta = results['metadata']
    lines = [
        '# Question Analysis Report',
        '',
        f"- **Analyzed at:** {meta['analyzed_at']}",
        f"- **Provider / model:** {meta['provider']} / {meta['model']}",
        f"- **Similarity threshold:** {meta['similarity_threshold']}",
        f"- **Total questions:** {results['total_questions']}",
        f"- **Question groups:** {results['total_groups']}",
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

    for rank, group in enumerate(results['groups'], 1):
        title = group.get('topic') or ''
        lines.append(f"### #{rank} — {title + ' — ' if title else ''}asked {group['count']} times")
        lines.append('')
        lines.append(f"**{group['representative_question']}**")
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
        lines.append(f"Average similarity: {group['avg_similarity']:.2%}")
        if group.get('answered'):
            lines.append(f"Answered occurrences: {group['answered']}")
        lines.append('')
        lines.append('<details><summary>All questions in this group</summary>')
        lines.append('')
        for q in group['questions']:
            lines.append(f"- {q['text']} _({q.get('date', 'Unknown')})_")
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
            lines.append(f"- {q['text']} _({q.get('date', 'Unknown')})_{suffix}")
        lines.append('')

    feedback = results.get('feature_requests', [])
    if feedback:
        lines.append(f"## Product Feedback ({len(feedback)})")
        lines.append('')
        lines.append('Feature requests routed out of the support funnel:')
        lines.append('')
        for q in feedback:
            lines.append(f"- {q['text']} _({q.get('date', 'Unknown')})_")
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
