"""
Week-in-Review statistics computed from analysis results.

Weeks are CALENDAR weeks (Monday through Sunday), the same bucketing the
topic-history chart uses, anchored to the most recent question date in the
analysis (not "today") so historical transcripts still produce a sensible
"this week vs last week" view. A specific week (its Monday) can be
selected; the default is the latest week with data.
"""

import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from .textutil import (DATE_MONTH_NAME, DATE_NUMERIC_MDY,
                       DATE_NUMERIC_YMD)

TREND_WEEKS = 6

_NUMERIC_YMD = re.compile(DATE_NUMERIC_YMD)
_NUMERIC_MDY = re.compile(DATE_NUMERIC_MDY)
_MONTH_NAME = re.compile(DATE_MONTH_NAME)


def parse_question_date(raw: Optional[str]) -> Optional[date]:
    """Parse a question's date string. Returns None when unparseable."""
    if not raw or raw == 'Unknown':
        return None

    match = _NUMERIC_YMD.search(raw)
    if match:
        try:
            return date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            return None

    match = _NUMERIC_MDY.search(raw)
    if match:
        try:
            return date(int(match[3]), int(match[1]), int(match[2]))  # MM/DD/YYYY
        except ValueError:
            return None

    match = _MONTH_NAME.search(raw)
    if match:
        # %b only accepts the 3-letter form, but the recognizer (and real
        # transcripts) also produce "Sept" — normalize to keep the invariant
        # that every recognized date is also parseable
        month = match[1]
        if month.lower() == 'sept':
            month = 'Sep'
        text = f"{month} {match[2]} {match[3]}"
        for fmt in ('%b %d %Y', '%B %d %Y'):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue

    return None


def _short_date(d: date) -> str:
    return f"{d.strftime('%b')} {d.day}"


def _week_label(start: date, end: date) -> str:
    if start.month == end.month:
        return f"{start.strftime('%b')} {start.day} – {end.day}, {end.year}"
    return f"{_short_date(start)} – {_short_date(end)}, {end.year}"


def _dated_questions(questions: List[Dict]) -> List[Dict]:
    """Attach parsed dates; drop questions without one."""
    dated = []
    for q in questions:
        parsed = parse_question_date(q.get('date'))
        if parsed is not None:
            dated.append({**q, '_parsed_date': parsed})
    return dated


def monday_of(d: date) -> date:
    """The Monday starting the calendar week that contains d."""
    return d - timedelta(days=d.weekday())


def compute_weekly_stats(results: Dict,
                         week: Optional[date] = None) -> Optional[Dict]:
    """
    Compute Week-in-Review stats from an analysis result.

    `week` selects the calendar week to review (any date inside it); the
    default is the newest question's week. Returns None when no question
    dates could be parsed (the caller should fall back to an
    "insufficient data" state).
    """
    # Build candidate rows: real groups plus each ungrouped question on its
    # own. Rows carry the same metadata the dashboard shows (theme, summary,
    # recurring badge, answered, needs-review) — the week view must not be a
    # stripped-down cousin.
    effective_threshold = (results.get('metadata') or {}).get('effective_threshold')
    rows = []
    for group in results.get('groups', []):
        # .get() everywhere: a legacy or hand-edited analysis must degrade,
        # not 500 the weekly endpoint (same contract as the md exporters)
        avg = group.get('avg_similarity')
        rows.append({
            'question': group.get('representative_question', ''),
            'topic': group.get('topic'),
            'theme': group.get('theme'),
            'summary': group.get('summary'),
            'seen_in': group.get('seen_in_analyses', 0),
            'ai_confirmed': bool(effective_threshold and avg is not None
                                 and avg < effective_threshold),
            'needs_review': any(q.get('needs_review')
                                for q in group.get('questions', [])),
            'keywords': group.get('keywords', []),
            'similarity': f"{round(avg * 100)}%" if avg is not None else '—',
            'questions': _dated_questions(group.get('questions', [])),
        })
    for q in results.get('ungrouped_questions', []):
        rows.append({
            'question': q.get('text', ''),
            'topic': None,
            'theme': q.get('theme'),
            'summary': None,
            'seen_in': 0,
            'ai_confirmed': False,
            'needs_review': bool(q.get('needs_review')),
            'keywords': [],
            'similarity': '—',
            'questions': _dated_questions([q]),
        })

    all_dates = [q['_parsed_date'] for row in rows for q in row['questions']]
    if not all_dates:
        return None

    latest_monday = monday_of(max(all_dates))
    selected_monday = monday_of(week) if week is not None else latest_monday

    def week_index(d: date) -> int:
        """0 = the selected calendar week, 1 = the week before, ..."""
        return (selected_monday - monday_of(d)).days // 7

    # Overall volume trend, oldest week first. The axis is ALWAYS anchored
    # to the newest data week — selecting a past week highlights its dot
    # and changes the stats, but never shifts the chart under the cursor
    # (rebasing the axis per click made navigation feel broken: every
    # click moved all the dots).
    week_totals = [0] * TREND_WEEKS
    for d in all_dates:
        idx = (latest_monday - monday_of(d)).days // 7
        if 0 <= idx < TREND_WEEKS:
            week_totals[idx] += 1
    trend = list(reversed(week_totals))
    trend_mondays = [latest_monday - timedelta(days=7 * idx)
                     for idx in reversed(range(TREND_WEEKS))]
    trend_labels = [_short_date(m) for m in trend_mondays]

    # Per-row counts for this week and last week
    for row in rows:
        row['this_week'] = [q for q in row['questions'] if week_index(q['_parsed_date']) == 0]
        row['last_week_count'] = sum(1 for q in row['questions']
                                     if week_index(q['_parsed_date']) == 1)

    # Last week's ranking (for movement)
    last_week_rows = sorted([r for r in rows if r['last_week_count'] > 0],
                            key=lambda r: r['last_week_count'], reverse=True)
    last_rank = {id(r): rank for rank, r in enumerate(last_week_rows, 1)}

    # This week's ranking
    this_week_rows = sorted([r for r in rows if r['this_week']],
                            key=lambda r: len(r['this_week']), reverse=True)

    groups = []
    for rank, row in enumerate(this_week_rows, 1):
        if id(row) in last_rank:
            movement = last_rank[id(row)] - rank  # positive = rose
        else:
            movement = 'new'
        groups.append({
            'rank': rank,
            'count': len(row['this_week']),
            'similarity': row['similarity'],
            'question': row['question'],
            'topic': row['topic'],
            'theme': row['theme'],
            'summary': row['summary'],
            'seenIn': row['seen_in'],
            'aiConfirmed': row['ai_confirmed'],
            'needsReview': row['needs_review'],
            'answered': sum(1 for q in row['this_week'] if q.get('answered')),
            'keywords': row['keywords'],
            'movement': movement,
            'questions': [
                {'text': q.get('text', ''), 'date': _short_date(q['_parsed_date']),
                 # Tri-state: True/False render the answered/unanswered
                 # chips, None (no thread / unmeasured) renders neither
                 'answered': q.get('answered')}
                for q in sorted(row['this_week'], key=lambda q: q['_parsed_date'], reverse=True)
            ],
        })

    # Selected-week totals come from week_index (selected-relative), not
    # from the latest-anchored trend array
    total_this_week = sum(1 for d in all_dates if week_index(d) == 0)
    total_last_week = sum(1 for d in all_dates if week_index(d) == 1)
    if total_last_week > 0:
        delta_pct = round((total_this_week - total_last_week) / total_last_week * 100)
    else:
        # No prior week to compare against: "+100%" would be a lie the very
        # first week — None tells the UI to say "first week of data"
        delta_pct = None

    answered_this_week = sum(
        1 for row in this_week_rows for q in row['this_week'] if q.get('answered'))

    # Product feedback pulse: feature requests are (rightly) excluded from
    # the ranking, but their weekly volume is part of the week's story.
    # The anchor comes from QUESTION dates only, so a feature request dated
    # after it gets a NEGATIVE week index — it still belongs to "this week"
    # (feedback trailing the last support question is common).
    # Trailing feedback (dated after the newest QUESTION) belongs to the
    # latest week; when reviewing a PAST week, only that week's feedback
    # counts — later feedback must not leak backward
    trailing_ok = selected_monday == latest_monday
    feedback_this_week = sum(
        1 for q in _dated_questions(results.get('feature_requests', []))
        if (week_index(q['_parsed_date']) <= 0 if trailing_ok
            else week_index(q['_parsed_date']) == 0))

    return {
        'weekLabel': _week_label(selected_monday, selected_monday + timedelta(days=6)),
        # Navigation metadata: the Monday of each trend point (chart dots
        # jump to that week), the selected week, and the newest week with
        # data (so the UI knows when "back to latest" applies)
        'week': selected_monday.isoformat(),
        'latestWeek': latest_monday.isoformat(),
        'trendWeeks': [m.isoformat() for m in trend_mondays],
        'totalThisWeek': total_this_week,
        'totalLastWeek': total_last_week,
        'deltaPct': delta_pct,
        'newQuestionTypes': sum(1 for g in groups if g['movement'] == 'new'),
        'groupsThisWeek': len(groups),
        'answered': answered_this_week,  # via LLM answer detection (threads only)
        'feedback': feedback_this_week,
        'trend': trend,
        'trendLabels': trend_labels,
        'groups': groups,
    }
