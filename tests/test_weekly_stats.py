"""Tests for Week-in-Review statistics."""

from datetime import date

from slack_question_analyzer.weekly_stats import parse_question_date, compute_weekly_stats, TREND_WEEKS


def test_parse_question_date_formats():
    assert parse_question_date('2024-03-20') == date(2024, 3, 20)
    assert parse_question_date('2024/3/5') == date(2024, 3, 5)
    assert parse_question_date('3/20/2024') == date(2024, 3, 20)
    assert parse_question_date('March 20, 2024') == date(2024, 3, 20)
    assert parse_question_date('Mar 20 2024') == date(2024, 3, 20)
    assert parse_question_date('Posted on 2024-03-20 by alice') == date(2024, 3, 20)


def test_parse_question_date_rejects_garbage():
    assert parse_question_date(None) is None
    assert parse_question_date('Unknown') is None
    assert parse_question_date('hello world') is None
    assert parse_question_date('2024-13-45') is None


def make_results():
    """Anchor date 2024-03-20: 'this week' is Mar 14-20, 'last week' Mar 7-13."""
    return {
        'groups': [
            {
                'representative_question': 'How do I reset my password?',
                'avg_similarity': 0.9,
                'keywords': ['password'],
                'questions': [
                    {'text': 'a1?', 'date': '2024-03-20'},
                    {'text': 'a2?', 'date': '2024-03-19'},
                    {'text': 'a3?', 'date': '2024-03-10'},  # last week
                ],
            },
            {
                'representative_question': 'What is the deploy schedule?',
                'avg_similarity': 0.88,
                'keywords': ['deploy'],
                'questions': [
                    {'text': 'b1?', 'date': '2024-03-18'},
                    {'text': 'b2?', 'date': '2024-03-17'},
                    {'text': 'b3?', 'date': '2024-03-16'},
                ],
            },
        ],
        'ungrouped_questions': [
            {'text': 'How does SSO work?', 'date': '2024-03-09'},   # last week only
            {'text': 'Where are the logs?', 'date': '2024-03-15'},  # this week
        ],
    }


def test_compute_weekly_stats_totals_and_trend():
    weekly = compute_weekly_stats(make_results())

    assert weekly['weekLabel'] == 'Mar 14 – 20, 2024'
    assert weekly['totalThisWeek'] == 6
    assert weekly['totalLastWeek'] == 2
    assert weekly['deltaPct'] == 200
    assert len(weekly['trend']) == TREND_WEEKS
    assert len(weekly['trendLabels']) == TREND_WEEKS
    assert weekly['trend'][-1] == 6  # newest week last
    assert weekly['trend'][-2] == 2


def test_compute_weekly_stats_ranking_and_movement():
    weekly = compute_weekly_stats(make_results())
    groups = weekly['groups']

    # Deploy group has 3 questions this week and didn't exist last week
    assert groups[0]['question'] == 'What is the deploy schedule?'
    assert groups[0]['count'] == 3
    assert groups[0]['movement'] == 'new'

    # Password group was rank 1 last week, rank 2 now: movement -1
    assert groups[1]['question'] == 'How do I reset my password?'
    assert groups[1]['count'] == 2
    assert groups[1]['movement'] == -1
    # Only this week's questions are listed
    assert all(q['date'] in ('Mar 20', 'Mar 19') for q in groups[1]['questions'])

    # Ungrouped question from this week appears as its own row
    assert groups[2]['question'] == 'Where are the logs?'
    assert groups[2]['similarity'] == '—'
    assert weekly['newQuestionTypes'] == 2

    # Last-week-only question must not appear
    assert all(g['question'] != 'How does SSO work?' for g in groups)


def test_compute_weekly_stats_without_dates():
    results = {
        'groups': [{
            'representative_question': 'A?',
            'avg_similarity': 1.0,
            'keywords': [],
            'questions': [{'text': 'a?', 'date': 'Unknown'}],
        }],
        'ungrouped_questions': [],
    }
    assert compute_weekly_stats(results) is None


def test_compute_weekly_stats_empty():
    assert compute_weekly_stats({'groups': [], 'ungrouped_questions': []}) is None


def test_answered_count_for_this_week():
    results = {
        'groups': [],
        'ungrouped_questions': [
            {'text': 'a?', 'date': '2024-03-20', 'answered': True},
            {'text': 'b?', 'date': '2024-03-19'},
            {'text': 'c?', 'date': '2024-03-09', 'answered': True},  # last week
        ],
    }
    weekly = compute_weekly_stats(results)
    assert weekly['answered'] == 1  # only this week's answered questions count


def test_delta_with_empty_last_week():
    results = {
        'groups': [],
        'ungrouped_questions': [
            {'text': 'a?', 'date': '2024-03-20'},
            {'text': 'b?', 'date': '2024-03-19'},
        ],
    }
    weekly = compute_weekly_stats(results)
    assert weekly['totalThisWeek'] == 2
    assert weekly['totalLastWeek'] == 0
    assert weekly['deltaPct'] == 100
