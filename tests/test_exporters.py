"""Tests for CSV/Markdown exporters."""

import csv
import io

from slack_question_analyzer.exporters import to_csv, to_faq_markdown, to_markdown


def test_faq_uses_grounded_draft_with_receipts_and_appendix():
    results = {
        'total_questions': 4,
        'groups': [{
            'topic': 'SFTP Timeouts', 'topic_id': 'topic-1', 'count': 2,
            'representative_question': 'How do I raise the SFTP timeout?',
            'avg_similarity': 0.9, 'summary': 'Timeout questions.',
            'date_range': {'first_asked': 'June 2', 'last_asked': 'June 9'},
            'draft_answer': 'Set transfer.timeout to 300 and restart.',
            'questions': [
                {'text': 'How do I raise the SFTP timeout?', 'answered': True,
                 'replies': ['Set transfer.timeout to 300 and restart.',
                             'thanks, that worked!']},
                {'text': 'Transfers time out?', 'answered': False},
            ],
        }],
        'ungrouped_questions': [
            {'text': 'Can I use PGP keys with partners?', 'answered': True,
             'replies': ['Yes, upload the public key in partner settings.']},
            {'text': 'Unanswered thing?', 'answered': False, 'replies': []},
        ],
    }
    faq = to_faq_markdown(results, published={'topic-1': '2026-07-01'})
    assert '## 1. SFTP Timeouts _(FAQ published 2026-07-01)_' in faq
    assert '**Draft answer**' in faq
    assert 'Set transfer.timeout to 300 and restart.' in faq
    assert '**Source replies:**' in faq           # receipts stay
    assert '## Also answered (asked once)' in faq  # singleton appendix
    assert 'Can I use PGP keys with partners?' in faq
    assert 'Unanswered thing?' not in faq


def test_faq_without_draft_quotes_raw_replies():
    results = {
        'total_questions': 2,
        'groups': [{
            'topic': 'Alerts', 'count': 2, 'avg_similarity': 0.9,
            'representative_question': 'How do I get failure alerts?',
            'questions': [
                {'text': 'How do I get failure alerts?', 'answered': True,
                 'replies': ['Enable notifications in the admin panel.']},
                {'text': 'Alert on failures?', 'answered': False},
            ],
        }],
        'ungrouped_questions': [],
    }
    faq = to_faq_markdown(results)
    assert '**Draft answer**' not in faq
    assert '**Draft answer material (from thread replies):**' in faq
    assert 'Enable notifications in the admin panel.' in faq


def make_results(question_text):
    return {
        'groups': [{
            'representative_question': question_text,
            'count': 1,
            'avg_similarity': 0.9,
            'keywords': ['transfer'],
            'theme': '',
            'questions': [{'text': question_text, 'date': '2024-01-05'}],
        }],
        'ungrouped_questions': [],
        'feature_requests': [],
    }


def test_csv_defuses_formula_injection():
    """Transcript text is attacker-controlled; Excel executes cells starting
    with = + - @ as formulas when the CSV is opened."""
    evil = '=HYPERLINK("http://evil","click")'
    rows = list(csv.reader(io.StringIO(to_csv(make_results(evil)))))
    data_row = rows[1]
    assert data_row[2] == "'" + evil  # representative_question
    assert data_row[5] == "'" + evil  # question
    assert not any(cell.startswith('=') for cell in data_row)


def test_csv_leaves_ordinary_text_alone():
    text = 'How do I reset my password?'
    rows = list(csv.reader(io.StringIO(to_csv(make_results(text)))))
    assert rows[1][5] == text


def test_markdown_export_tolerates_legacy_analyses():
    """Analyses saved by old app versions can miss newer metadata keys —
    exporting one must degrade to 'Unknown', not crash the download."""
    legacy = {
        'groups': [{
            'representative_question': 'How do I reset my password?',
            'count': 2,
            'questions': [{'text': 'How do I reset my password?'}],
        }],
    }
    report = to_markdown(legacy)
    assert '# Question Analysis Report' in report
    assert 'Unknown' in report
    assert 'How do I reset my password?' in report


def test_faq_curated_answer_wins_over_draft():
    """A human-approved answer saved on the bank is canonical: it replaces
    the AI draft in the export, retroactively for old analyses (the
    `curated` map) and via the group snapshot."""
    results = {
        'total_questions': 3,
        'groups': [{
            'topic': 'SFTP Timeouts', 'topic_id': 'topic-1', 'count': 3,
            'representative_question': 'How do I raise the SFTP timeout?',
            'draft_answer': 'Old AI draft that should not appear.',
            'questions': [{'text': 'How do I raise the SFTP timeout?',
                           'answered': True,
                           'replies': ['Set transfer.timeout to 300.']}],
        }],
    }
    faq = to_faq_markdown(results, curated={'topic-1': 'Set transfer.timeout '
                                            'to 300 in partner settings.'})
    assert 'curated' in faq
    assert 'Set transfer.timeout to 300 in partner settings.' in faq
    assert 'Old AI draft' not in faq
    assert 'Source replies' in faq  # receipts still shown

    # Snapshot fallback: no bank map, the group carries its own copy
    results['groups'][0]['curated_answer'] = 'Snapshot answer.'
    faq = to_faq_markdown(results)
    assert 'Snapshot answer.' in faq and 'Old AI draft' not in faq


def test_faq_needs_owner_ranked_by_recency_and_frequency():
    """The write-first list puts frequent AND recent topics on top, with
    evidence per line, measured against the newest date in the data."""
    def group(topic, tid, dates):
        return {
            'topic': topic, 'topic_id': tid, 'count': len(dates),
            'representative_question': f'{topic}?',
            'date_range': {'first_asked': dates[0], 'last_asked': dates[-1]},
            'questions': [{'text': f'{topic}?', 'date': d} for d in dates],
        }
    results = {
        'total_questions': 5,
        'groups': [
            # More total asks, but all old
            group('Old Frequent Topic', 't-old',
                  ['2026-01-05', '2026-01-12', '2026-01-19']),
            # Fewer asks, but recent (data anchor is 2026-07-01)
            group('Recent Hot Topic', 't-hot', ['2026-06-20', '2026-07-01']),
        ],
    }
    faq = to_faq_markdown(results)
    section = faq.split('## Topics still needing an answer', 1)[1]
    assert section.index('Recent Hot Topic') < section.index('Old Frequent Topic')
    assert '2 in the newest 30 days of data' in section
    assert 'asked 3 time(s)' in section


def test_faq_curated_answer_stale_nudge():
    """Confirmed replies that arrive AFTER the answer was approved may
    contain fixes the doc lacks — the export says so; answers with no
    newer confirmed replies stay quiet."""
    def results(ask_date):
        return {
            'total_questions': 2,
            'groups': [{
                'topic': 'SFTP Timeouts', 'topic_id': 'topic-1', 'count': 2,
                'representative_question': 'How do I raise the SFTP timeout?',
                'questions': [
                    {'text': 'How do I raise the SFTP timeout?',
                     'date': ask_date, 'answered': True,
                     'replies': ['Set transfer.timeout to 600 now.']},
                ],
            }],
        }
    curated = {'topic-1': {'answer': 'Set transfer.timeout to 300.',
                           'updated': '2026-06-01'}}

    faq = to_faq_markdown(results('2026-06-20'), curated=curated)
    assert '1 newly answered ask(s) since this answer was saved (2026-06-01)' in faq

    faq = to_faq_markdown(results('2026-05-20'), curated=curated)
    assert 'newly answered' not in faq
