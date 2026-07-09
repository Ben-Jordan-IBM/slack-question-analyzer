"""Tests for multi-format transcript parsing and Slack markup cleanup."""

import json

from slack_question_analyzer.question_extractor import QuestionExtractor


def test_json_list_with_slack_ts():
    content = json.dumps([
        {"type": "message", "user": "U1", "text": "How do I reset my password?",
         "ts": "1704412800.000100"},
        {"type": "message", "user": "U2", "text": "The deploy finished.",
         "ts": "1704499200.000100"},
    ])
    questions = QuestionExtractor().parse_slack_content(content)
    assert len(questions) == 1
    assert questions[0]['text'] == 'How do I reset my password?'
    assert questions[0]['date'] == '2024-01-05'  # converted from epoch ts


def test_json_messages_envelope():
    content = json.dumps({"messages": [
        {"text": "Anyone know the wifi password?", "date": "2024-02-01"},
    ]})
    questions = QuestionExtractor().parse_slack_content(content)
    assert len(questions) == 1
    assert questions[0]['date'] == '2024-02-01'


def test_csv_with_date_and_message_columns():
    content = (
        "date,message\n"
        "2024-01-05,How do I reset my password?\n"
        "2024-01-06,All systems are green.\n"
    )
    questions = QuestionExtractor().parse_slack_content(content)
    assert len(questions) == 1
    assert questions[0]['date'] == '2024-01-05'


def test_csv_with_epoch_timestamp_column():
    content = (
        "ts,text\n"
        "1704412800,How do I reset my password?\n"
    )
    questions = QuestionExtractor().parse_slack_content(content)
    assert questions[0]['date'] == '2024-01-05'


def test_invalid_json_falls_back_to_text():
    content = "[2024-01-05] How do I reset my password?"
    questions = QuestionExtractor().parse_slack_content(content)
    assert len(questions) == 1
    # The inline date is captured without swallowing the question text
    assert questions[0]['date'] == '2024-01-05'


def test_commas_without_known_headers_fall_back_to_text():
    content = "Hello, world. How do I reset my password?"
    questions = QuestionExtractor().parse_slack_content(content)
    assert len(questions) == 1


def test_clean_slack_markup():
    clean = QuestionExtractor.clean_slack_markup
    assert clean("<@U123ABC> how do I configure <http://example.com|the webhook>?") == \
        "how do I configure the webhook?"
    assert clean("see <#C042XYZ|deploys> :rocket:") == "see #deploys"
    assert clean("is this broken? ```Traceback (most recent call last)```") == "is this broken?"
    assert clean("a &amp; b &lt;ok&gt;") == "a & b <ok>"
    assert clean("plain `inline code` text") == "plain inline code text"
    assert clean("<!channel> anyone seen <https://status.example.com>?") == "anyone seen ?"


def test_thread_replies_attached_to_parent_question():
    content = json.dumps([
        {"text": "How do I reset my password?", "ts": "1704412800.0"},
        {"text": "Go to settings > security > reset.", "ts": "1704412900.0",
         "thread_ts": "1704412800.0"},
        {"text": "thanks, worked! :tada:", "ts": "1704413000.0",
         "thread_ts": "1704412800.0"},
    ])
    questions = QuestionExtractor().parse_slack_content(content)
    assert len(questions) == 1  # replies are not standalone messages
    assert questions[0]['replies'] == ['Go to settings > security > reset.',
                                       'thanks, worked!']


def test_blocks_used_when_text_is_empty():
    """Modern Slack exports: empty 'text', content in rich-text 'blocks'."""
    content = json.dumps([{
        'ts': '1704412800.0',
        'text': '',
        'blocks': [{
            'type': 'rich_text',
            'elements': [
                {'type': 'rich_text_section', 'elements': [
                    {'type': 'text', 'text': 'How do I configure '},
                    {'type': 'link', 'url': 'https://x.test', 'text': 'the webhook'},
                    {'type': 'text', 'text': '?'},
                ]},
                {'type': 'rich_text_preformatted', 'elements': [
                    {'type': 'text', 'text': 'Traceback: should be skipped'},
                ]},
            ],
        }],
    }])
    questions = QuestionExtractor().parse_slack_content(content)
    assert len(questions) == 1
    assert questions[0]['text'] == 'How do I configure the webhook?'
    assert 'Traceback' not in questions[0]['original_message']


def test_blocks_preserve_announcement_signals():
    """Emoji shortcodes and list bullets must survive block flattening —
    they are the structural signals looks_like_announcement reads. The old
    join dropped emoji elements entirely and glued list items into one
    line, making blocks-sourced announcements invisible to the filter."""
    from slack_question_analyzer.question_extractor import QuestionExtractor
    blocks = [{
        'type': 'rich_text',
        'elements': [
            {'type': 'rich_text_section', 'elements': [
                {'type': 'emoji', 'name': 'rocket'},
                {'type': 'text', 'text': ' New & Improved Widget Sales Kit '
                                         'is Here! '},
                {'type': 'emoji', 'name': 'sparkles'},
                {'type': 'text', 'text': " We're excited to share the "
                                         'latest version.'},
            ]},
            {'type': 'rich_text_list', 'elements': [
                {'type': 'rich_text_section', 'elements': [
                    {'type': 'emoji', 'name': 'movie_camera'},
                    {'type': 'text', 'text': ' Demo-ready assets'}]},
                {'type': 'rich_text_section', 'elements': [
                    {'type': 'text', 'text': 'Simplified structure'}]},
                {'type': 'rich_text_section', 'elements': [
                    {'type': 'text', 'text': 'Sharper messaging'}]},
            ]},
            {'type': 'rich_text_section', 'elements': [
                {'type': 'text', 'text': 'Explore the new kit here. '
                                         'Happy selling! cc: @sales'}]},
        ],
    }]
    text = QuestionExtractor._text_from_blocks(blocks)
    assert text.count(':') >= 6          # :rocket: :sparkles: :movie_camera:
    assert text.count('\n* ') == 3       # list structure survives

    from slack_question_analyzer.textutil import looks_like_announcement
    assert looks_like_announcement(text)


def test_json_announcement_filtered_but_threaded_question_kept(monkeypatch):
    """End-to-end on a Slack JSON export: a blocks-only announcement yields
    zero questions, while a threaded real question survives with its
    replies attached (announcements stay in the corpus for accounting)."""
    import numpy as np
    from slack_question_analyzer.analyzer import QuestionAnalyzer
    content = json.dumps([
        {
            'ts': '1704412800.0',
            'text': '',
            'blocks': [{'type': 'rich_text', 'elements': [
                {'type': 'rich_text_section', 'elements': [
                    {'type': 'emoji', 'name': 'rocket'},
                    {'type': 'text', 'text': " New Sales Kit is Here! We're "
                                             'excited to share it. Why '
                                             'should customers care? '},
                    {'type': 'emoji', 'name': 'fire'},
                    {'type': 'emoji', 'name': 'briefcase'},
                    {'type': 'text', 'text': ' Explore the new kit here. '
                                             'cc: @sales'},
                ]}]}],
        },
        {'ts': '1704499200.0', 'text': 'How do I configure retry limits?',
         'thread_ts': '1704499200.0'},
        {'ts': '1704499300.0', 'text': 'Set retries in the transfer settings.',
         'thread_ts': '1704499200.0'},
    ])
    monkeypatch.setenv('GROUP_LABELS', 'off')
    analyzer = QuestionAnalyzer(provider='ollama', use_disk_cache=False)
    index = {}

    def fake_batch(texts, progress_callback=None):
        for t in texts:
            index.setdefault(t, len(index))
        dim = max(8, len(index))
        return np.array([[1.0 if j == index[t] else 0.0 for j in range(dim)]
                         for t in texts])
    monkeypatch.setattr(analyzer.similarity_analyzer, 'get_embeddings_batch',
                        fake_batch)
    results = analyzer.analyze_slack_content(content)

    texts = ([q['text'] for g in results['groups'] for q in g['questions']]
             + [q['text'] for q in results['ungrouped_questions']])
    assert any('retry limits' in t for t in texts)
    assert not any('customers care' in t.lower() for t in texts)
    assert results['total_questions'] == 1
    assert results['threads_present'] is True  # replies still counted


def test_nonempty_text_preferred_over_blocks():
    content = json.dumps([{
        'ts': '1704412800.0',
        'text': 'How do I reset my password?',
        'blocks': [{'type': 'rich_text', 'elements': [
            {'type': 'rich_text_section', 'elements': [
                {'type': 'text', 'text': 'different blocks content?'}]}]}],
    }])
    questions = QuestionExtractor().parse_slack_content(content)
    assert questions[0]['text'] == 'How do I reset my password?'


def test_markup_in_json_messages_is_cleaned():
    content = json.dumps([
        {"text": "<@U123> how do I configure <http://ex.com|the webhook>?", "ts": "1704412800"},
    ])
    questions = QuestionExtractor().parse_slack_content(content)
    assert questions[0]['text'] == 'how do I configure the webhook?'


SLACK_THREAD_COPY = """Ben Jordan
  10:15 AM
How do I raise the SFTP timeout for big transfers?
It keeps aborting around 2GB.

3 replies

Jane Doe
  10:20 AM
Set transfer.timeout to 300 in the partner settings and restart the listener.

Ben Jordan
  10:24 AM
thanks, that worked!
"""

SLACK_CHANNEL_COPY = """Monday, June 8th

Ben Jordan
  10:15 AM
How do I raise the SFTP timeout for big transfers?

Jane Doe
Jun 9th at 2:30 PM
Is there a REST API to deactivate scheduled actions?
(edited)
"""


def test_pasted_slack_thread_becomes_root_plus_replies():
    """Text copied out of a Slack THREAD (author + timestamp lines, with
    the 'N replies' divider) parses as ONE message whose replies feed
    answer detection — the paste-a-thread path."""
    extractor = QuestionExtractor()
    messages = extractor.extract_messages(SLACK_THREAD_COPY)
    assert len(messages) == 1
    assert 'SFTP timeout' in messages[0]['text']
    assert 'aborting around 2GB' in messages[0]['text']
    assert len(messages[0]['replies']) == 2
    assert messages[0]['replies'][0].startswith('Set transfer.timeout')
    # Furniture never leaks into content
    assert 'replies' not in messages[0]['text']


def test_pasted_slack_channel_copy_keeps_messages_standalone():
    """Without the replies divider, each author+timestamp message stands
    alone (a channel copy behaves like a dashed transcript), and day
    dividers / timestamp lines supply dates."""
    extractor = QuestionExtractor()
    messages = extractor.extract_messages(SLACK_CHANNEL_COPY)
    assert len(messages) == 2
    assert messages[0]['text'].startswith('How do I raise')
    assert messages[1]['text'].startswith('Is there a REST API')
    assert 'edited' not in messages[1]['text']
    assert messages[0]['date'] and 'June 8' in messages[0]['date']
    assert messages[1]['date'] and 'Jun 9' in messages[1]['date']


def test_dashed_transcripts_do_not_trigger_the_slack_copy_parser():
    """The Slack-copy detector must not claim dashed transcripts (they
    have date lines, never author + clock-time line pairs)."""
    content = ("June 5, 2026\nHow do I reset my password?\n"
               "-----------------------------------------------------------\n"
               "June 6, 2026\nWhat is the deploy schedule?\n")
    extractor = QuestionExtractor()
    messages = extractor.extract_messages(content)
    assert len(messages) == 2
    assert messages[0]['date'] == 'June 5, 2026'


def test_incident_notes_and_logs_are_not_hijacked_as_slack_copies():
    """Audit regression: 'short label / clock time / body' is also the
    shape of incident notes and service logs. Single-word labels with bare
    24-hour times (and any line with seconds) must NOT trigger the
    Slack-copy parser — the text parser keeps every line."""
    notes = ("Incident\n14:32\nTransfers started failing on node 2.\n\n"
             "Mitigation\n15:10\nFailover to node 1 restored service.\n")
    log = ("Startup\n06:47:18\nService initialized in 2.3s.\n\n"
           "Shutdown\n23:59:01\nGraceful stop requested.\n")
    extractor = QuestionExtractor()
    for content, label in ((notes, 'Incident'), (log, 'Startup')):
        messages = extractor.extract_messages(content)
        joined = ' '.join(m['text'] for m in messages)
        assert label in joined  # label lines survive (text parser path)

    # Real Slack copies still work both ways: AM/PM with a single-word
    # display name, and bare 24-hour times with a full name
    ampm = ("ben\n3:42 PM\nHow do I raise the timeout?\n\n"
            "jane\n3:50 PM\nIs there a REST API for actions?\n")
    h24 = ("Ben Jordan\n14:32\nHow do I raise the timeout?\n\n"
           "Jane Doe\n14:50\nIs there a REST API for actions?\n")
    for content in (ampm, h24):
        messages = extractor.extract_messages(content)
        assert len(messages) == 2
        assert messages[0]['text'].startswith('How do I raise')


def test_yearless_copy_dates_never_land_in_the_future(monkeypatch):
    """A yearless 'Dec 20th' pasted in January means LAST December —
    Slack omits the year for the past twelve months, not the future."""
    import slack_question_analyzer.question_extractor as qe

    class FrozenDT(qe.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 5)

    monkeypatch.setattr(qe, 'datetime', FrozenDT)
    extractor = QuestionExtractor()
    assert extractor._copy_date('Dec 20th at 3:42 PM') == 'Dec 20, 2025'
    assert extractor._copy_date('Jan 3rd at 9:00 AM') == 'Jan 3, 2026'
    assert extractor._copy_date('Jul 7th, 2026 at 1:00 PM') == 'Jul 7, 2026'
