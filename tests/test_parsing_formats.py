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
