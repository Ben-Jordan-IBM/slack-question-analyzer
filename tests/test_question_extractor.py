"""Tests for question extraction and normalization."""

from slack_question_analyzer.question_extractor import QuestionExtractor


def test_detects_question_mark():
    extractor = QuestionExtractor()
    assert extractor.is_question("Is the VPN down?")


def test_detects_question_word_at_start():
    extractor = QuestionExtractor()
    assert extractor.is_question("How do I reset my password")
    assert extractor.is_question("Can someone help with the deploy")


def test_detects_help_seeking_phrases():
    extractor = QuestionExtractor()
    assert extractor.is_question("Anyone know the wifi password")
    assert extractor.is_question("I was wondering if we support SSO")
    assert extractor.is_question("Not sure how to configure the linter")


def test_rejects_declarative_sentences():
    extractor = QuestionExtractor()
    # These contain auxiliary verbs mid-sentence and used to be false positives
    assert not extractor.is_question("The deploy is finished")
    assert not extractor.is_question("Our team has shipped the feature")
    assert not extractor.is_question("I did the migration yesterday")


def test_rejects_short_fragments_without_question_mark():
    extractor = QuestionExtractor()
    assert not extractor.is_question("How interesting")
    # A bare one-word '?' is a context continuation, never a standalone
    # question — extract_questions attaches it to its preceding sentence
    assert not extractor.is_question("Why?")
    assert extractor.is_question("Why not?")  # two words carry substance


def test_bare_question_fragment_attaches_to_preceding_sentence():
    extractor = QuestionExtractor()
    questions = extractor.extract_questions(
        "The sync job dies every night at 2am. Why? Nothing in the logs.")
    assert questions == ['The sync job dies every night at 2am. Why?']


def test_url_keeps_its_hostname_as_the_subject():
    """Deleting the whole URL made 'tried <toolA>?' and 'tried <toolB>?'
    normalize identically — a phantom 'asked 2x'."""
    extractor = QuestionExtractor()
    a = extractor.clean_slack_markup('has anyone tried https://toolA.io?')
    b = extractor.clean_slack_markup('has anyone tried https://toolB.example.com?')
    assert 'toolA.io' in a and 'toolB.example.com' in b
    assert a != b


def test_hindsight_and_forwarded_and_interjection_noise():
    extractor = QuestionExtractor()
    assert extractor.extract_questions(
        'thanks! that did it. should have read the docs first') == []
    assert extractor.extract_questions(
        'Forwarding from customer: "Can we cap concurrent transfers per node?"'
    ) == ['Can we cap concurrent transfers per node?']


def test_naked_urls_do_not_become_questions():
    extractor = QuestionExtractor()
    # A query string's '?' used to split into a garbage "question" ('com/search?')
    cleaned = extractor.clean_slack_markup(
        "check https://example.com/search?q=retry for details")
    assert 'http' not in cleaned and '?' not in cleaned
    # Through the real parse path (which cleans before extracting)
    questions = extractor.parse_slack_content(
        "2024-01-05\ncheck https://example.com/search?q=retry for details\n")
    assert questions == []
    # ...but a '?' that belongs to the SENTENCE survives the strip: the
    # question must not vanish with its URL
    cleaned = extractor.clean_slack_markup('Anyone tried https://newtool.io?')
    assert cleaned.endswith('?') and 'http' not in cleaned


def test_extract_questions_from_mixed_text():
    extractor = QuestionExtractor()
    text = "The build passed. How do I get access to staging? Thanks all."
    questions = extractor.extract_questions(text)
    assert questions == ["How do I get access to staging?"]


def test_normalize_strips_fillers_and_case():
    extractor = QuestionExtractor()
    normalized = extractor.normalize_question("Hi team, How do I reset my password???")
    assert normalized == "how do i reset my password???"


def test_parse_slack_content_with_separator_and_dates():
    extractor = QuestionExtractor()
    content = (
        "2024-01-05\n"
        "How do I reset my password?\n"
        "-----------------------------------------------------------\n"
        "2024-01-08\n"
        "What is the deploy schedule?\n"
    )
    questions = extractor.parse_slack_content(content)
    assert len(questions) == 2
    assert questions[0]['date'] == '2024-01-05'
    assert questions[1]['date'] == '2024-01-08'
    assert questions[0]['text'] == 'How do I reset my password?'


def test_parse_slack_content_empty():
    extractor = QuestionExtractor()
    assert extractor.parse_slack_content("") == []


def test_document_header_not_glued_onto_question():
    """A header line above the date must not leak into the first question."""
    extractor = QuestionExtractor()
    content = (
        "MFT Content from the Slack threads\n"
        "June 9, 2026\n"
        "\n"
        "Hi Team, Can I check if the Metering Agent comes pre-installed?\n"
    )
    questions = extractor.parse_slack_content(content)
    assert len(questions) == 1
    assert questions[0]['text'] == 'Can I check if the Metering Agent comes pre-installed?'
    assert 'MFT Content' not in questions[0]['text']
    assert questions[0]['date'] == 'June 9, 2026'


def test_leading_greetings_stripped():
    extractor = QuestionExtractor()
    assert extractor.strip_greeting('Hi Team, how do I reset?') == 'how do I reset?'
    assert extractor.strip_greeting('Hello everyone! Hey folks, is VPN down?') == 'is VPN down?'
    assert extractor.strip_greeting('Good morning - can someone help?') == 'can someone help?'
    # No greeting: untouched; greeting-only: kept rather than emptied
    assert extractor.strip_greeting('How do I reset?') == 'How do I reset?'
    assert extractor.strip_greeting('Hi team!') == 'Hi team!'


def test_multiline_message_keeps_sentences_apart():
    """Lines are sentence boundaries even without punctuation."""
    extractor = QuestionExtractor()
    questions = extractor.questions_from_messages([{
        'text': 'Some context line without punctuation\nHow do I configure alerts?',
        'date': '2024-01-05',
    }])
    assert len(questions) == 1
    assert questions[0]['text'] == 'How do I configure alerts?'


def test_abbreviations_do_not_split_sentences():
    """Fixture-2 regression: the splitter tore 'maintenance window, e.g.
    1am to 4am?' into fragments at the 'e.g.' period."""
    from slack_question_analyzer.question_extractor import QuestionExtractor
    extractor = QuestionExtractor()
    questions = extractor.extract_questions(
        'Is there a way to set a transfer to only run during a maintenance '
        'window, e.g. 1am to 4am?')
    assert questions == ['Is there a way to set a transfer to only run '
                         'during a maintenance window, e.g. 1am to 4am?']

    questions = extractor.extract_questions(
        'Can we whitelist protocols, i.e. SFTP and FTPS only?')
    assert len(questions) == 1 and 'i.e. SFTP' in questions[0]


def test_decimal_numbers_do_not_split_sentences():
    from slack_question_analyzer.question_extractor import QuestionExtractor
    extractor = QuestionExtractor()
    questions = extractor.extract_questions(
        'How can my customer check transaction statistics in WM MFT 10.15 for commercial purposes?')
    assert len(questions) == 1
    assert '10.15' in questions[0]


def test_quoted_lines_become_replies_not_questions():
    """'>' quoted lines are thread replies: attached to the parent for
    answer detection, never extracted as the asker's questions — a
    responder's clarifying question is not an ask."""
    extractor = QuestionExtractor()
    content = (
        'June 4, 2026\n\n'
        'Why are transfers failing intermittently to partner X?\n'
        '> Reply (Sam): Are they failing around the same time of day?\n'
        '-----------------------------------------------------------\n'
        'June 9, 2026\n\n'
        'How do I increase the SFTP connection timeout?\n'
        '> Reply (Priya): Set mft.sftp.connectionTimeout to 300000\n'
        '> and restart the Integration Server.\n'
    )
    messages = extractor.extract_messages(content)
    assert len(messages) == 2
    assert messages[0]['replies'] == [
        'Reply (Sam): Are they failing around the same time of day?']
    assert 'same time of day' not in messages[0]['text']
    # Consecutive quoted lines are ONE reply
    assert len(messages[1]['replies']) == 1
    assert 'restart the Integration Server' in messages[1]['replies'][0]

    questions = extractor.questions_from_messages(messages)
    texts = [q['text'] for q in questions]
    assert not any('same time of day' in t for t in texts)
    assert all(q.get('replies') for q in questions)


def test_blank_line_separates_replies():
    extractor = QuestionExtractor()
    messages = extractor.extract_messages(
        'June 3, 2026\n\nCan we set retention policies?\n'
        '> First reply here.\n\n> Second reply here.\n')
    assert messages[0]['replies'] == ['First reply here.', 'Second reply here.']


def test_heading_lines_are_not_message_content():
    """'# ' heading/comment lines are structural markup, like fenced code."""
    extractor = QuestionExtractor()
    messages = extractor.extract_messages(
        'Transcript header line\n'
        '# Convention: each block is one thread.\n\n'
        'June 9, 2026\n\n'
        'How do I increase the timeout?\n')
    assert len(messages) == 1
    assert 'Convention' not in messages[0]['text']
    # A '#channel' mention is NOT a heading (no space after #)
    messages = extractor.extract_messages(
        'June 9, 2026\n\n#prod-support is asking: how do I increase the timeout?\n')
    assert '#prod-support' in messages[0]['text']


def test_reply_only_block_produces_no_message():
    extractor = QuestionExtractor()
    assert extractor.extract_messages('> just a stray quoted line\n') == []


def test_and_separately_splits_compound_questions():
    """'and separately' explicitly joins two DISTINCT asks in one sentence;
    the splitter must surface both so the under-extraction safety net can
    count them (the Kafka half of such a sentence vanished silently in two
    field rounds)."""
    extractor = QuestionExtractor()
    questions = extractor.extract_questions(
        'Does MFT support mutual TLS for HTTPS partner endpoints, and '
        'separately, can it post a transfer-complete event to Kafka?')
    assert len(questions) == 2
    assert any('mutual TLS' in q for q in questions)
    assert any('Kafka' in q for q in questions)


def test_document_title_stripped_from_first_block():
    """A title line before the FIRST date header is file furniture: glued
    onto the first message it inflated the source past its 200-char
    identity cap and '(test set 2)' even faked an enumerated multi-ask
    split. Blocks without any date keep all their lines."""
    extractor = QuestionExtractor()
    content = ('MFT Content from the Slack threads (synthetic test set 2)\n'
               'June 9, 2026\n\n'
               'How do I set up a retry policy for failed transfers?\n'
               '-----------------------------------------------------------\n'
               'June 8, 2026\n\nSecond message?\n')
    messages = extractor.extract_messages(content)
    assert messages[0]['text'] == 'How do I set up a retry policy for failed transfers?'
    assert messages[0]['date'] == 'June 9, 2026'
    # A dateless snippet is NOT furniture
    assert extractor.extract_messages('no date here, just a question?\n')[0][
        'text'] == 'no date here, just a question?'


def test_hard_wrapped_sentence_is_one_question():
    """Pasted Slack text wraps mid-sentence; splitting at the wrap used to
    extract fragments that ranked (and routed) as separate questions."""
    extractor = QuestionExtractor()
    questions = extractor.extract_questions(
        "Is there a way to configure the retry\ncount for outbound transfers?")
    assert questions == [
        'Is there a way to configure the retry count for outbound transfers?']


def test_wrap_join_keeps_headers_and_new_asks_separate():
    extractor = QuestionExtractor()
    # A heading line (capitalized) is never glued onto the question
    assert extractor.extract_questions(
        "MFT Question\nHow do I enable PGP encryption?") == [
        'How do I enable PGP encryption?']
    # Two capitalized asks on their own lines stay two questions
    assert extractor.extract_questions(
        "Does it support TLS?\nCan it post to Kafka?") == [
        'Does it support TLS?', 'Can it post to Kafka?']


def test_or_continuation_stays_with_its_question():
    """'Is X better? or should we use Y?' is ONE ask phrased as
    alternatives — the '?' split used to strand the 'or' half."""
    extractor = QuestionExtractor()
    questions = extractor.extract_questions(
        "Is SFTP better for this? or should we just use HTTPS?")
    assert questions == ['Is SFTP better for this? or should we just use HTTPS?']
