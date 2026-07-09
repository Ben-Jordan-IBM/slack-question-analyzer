"""
Question extraction and parsing module.
Extracts questions from Slack content in multiple formats:

- Plain text with dashed separators between messages
- Slack JSON exports (a list of message objects, or {"messages": [...]})
- CSV with a text/message/question column and optional date column

Slack markup (mentions, links, emoji codes, code blocks) is stripped before
question detection so it doesn't pollute grouping.
"""

import re
import csv
import io
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional

from .textutil import canonical_source, DATE_PATTERNS


class QuestionExtractor:
    """Extracts and normalizes questions from Slack messages."""

    # A sentence is treated as a question when it starts with an
    # interrogative/auxiliary word, not merely when it contains one —
    # otherwise nearly every declarative sentence matches.
    QUESTION_STARTERS = (
        r'^(how|what|when|where|why|who|whom|whose|which|can|could|would|will|'
        r'should|shall|is|are|was|were|does|do|did|has|have|had|may|might|am)\b'
    )

    # Help-seeking phrases that signal a question anywhere in the sentence
    HELP_PATTERNS = (
        r'\b(anyone|anybody|someone|somebody)\s+(know|knows|have|has|tried|'
        r'familiar|help|else|here)\b'
        r'|\b(any\s+(idea|ideas|thoughts|suggestions|recommendations|pointers))\b'
        r'|\b(is\s+there\s+a\s+way)\b'
        r'|\b(wondering\s+(if|how|whether|what|why))\b'
        r'|\b(not\s+sure\s+(how|if|whether|why|what))\b'
    )

    MIN_QUESTION_WORDS = 3  # Ignore fragments like "Why?" split from context

    def __init__(self):
        self.starter_regex = re.compile(self.QUESTION_STARTERS, re.IGNORECASE)
        self.help_regex = re.compile(self.HELP_PATTERNS, re.IGNORECASE)

    # 'should have read the docs first' is hindsight, not a question — the
    # bare 'should' starter must not claim past-regret statements
    _HINDSIGHT = re.compile(r'^(?:should|could|would)\s+(?:have|not\s+have)\b',
                            re.IGNORECASE)

    def is_question(self, sentence: str) -> bool:
        """Determine whether a sentence is a question or help request."""
        sentence = sentence.strip()
        if not sentence:
            return False
        if sentence.endswith('?'):
            # A '?' still needs minimal substance: bare fragments like
            # 'Why?' are context continuations, not standalone questions
            # (extract_questions attaches them to their preceding sentence)
            return len(sentence.split()) >= 2
        if len(sentence.split()) < self.MIN_QUESTION_WORDS:
            return False
        if self._HINDSIGHT.match(sentence):
            return False
        return bool(self.starter_regex.search(sentence) or self.help_regex.search(sentence))

    # Leading greetings carry no meaning and clutter the dashboard
    _GREETING = re.compile(
        r'^(?:(?:hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening)'
        r'|morning|quick\s+(?:one|question)|one\s+more\s+(?:thing|question))'
        r'(?:\s+(?:team|all|everyone|folks|guys|there))?[\s,!.:-]+)+',
        re.IGNORECASE)

    # Leading list markers ('* ', '- ', '3. ', '>') from Slack formatting
    _BULLET = re.compile(r'^(?:[*\-•>]+|\d{1,2}[.)])\s+')

    # 'Forwarding from customer:' / 'FWD:' preambles wrap someone else's ask
    _FORWARD = re.compile(r'^(?:forwarding|forwarded|fwd|fw)\b[^:]{0,60}:\s*',
                          re.IGNORECASE)

    def strip_greeting(self, question: str) -> str:
        """Remove leading greetings, forward preambles, and list markers."""
        stripped = self._GREETING.sub('', question).strip()
        stripped = self._FORWARD.sub('', stripped).strip()
        stripped = self._BULLET.sub('', stripped).strip()
        return stripped if stripped else question

    # Abbreviations whose periods are NOT sentence boundaries — splitting on
    # "e.g." tore "...maintenance window, e.g. 1am to 4am?" into fragments
    _ABBREVIATIONS = re.compile(
        r'\b(?:e\.g\.|i\.e\.|etc\.|vs\.|approx\.|cf\.)', re.IGNORECASE)
    _DOT_SENTINEL = '\x01'
    # A sentence starting like this right after a question is the same ask
    # continued as an alternative, not a new question
    _CONTINUATION_RE = re.compile(r'^(?:or|otherwise|or\s+else)\b',
                                  re.IGNORECASE)

    def extract_questions(self, text: str) -> List[str]:
        """
        Extract questions from text.

        Args:
            text: Raw text from Slack message

        Returns:
            List of extracted question strings
        """
        # Protect abbreviation and decimal periods ("e.g.", "10.15") before
        # sentence splitting; restored after
        protected = self._ABBREVIATIONS.sub(
            lambda m: m.group(0).replace('.', self._DOT_SENTINEL), text)
        protected = re.sub(r'(?<=\d)\.(?=\d)', self._DOT_SENTINEL, protected)

        # Hard-wrapped lines inside one message are ONE sentence: pasted
        # Slack text wraps mid-sentence, and splitting there extracts
        # fragments ("...configure the retry" / "count for outbound
        # transfers?") that then rank — and route — as separate questions.
        # A line joins the previous one when the previous doesn't end a
        # sentence and the next starts lowercase; headers, greetings, and
        # new asks start with capitals, digits, or punctuation, so the
        # header/greeting isolation the newline boundary exists for holds.
        # This runs BEFORE the explicit-marker split below, which creates
        # boundaries of its own.
        protected = re.sub(r'(?<=[^\s.!?:;])[ \t]*\n[ \t]*(?=[a-z])', ' ',
                           protected)

        # An explicit coordination marker joins two DISTINCT asks in one
        # sentence ("does X support TLS, and separately, can it post to
        # Kafka?") — break it so each part is seen as its own candidate
        protected = re.sub(r',?\s+and,?\s+separately,?\s+', '\n', protected,
                           flags=re.IGNORECASE)

        # Split into sentences, keeping the trailing '?' so it can be detected.
        # Newlines are sentence boundaries: a header or greeting on its own
        # line must never get glued onto the next line's question.
        sentences = re.findall(r'[^.!?\n]+[.!?]?', protected)

        questions = []
        prev_was_question = False
        prev_sentence = ''
        for sentence in sentences:
            sentence = sentence.replace(self._DOT_SENTINEL, '.').strip()
            # Strip the greeting BEFORE the question test: 'Quick one - does
            # X support TLS' hides its question word behind the opener, and
            # an unrecognized question here blinds the under-extraction
            # safety net (the counts matched while the QUESTIONS differed)
            candidate = self.strip_greeting(sentence)
            # A quoted/forwarded ask arrives wrapped in quote marks — the
            # marks are packaging, not content
            candidate = candidate.strip('"\'“”‘’ ')
            # "Is X better? or should we use Y?" is ONE ask phrased as
            # alternatives — the '?' split would strand "or should we use
            # Y?" as a second, half-a-thought question
            if prev_was_question and questions \
                    and self._CONTINUATION_RE.match(candidate):
                questions[-1] += ' ' + candidate.rstrip('.!').strip()
                continue
            # A bare '?' fragment ('Why?', 'the proxy?') is a continuation
            # of the PRECEDING sentence — that sentence is the actual ask
            # ('The sync job dies every night. Why?'), never the fragment
            # alone
            if candidate.endswith('?') and len(candidate.split()) < 3:
                if prev_was_question and questions:
                    questions[-1] += ' ' + candidate
                elif prev_sentence:
                    questions.append(f"{prev_sentence} {candidate}")
                    prev_was_question = True
                prev_sentence = candidate
                continue
            if self.is_question(candidate):
                # Drop trailing '.'/'!' but keep '?'
                questions.append(candidate.rstrip('.!').strip())
                prev_was_question = True
            else:
                prev_was_question = False
            prev_sentence = candidate

        return questions

    def normalize_question(self, question: str) -> str:
        """
        Normalize a question for better comparison.

        Args:
            question: Raw question text

        Returns:
            Normalized question text
        """
        # Convert to lowercase
        normalized = question.lower()

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)

        # Remove special characters but keep question marks
        normalized = re.sub(r'[^\w\s?-]', '', normalized)

        # Remove greeting/filler words from the LEADING EDGE only. Mid-text
        # removal deleted content words ('the thanks page' -> 'the page'),
        # and \b treats '-' as a boundary, so 'hi-res' became '-res' — this
        # text is both the dedup key and what gets embedded, so corruption
        # here corrupts grouping itself.
        filler = re.compile(
            r'^(?:(?:hi|hello|hey|team|guys|folks|please|thanks|thank you)'
            r'(?![\w-]))[\s,!.:-]*', re.IGNORECASE)
        while True:
            stripped = filler.sub('', normalized)
            if stripped == normalized:
                break
            normalized = stripped

        # Clean up extra spaces again
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def parse_slack_content(self, content: str) -> List[Dict]:
        """
        Parse Slack content and extract questions with metadata.
        The format (JSON, CSV, or plain text) is detected automatically.

        Args:
            content: Raw Slack content string

        Returns:
            List of dictionaries containing questions and metadata
        """
        return self.questions_from_messages(self.extract_messages(content))

    def extract_messages(self, content: str) -> List[Dict]:
        """
        Split raw content into individual messages, detecting the format.

        Returns a list of {'text', 'date', 'replies'} dicts (replies only
        present for Slack JSON exports with threads).
        """
        stripped = content.lstrip()
        if stripped.startswith('{') or stripped.startswith('['):
            messages = self._messages_from_json(stripped)
            if messages is not None:
                return messages

        messages = self._messages_from_csv(content)
        if messages is not None:
            return messages

        return self._messages_from_text(content)

    # ---- Slack markup ----

    @staticmethod
    def clean_slack_markup(text: str) -> str:
        """Strip Slack-specific markup so it doesn't pollute question grouping."""
        # Fenced code blocks are usually logs/stack traces, not question text
        text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
        text = text.replace('`', '')
        # <@U123> / <@U123|name> user mentions
        text = re.sub(r'<@[A-Z0-9]+(?:\|[^>]*)?>', '', text)
        # <#C123|channel-name> channel links
        text = re.sub(r'<#[A-Z0-9]+\|([^>]*)>', r'#\1', text)
        # <!here>, <!channel>, <!everyone> broadcasts
        text = re.sub(r'<!(?:here|channel|everyone)>', '', text)
        # <http://url|label> -> label, bare <http://url> -> dropped
        text = re.sub(r'<https?://[^|>]+\|([^>]*)>', r'\1', text)
        text = re.sub(r'<https?://[^>]+>', '', text)
        # Naked URLs (plain-text/CSV transcripts): a query string's '?' would
        # otherwise split the sentence and leave a garbage "question" like
        # 'com/search?'. Lazy match + lookahead so SENTENCE punctuation right
        # after the URL survives: "tried https://newtool.io?" must keep its
        # '?' or the question vanishes entirely. The HOSTNAME is kept — it is
        # the question's subject, and deleting it made "tried <toolA>?" and
        # "tried <toolB>?" normalize identically (phantom 'asked 2x' merges).
        text = re.sub(r'\bhttps?://([^/\s?#>]+)\S*?(?=[?.!,;:)\]]*(?:\s|$))',
                      r'\1', text)
        # :emoji_codes:
        text = re.sub(r':[a-z0-9_+\-]+:', '', text)
        # HTML entities Slack escapes
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        return re.sub(r'[ \t]+', ' ', text).strip()

    # ---- Format-specific parsers ----

    @staticmethod
    def _slack_ts_to_date(ts) -> Optional[str]:
        """Convert a Slack epoch timestamp ('1717589200.000200') to YYYY-MM-DD."""
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime('%Y-%m-%d')
        except (TypeError, ValueError, OSError, OverflowError):
            return None

    def _messages_from_json(self, content: str) -> Optional[List[Dict]]:
        """
        Parse a Slack JSON export. Returns None when it isn't one.

        Thread replies (messages whose thread_ts points at another message)
        are attached to their parent as 'replies' instead of being treated
        as standalone messages, enabling answer detection.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None

        if isinstance(data, dict):
            data = data.get('messages')
        if not isinstance(data, list):
            return None

        # Modern Slack exports often leave 'text' empty and put the real
        # content in rich-text 'blocks' — read both
        items = []
        for item in data:
            if not isinstance(item, dict):
                continue
            text = item.get('text')
            if not isinstance(text, str) or not text.strip():
                text = self._text_from_blocks(item.get('blocks'))
            if text and text.strip():
                items.append((text, item))

        # Split into thread parents and replies
        replies_by_parent: Dict[str, List[str]] = {}
        parents = []
        for text, item in items:
            ts = item.get('ts')
            thread_ts = item.get('thread_ts')
            if thread_ts and ts and thread_ts != ts:
                replies_by_parent.setdefault(thread_ts, []).append(
                    self.clean_slack_markup(text)[:300])
            else:
                parents.append((text, item))

        messages = []
        for text, item in parents:
            date = item.get('date') or self._slack_ts_to_date(item.get('ts'))
            message = {'text': text, 'date': date}
            replies = replies_by_parent.get(item.get('ts'), [])
            if replies:
                message['replies'] = replies[:5]
            messages.append(message)
        return messages

    @staticmethod
    def _text_from_blocks(blocks) -> str:
        """
        Extract plain text from Slack rich-text 'blocks'. Preformatted blocks
        (code/logs) are skipped, matching how fenced code is stripped from
        plain text.

        Emoji elements are kept as :shortcodes: and list items become
        '* '-prefixed lines: the announcement detector
        (textutil.looks_like_announcement) reads emoji density and bullet
        structure as broadcast signals, and flattening them away here made
        blocks-sourced announcements invisible to it.
        """
        if not isinstance(blocks, list):
            return ''
        parts = []

        def walk(elements, in_list=False):
            for element in elements or []:
                if not isinstance(element, dict):
                    continue
                etype = element.get('type')
                if etype == 'rich_text_preformatted':
                    continue
                if etype == 'text' and isinstance(element.get('text'), str):
                    parts.append(element['text'])
                elif etype == 'link':
                    parts.append(element.get('text') or '')
                elif etype == 'emoji' and element.get('name'):
                    parts.append(f":{element['name']}:")
                elif etype == 'rich_text_list':
                    for item in element.get('elements') or []:
                        parts.append('\n* ')
                        walk([item] if isinstance(item, dict) else [],
                             in_list=True)
                    parts.append('\n')
                elif etype in ('rich_text_section', 'rich_text_quote',
                               'rich_text'):
                    if not in_list and parts and etype == 'rich_text_section':
                        parts.append('\n')
                    walk(element.get('elements'), in_list=in_list)

        for block in blocks:
            if isinstance(block, dict) and block.get('type') == 'rich_text':
                walk(block.get('elements'))
        return ''.join(parts).strip()

    _CSV_TEXT_COLUMNS = ('text', 'message', 'question', 'content', 'body')
    _CSV_DATE_COLUMNS = ('date', 'ts', 'timestamp', 'time', 'datetime')

    def _messages_from_csv(self, content: str) -> Optional[List[Dict]]:
        """Parse CSV with a recognizable text column. Returns None otherwise."""
        first_line = content.lstrip().split('\n', 1)[0]
        if ',' not in first_line:
            return None

        try:
            reader = csv.DictReader(io.StringIO(content.lstrip()))
            headers = {h.strip().lower(): h for h in (reader.fieldnames or [])}
        except csv.Error:
            return None

        text_col = next((headers[c] for c in self._CSV_TEXT_COLUMNS if c in headers), None)
        if text_col is None:
            return None
        date_col = next((headers[c] for c in self._CSV_DATE_COLUMNS if c in headers), None)

        messages = []
        try:
            for row in reader:
                text = (row.get(text_col) or '').strip()
                if not text:
                    continue
                date = (row.get(date_col) or '').strip() if date_col else None
                # Numeric timestamps (Slack ts) get converted; date strings pass through
                if date and re.fullmatch(r'\d{9,}(\.\d+)?', date):
                    date = self._slack_ts_to_date(date)
                messages.append({'text': text, 'date': date or None})
        except csv.Error:
            return None
        return messages

    def questions_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """Extract questions from structured {'text', 'date', 'replies'} messages."""
        parsed_questions = []
        for message in messages:
            # Keep newlines: they are sentence boundaries (headers/greetings on
            # their own line must not merge into the next line's question)
            text = self.clean_slack_markup(message['text'])
            for question in self.extract_questions(text):
                parsed = {
                    'text': question,
                    'normalized_text': self.normalize_question(question),
                    'date': message.get('date') or 'Unknown',
                    'original_message': canonical_source(text)
                }
                if message.get('replies'):
                    parsed['replies'] = message['replies']
                parsed_questions.append(parsed)
        return parsed_questions

    def _messages_from_text(self, content: str) -> List[Dict]:
        """
        Parse plain text with dashed separator lines between messages.

        Lines quoted with '>' (the Slack/markdown reply convention) are
        thread replies: they attach to the message as 'replies' for answer
        detection and are excluded from question extraction — a responder's
        "Are they failing at the same time?" is not the asker's question.
        Consecutive quoted lines form one reply; a blank line starts a new
        one. '#'-heading lines are structural markup, not message content
        (same rationale as stripping fenced code blocks).
        """
        # Split by separator line (a run of 10+ dashes on its own line)
        blocks = re.split(r'\n-{10,}\n?|^-{10,}\n', content)

        messages = []

        for block_index, block in enumerate(blocks):
            block = block.strip()
            if not block:
                continue

            # Extract date (first line typically)
            lines = block.split('\n')
            date = None
            text_lines = []
            replies = []
            in_reply = False
            pre_date_lines = 0  # title furniture before the FIRST date line

            for line in lines:
                line = line.strip()
                if not line:
                    in_reply = False
                    continue

                if line.startswith('>'):
                    quoted = line.lstrip('> ').strip()
                    if quoted:
                        if in_reply and replies:
                            replies[-1] = (replies[-1] + ' ' + quoted)[:300]
                        else:
                            replies.append(quoted[:300])
                    in_reply = True
                    continue
                in_reply = False

                if re.match(r'#{1,6}\s', line):
                    continue

                # Take the first date found; a line that is ONLY a date is
                # consumed, but a line with a date AND text keeps its text
                if not date:
                    found, pure_date_line = self._extract_date(line)
                    if found:
                        date = found
                        if pure_date_line:
                            continue
                    else:
                        pre_date_lines = len(text_lines) + 1
                text_lines.append(line)

            # A document title before the FIRST block's date line is file
            # furniture, not message content (same class as '# ' comments).
            # Glued onto the first message it inflates the source past its
            # identity cap and can fake enumeration ('...test set 2)').
            if block_index == 0 and date and pre_date_lines:
                text_lines = text_lines[pre_date_lines:]

            if text_lines:
                # Newline-join: each source line stays its own sentence
                message = {'text': '\n'.join(text_lines), 'date': date}
                if replies:
                    message['replies'] = replies[:5]
                messages.append(message)

        return messages

    # Shared with weekly_stats' parser via textutil: recognition and
    # parsing must accept the same shapes
    _DATE_PATTERNS = DATE_PATTERNS

    def _extract_date(self, line: str):
        """
        Find a date in a line.

        Returns (date_string, is_pure_date_line): is_pure_date_line is True
        when the line contains nothing meaningful besides the date.
        """
        for pattern in self._DATE_PATTERNS:
            match = re.search(pattern, line)
            if match:
                rest = line[:match.start()] + line[match.end():]
                rest_words = re.findall(r'[A-Za-z0-9]+', rest)
                return match.group(0), len(rest_words) < 3
        return None, False
