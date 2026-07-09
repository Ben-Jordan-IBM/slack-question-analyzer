"""
Shared text primitives with load-bearing semantics.

These existed as near-identical copies across modules; a copy drifting is
how the round-14 header bug happened (a source string capped differently
in two places stops being the same identity key). One definition each:

- SOURCE_KEY_LEN / canonical_source: the cap that makes a cleaned message
  text usable as an identity key ('original_message'). Every producer and
  every comparer must use the same cap or same-message logic silently
  breaks.
- stem / stem_tokens: the crude suffix folding used for content-word
  overlap tests. It is deliberately crude (fast, no dependencies) and
  ASYMMETRIC ('failures' -> 'failur' but 'failure' -> 'failure'), which is
  why comparisons must go through prefix_match.
- prefix_match: stem equality with prefix tolerance in both directions.
"""

import re

# A cleaned message is capped at this length to become an identity key.
# Sources longer than this are TRUNCATED — same-message passes treat a
# len >= SOURCE_KEY_LEN source as "possibly not the whole message" and
# stand down, and source-support checks can't vouch for asks extracted
# past the cap. 600 matches the per-message view the LLM extraction
# prompt gets; the cap exists only to keep pathological pastes (logs,
# stack traces) from becoming multi-KB identity keys.
SOURCE_KEY_LEN = 600


def canonical_source(text: str) -> str:
    """Whitespace-collapsed, capped message text — the identity key form."""
    return ' '.join((text or '').split())[:SOURCE_KEY_LEN]


def stem(token: str) -> str:
    """Crude suffix folding: 'failed'/'failing'/'failures' fold toward a
    shared stem so content-word overlap survives inflection."""
    for suffix in ('ing', 'ed', 'es', 's'):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:len(token) - len(suffix)]
    return token


def stem_tokens(text: str, min_len: int = 4) -> set:
    """Stemmed content tokens of a text (tokens shorter than min_len are
    treated as non-content)."""
    return {stem(t) for t in re.findall(r'[a-z0-9]+', (text or '').lower())
            if len(t) >= min_len}


def prefix_match(a: str, b: str) -> bool:
    """Stems match when equal or one is a prefix of the other — the crude
    folding is asymmetric, so a one-sided comparison drops real matches."""
    return a == b or a.startswith(b) or b.startswith(a)


# --- Announcement / notice detection ---------------------------------------
# Broadcast posts (webinar promos, sales-kit launches, win wires) and
# "please note" channel notices are CONTEXT, not questions. Left in the
# extraction input they are the #1 phantom source: rhetorical headers
# ("Why should customers care?"), schedule lines ("When: July 8"), and
# marketing copy get inverted into "questions" nobody asked — and their
# vocabulary bleeds into rewrites of real questions in the same batch.
#
# Detection is deliberately precision-first: a genuine ask slipping
# through costs nothing (status quo), but a real question misread as an
# announcement would be silently lost. Hence the three-part shape:
#   1. a HELP-SEEKING VETO: any first-person ask phrasing disqualifies
#      the message outright, no matter how emoji-heavy it is
#   2. marketing announcements need at least one CONTENT signal (call to
#      action, marketing phrasing, cc:-broadcast) — structural signals
#      alone (emoji, bullet lists) never suffice, because real
#      troubleshooting posts use both
#   3. notices need a leading please-directive AND zero question marks

_EMOJI_SHORTCODE_RE = re.compile(r':[a-z0-9_+-]{2,}:')
_BULLET_LINE_RE = re.compile(r'^\s*\*\s+\S', re.MULTILINE)
_CTA_RE = re.compile(
    r'register here|sign ?up|save the date|join us|rsvp'
    r'|explore the|check out|learn more|happy selling'
    r'|spread the word|stay tuned|mark your calendars?', re.IGNORECASE)
_MARKETING_RE = re.compile(
    r"we'?re (?:excited|thrilled|proud)|excited to (?:share|announce)"
    r'|new & improved|is here!|win wire|thank you @'
    r'|this session will|this is just the start'
    r'|make (?:this|every) .{0,40}count', re.IGNORECASE)
_BROADCAST_CC_RE = re.compile(r'\bcc:? ?@', re.IGNORECASE)
_HELP_SEEKING_RE = re.compile(
    r'\b(?:can|could|may|would|should|does|do|did|is|are)\s+'
    r'(?:i|we|you|someone|anyone|anybody)\b'
    r'|\bdoes anyone\b|\banyone (?:have|know|else|here|help)\b'
    r'|\bany (?:ideas?|thoughts?|pointers?|suggestions?|luck)\b'
    r'|\bwe have (?:a )?question\b'
    r'|\bcustomer (?:is asking|would like|wants?|needs?)\b'
    # 'please help spread the word' is broadcast language, not a plea —
    # the lookahead keeps promotion verbs from vetoing real announcements
    r'|\bplease (?:advise|assist|suggest|look into|confirm|check)\b'
    r'|\bplease help(?! (?:spread|share|celebrate|welcome|promote|make))\b'
    r'|\blooking for (?:some )?(?:guidance|help|input|advice)\b'
    r'|\bhow (?:do|can|to|are|does|would)\b|\bwhere (?:can|do|is|are)\b'
    r'|\bseek(?:ing)? (?:for )?help\b', re.IGNORECASE)
_NOTICE_RE = re.compile(
    r'\s*(?:(?:hi|hello|hey|team|all|everyone|folks)[\s,!:-]+)*'
    r'please\s+(?:use|see|refer|review|note|find|bookmark|be aware)\b',
    re.IGNORECASE)


def looks_like_announcement(text: str) -> bool:
    """True when a message is a broadcast announcement or channel notice —
    context for other messages, never a source of questions itself."""
    t = text or ''
    if len(t.strip()) < 40:
        return False
    if _HELP_SEEKING_RE.search(t):
        return False
    content = (bool(_CTA_RE.search(t)) + bool(_MARKETING_RE.search(t))
               + bool(_BROADCAST_CC_RE.search(t)))
    structural = ((len(_EMOJI_SHORTCODE_RE.findall(t)) >= 3)
                  + (len(_BULLET_LINE_RE.findall(t)) >= 3))
    if content >= 1 and content + structural >= 2:
        return True
    # "Please use/see/note..." with no question mark anywhere: a resource
    # notice. Statements in it must not be inverted into questions.
    return '?' not in t and bool(_NOTICE_RE.match(t))


# --- Status/action-request detection ----------------------------------------
# 'Can someone check on https://prod537147...' asks a PERSON to act on an
# identifier, not a question any category answers. Same precision-first
# design as the announcement gate: three conjunctive conditions, so a real
# knowledge question ('Can someone check whether X supports Y?') passes
# through untouched — it has content words and no bare identifier target.

_STATUS_PHRASE_RE = re.compile(
    r'\b(?:can|could) (?:someone|somebody|anyone|anybody|you)\s+'
    r'(?:please\s+)?(?:check|look|take a look|follow up)\b'
    r'|\bplease (?:check|look)(?: at| into| on)?\b'
    r"|\bwhat(?: is|'s) the status of\b"
    r'|\bany updates? on\b|\bstatus of\b', re.IGNORECASE)
_STATUS_TARGET_RE = re.compile(
    r'https?://\S+'                    # a URL
    r'|\b[a-z]+\d+(?:[.-]\w+)+\b'      # host-like: prod537147.a-vir-s100...
    r'|\b[A-Za-z]{2,6}-\d{3,}\b'       # ticket: MAT-26382, INC-4821
    r'|\b[A-Z]{2,4}\d{6,}\b')          # case id: TS022317449
_STATUS_FILLER_WORDS = frozenset(
    'please this that someone somebody anyone anybody could check status'
    ' update updates thanks thank team hello there'.split())


def looks_like_status_request(text: str) -> bool:
    """True when a question is 'please act on this identifier' — a status
    or action request that belongs in the review pile, not a bucket."""
    t = text or ''
    if not (_STATUS_PHRASE_RE.search(t) and _STATUS_TARGET_RE.search(t)):
        return False
    # Strip the target and the request phrasing; if almost no content
    # remains, there is no knowledge question here
    remainder = _STATUS_PHRASE_RE.sub(' ', _STATUS_TARGET_RE.sub(' ', t))
    content = [w for w in re.findall(r'[a-z]{4,}', remainder.lower())
               if w not in _STATUS_FILLER_WORDS]
    return len(content) <= 3


# --- Reply ranking for FAQ material ------------------------------------------
# When a thread ends with "thanks, that worked!", the reply BEFORE it is
# almost certainly the fix — the strongest deterministic signal of which
# reply actually answered. Gratitude messages themselves are confirmations,
# not answers.

_GRATITUDE_RE = re.compile(
    r'\b(?:thanks|thank you|thx|that work(?:s|ed)|worked (?:perfectly|great'
    r'|fine)|perfect|resolved|fixed (?:it|the issue)|got it|solved)\b',
    re.IGNORECASE)


def rank_replies(replies) -> list:
    """Replies ordered by answer-likelihood: confirmed-by-gratitude first,
    then substantial replies, with bare gratitude notes last. Stable within
    a score so thread order still breaks ties."""
    scored = []
    for i, reply in enumerate(replies or []):
        text = (reply or '').strip()
        if not text:
            continue
        following = (replies[i + 1] or '') if i + 1 < len(replies) else ''
        score = 0
        if _GRATITUDE_RE.search(following):
            score += 2          # the next person confirmed this one worked
        if len(text.split()) >= 8:
            score += 1          # substance beats "try restarting?"
        if _GRATITUDE_RE.search(text) and len(text.split()) <= 6:
            score -= 3          # a bare "thanks, worked!" is not an answer
        scored.append((-score, i, text))
    return [text for _, _, text in sorted(scored)]


# The three date shapes the pipeline understands. RECOGNITION (finding a
# date in a line — question_extractor) and PARSING (turning it into a
# datetime.date — weekly_stats) must accept the SAME shapes: adding a
# format to one side and not the other makes dates silently invisible to
# half the pipeline.
DATE_NUMERIC_YMD = r'\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b'
DATE_NUMERIC_MDY = r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b'
# Anchored to REAL month names: any 3-9 letter word would recognize
# 'Monday 3 2024' as a date that parse_question_date then rejects,
# breaking the 'every recognized date is also parseable' invariant
DATE_MONTH_NAME = (r'(?i)\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|'
                   r'nov|dec)[a-z]*)\.?\s+(\d{1,2}),?\s+(\d{4})\b')
DATE_PATTERNS = (DATE_NUMERIC_YMD, DATE_MONTH_NAME, DATE_NUMERIC_MDY)
