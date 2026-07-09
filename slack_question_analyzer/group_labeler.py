"""
LLM prompting layer for the analysis pipeline.

All generation goes through one chat-based core with:
- JSON-schema-enforced output (Ollama `format` schema)
- temperature 0 and a fixed seed for reproducibility
- keep_alive so Ollama keeps the model loaded across sequential calls
- semantic validation with one corrective retry

Capabilities (all optional — the pipeline works without any of them):
- extract_questions:      LLM-first question extraction & standalone rewriting
- detect_questions:       find help requests the regex extractor missed
- consolidate_same_ask:   within-message restatement consolidation (two-judge)
- confirm_feature_request: support-vs-feedback second opinion
- verify_same_topic:      yes/no check for borderline group merges
- audit_group:            evict members that don't belong (two-judge)
- choose_bucket:          closed-choice routing adjudication (0 = abstain)
- label_group:            topic name + one-sentence summary per group
- assign_themes:          roll topics up into broad dashboard themes
- summarize_analysis:     executive summary of the whole analysis
- is_answered:            decide whether thread replies resolved a question
"""

import os
import json
import re
import time
import logging
import threading
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

from .disk_cache import JsonDiskCache
from .model_defaults import default_generation_model, FALLBACK_GENERATION_MODEL
from .ollama_http import raise_with_detail

logger = logging.getLogger(__name__)

MAX_QUESTIONS_IN_PROMPT = 8
MAX_TOPIC_WORDS = 6

# The second axis: WHAT KIND of question, independent of subject. A
# feature request and a defect report about the same feature need
# completely different handling.
QUESTION_TYPES = {'how-to', 'troubleshooting', 'is-it-possible',
                  'feature-request', 'defect-report'}

# Topics made up entirely of these words are useless and get retried
GENERIC_TOPIC_WORDS = {
    'general', 'misc', 'miscellaneous', 'various', 'questions', 'question',
    'help', 'issues', 'issue', 'stuff', 'other', 'others', 'support',
    'technical', 'assorted', 'random', 'topics', 'topic', 'inquiries', 'asks',
}

LABEL_SCHEMA = {
    'type': 'object',
    'properties': {
        'topic': {'type': 'string'},
        'summary': {'type': 'string'},
    },
    'required': ['topic', 'summary'],
}

VERIFY_SCHEMA = {
    'type': 'object',
    'properties': {'same_topic': {'type': 'boolean'}},
    'required': ['same_topic'],
}

SAME_ASK_SCHEMA = {
    'type': 'object',
    'properties': {'same_ask': {'type': 'boolean'}},
    'required': ['same_ask'],
}

AUDIT_SCHEMA = {
    'type': 'object',
    'properties': {
        'outliers': {'type': 'array', 'items': {'type': 'integer'}},
    },
    'required': ['outliers'],
}

THEMES_SCHEMA = {
    'type': 'object',
    'properties': {
        'themes': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'items': {'type': 'array', 'items': {'type': 'integer'}},
                },
                'required': ['name', 'items'],
            },
        },
    },
    'required': ['themes'],
}

SUMMARY_SCHEMA = {
    'type': 'object',
    'properties': {'summary': {'type': 'string'}},
    'required': ['summary'],
}

FAQ_DRAFT_SCHEMA = {
    'type': 'object',
    'properties': {'answer': {'type': 'string'}},
    'required': ['answer'],
}

# The FAQ draft is SUMMARIZATION WITH RECEIPTS, never answering from the
# model's own weights: a general 8B's beliefs about a specific product are
# exactly the confidently-wrong content the pipeline exists to keep out.
# The model contributes sentence structure; every fact must come from the
# replies, and a deterministic grounding check (analyzer._answer_grounded)
# rejects drafts that smuggle anything in.
FAQ_DRAFT_SYSTEM = (
    "If the replies are empty or do not actually answer the question, "
    "return {\"answer\": \"\"}. Do not guess.\n\n"
    "You condense a support thread's replies into a short draft FAQ "
    "answer.\n"
    "HARD RULE: use ONLY facts that appear in the replies. Every setting "
    "name, number, file path, command, and product term in your answer "
    "must be copied from the replies. You contribute wording and "
    "structure, never information. Do not add caveats, steps, or defaults "
    "the replies don't state.\n"
    "Write 2-4 direct sentences, imperative where natural ('Set X to "
    "Y.'). No greetings, no 'based on the replies', no meta-commentary.\n\n"
    "Respond with JSON only: {\"answer\": \"<draft>\"}"
)

DETECT_SCHEMA = {
    'type': 'object',
    'properties': {
        'questions': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'index': {'type': 'integer'},
                    'question': {'type': 'string'},
                    'type': {'type': 'string',
                             'enum': ['how-to', 'troubleshooting',
                                      'is-it-possible', 'feature-request',
                                      'defect-report']},
                },
                'required': ['index', 'question'],
            },
        },
    },
    'required': ['questions'],
}

ANSWERED_SCHEMA = {
    'type': 'object',
    'properties': {'verdict': {'type': 'string',
                               'enum': ['answered', 'unanswered', 'unknown']}},
    'required': ['verdict'],
}

CONSOLIDATE_SCHEMA = {
    'type': 'object',
    'properties': {'keep': {'type': 'array', 'items': {'type': 'integer'}}},
    'required': ['keep'],
}

# Lexical overlap catches near-verbatim restatement; this catches REPHRASED
# restatement within one message ("wrong timezone after DST?" / "is the
# DST-stopping issue timezone-related?" = one ask, two sentences)
CONSOLIDATE_SYSTEM = (
    "If the input is empty or malformed, or you are unsure, return every "
    "question number. Do not guess.\n\n"
    "All of these questions were extracted from ONE message. Questions that "
    "restate the same ask from a different angle, or break one goal into "
    "steps, are ONE ask. The test: would one answer resolve both?\n"
    "Restatement cues: in the original message, a sentence introduced by "
    "'Basically', 'I mean', 'in other words', 'so', or 'that is' usually "
    "REWORDS the previous ask — same ask. A forwarded quote plus the "
    "forwarder's own paraphrase of it is also ONE ask. Stating the problem "
    "('we don't want X to happen') and then asking how to achieve that "
    "outcome is ONE ask, not two.\n"
    "But two questions wanting DIFFERENT outcomes are distinct even when "
    "they share a feature area or phrasing template — a retention policy "
    "and an auto-delete schedule need different answers.\n"
    "Output the numbers to KEEP — exactly one per distinct ask, keeping the "
    "most complete phrasing. If every question is a distinct ask, return "
    "every number.\n\n"
    "Example: 1. Could the scheduler be running in the wrong timezone after "
    "DST? 2. Is the transfers-stopping-after-DST issue timezone-related? "
    "One answer resolves both: {\"keep\": [1]}\n"
    "Example: 1. Can we prevent the import from hard-failing on a bad row? "
    "2. How can we handle a bad row gracefully and divert it for review? "
    "Same outcome wanted (don't fail outright), one answer resolves both: "
    "{\"keep\": [2]}\n"
    "Example: 1. How do I increase the API request timeout? 2. Why do "
    "large uploads time out before finishing? The second restates the "
    "first's SYMPTOM from the message's context — one answer (raise the "
    "timeout) resolves both: {\"keep\": [1]}\n"
    "Example: 1. Is there a maximum attachment size? 2. Can the system "
    "handle very large attachments? The same ask from both directions — "
    "one answer (the size limit) resolves both: {\"keep\": [1]}\n"
    "Example: 1. Does the platform have an API for creating exports? "
    "2. Any docs link would help? The second is a trailing CONTINUATION "
    "leaning on the first — the answer to the first (the API docs link) IS "
    "the answer to the second. Short deictic follow-ups ('is that possible "
    "here?', 'so it just needs configuring?', 'all in one place?') that "
    "cannot stand alone are the same ask: {\"keep\": [1]}\n"
    "Example: 1. What alternative solutions exist for integrating with an "
    "external vault? 2. Can agent-based integration be leveraged here? The "
    "second proposes ONE CANDIDATE inside the first's request for "
    "alternatives — the first's answer (the list of options, including "
    "whether that candidate works) resolves both: {\"keep\": [1]}\n"
    "Example: 1. Can we trigger transfers via REST? 2. Is there a way to "
    "bulk-disable actions? Different asks: {\"keep\": [1, 2]}\n\n"
    "Respond with JSON only: {\"keep\": [<numbers>]}"
)

FEEDBACK_SCHEMA = {
    'type': 'object',
    'properties': {'feature_request': {'type': 'boolean'}},
    'required': ['feature_request'],
}

# The extractor's type tags are a noisy 3B signal; nothing leaves the
# support funnel without this closed second opinion from the quality model
FEEDBACK_SYSTEM = (
    "If the question is empty or malformed, answer "
    "{\"feature_request\": false}. Do not guess.\n\n"
    "You decide whether a question is PRODUCT FEEDBACK or a SUPPORT "
    "question.\n"
    "feature_request is true ONLY if the asker wants a capability the "
    "product does not currently have — a new feature or enhancement "
    "('it would be great if...', 'can you add...', 'the product should...').\n"
    "Asking HOW to do something, WHETHER something is possible today, or WHY "
    "something is broken is SUPPORT — even if the answer turns out to be "
    "'not supported'. A capability question ('can we cap transfers per "
    "node?') is SUPPORT, not a feature request, unless the asker says the "
    "capability is missing and wished for. When in doubt, answer false.\n"
    "Judge the asker's INTENT from the original message when it is shown: "
    "wish-phrasing ('would be great', 'would love', 'please add', 'any plans "
    "to', 'doesn't exist today') signals feedback even when the rewritten "
    "question sounds like a support ask.\n"
    "Respond with JSON only: {\"feature_request\": true} or "
    "{\"feature_request\": false}"
)

# Prompt pack version: stamped into results metadata so drift is traceable
# (the LLM cache keys on full prompt text, so bumps also invalidate caches)
PROMPT_PACK_VERSION = 25

LABEL_SYSTEM = (
    "If the group is empty, malformed, or too mixed to share one honest "
    "label, respond {\"topic\": \"NEEDS_REVIEW\", \"summary\": \"\"} and "
    "nothing else. Do not guess.\n\n"
    "You name a group of related support questions with a short topic label.\n"
    "Rules:\n"
    "- Topic: 2-4 words, Title Case, using words that ACTUALLY APPEAR in the "
    "questions — do not introduce new vocabulary.\n"
    "- The topic must name the SPECIFIC feature, system, or task being asked about.\n"
    "- Never use vague topics like 'General Questions', 'Help', 'Miscellaneous', "
    "'Various Issues', or 'Technical Questions'.\n"
    "- Never use the bare product name alone as the topic — name the specific "
    "capability inside it.\n"
    "- Never start the topic with 'How to' — it is a category name, not a question.\n"
    "- Keep exact product and feature terms from the questions (don't paraphrase "
    "technical names).\n"
    "- Summary: one sentence describing what people are asking.\n"
    "- Respond with JSON only."
)

LABEL_FEW_SHOT = """Example:
Questions:
- Copy Task failing with "Exception while scanning for virus" — please advise.
- When a virus is detected, how can we send an email notification to an admin?
Answer: {"topic": "Antivirus Scanning", "summary": "Users ask how to configure virus scanning and handle scan failures."}

Example:
Questions:
- Is there a REST API to deactivate a list of Scheduled Actions?
- Can we trigger a file transfer via a REST API call instead of a scheduled action?
Answer: {"topic": "Scheduled Action APIs", "summary": "Users want REST APIs to trigger or manage scheduled actions instead of using the scheduler."}

Example:
Questions:
- Why won't the MFT UI open after upgrading to v12?
- Debug log shows NullPointerException while fetching UI settings — anyone seen this?
Answer: {"topic": "MFT UI Errors", "summary": "Users hit internal errors and exceptions opening the MFT UI after upgrades."}

Now label this group."""

VERIFY_SYSTEM = (
    "If either group is empty or malformed, answer {\"same_topic\": false} "
    "and nothing else. Do not guess.\n\n"
    "You decide whether two groups of support questions are about the SAME "
    "topic.\n"
    "The test: would ONE ANSWER — the same concrete instructions or the same "
    "single fix — fully resolve every question in both groups? If yes -> "
    "same topic. If each group needs its own answer -> different topics, "
    "even when both answers would live on the same documentation page.\n"
    "Same product, same feature area, or shared vocabulary do NOT make two "
    "groups the same topic — only a shared answer does. When in doubt, "
    "answer false.\n\n"
    "Example: Group A asks how to get notified when a backup fails; Group B "
    "asks what setting controls how long a backup may run before timing "
    "out. Both mention backups failing, but the answers are different "
    "(alert configuration vs a timeout setting): {\"same_topic\": false}\n"
    "Example: Group A asks how the metering agent gets installed; Group B asks "
    "how to set up monitoring alerts. Same product, different root causes: "
    "{\"same_topic\": false}\n"
    "Example: Group A asks where quarantined files go; Group B asks what happens "
    "to a file when its virus scan fails. One antivirus-handling page covers "
    "both: {\"same_topic\": true}\n"
    "Example: Group A asks the quickest way to onboard a new employee end to "
    "end; Group B asks for a streamlined process to set up a new hire from "
    "scratch. The SAME end-to-end goal in different words — one page: "
    "{\"same_topic\": true}\n"
    "Example: Group A reports that uploads keep getting cut off on slow "
    "connections; Group B asks which setting raises the upload wait limit. "
    "The SYMPTOM and ITS FIX: the one answer (raise that setting) fully "
    "resolves both: {\"same_topic\": true}\n"
    "Example: Group A asks why Azure container-level token auth fails; Group B "
    "asks why SFTP key authentication fails. Both are authentication, but "
    "different protocols, different root causes, different fixes: "
    "{\"same_topic\": false}\n"
    "Example: Group A asks about triggering transfers via REST API; Group B asks "
    "about scheduling recurring transfers: {\"same_topic\": false}\n"
    "Example: Group A asks how to set up a new user account end to end; Group B "
    "asks how to enforce password rules on accounts. Setting something up and "
    "configuring ONE property of it are different workflow stages with "
    "different doc pages: {\"same_topic\": false}\n"
    "Example: Group A asks how to rotate or renew their own credentials; "
    "Group B asks how to verify the other party's identity before "
    "connecting. Credential lifecycle vs identity verification — related "
    "security tasks, different doc pages: {\"same_topic\": false}\n"
    "Example: Group A asks how to renew a TLS certificate; Group B asks how "
    "to renew an API token. The SAME action on DIFFERENT objects is a "
    "different topic — each renewal has its own procedure and page: "
    "{\"same_topic\": false}\n"
    "Example: Group A asks how to merge the content of two files; Group B asks "
    "whether a control file can trigger a transfer. Both mention files, but "
    "merging contents and trigger-on-control-file are different operations "
    "with different doc pages: {\"same_topic\": false}\n"
    "Example: Group A asks whether a specific credential type is supported; "
    "Group B asks for alternative ways to integrate with an external secrets "
    "vault. Both mention the same cloud deployment, but a supported-feature "
    "check and an integration workaround need different answers: "
    "{\"same_topic\": false}\n\n"
    "Respond with JSON only."
)

# The audit has the OPPOSITE bias to VERIFY_SYSTEM: these questions were
# already matched, so evicting needs confidence, keeping doesn't. Same
# yardstick as merging, opposite tie-breaker — that's what keeps the two
# passes from fighting each other.
AUDIT_SYSTEM = (
    "If the group is empty or malformed, return {\"outliers\": []} and "
    "nothing else. Do not guess.\n\n"
    "You quality-check a group of support questions that were matched as one "
    "topic, and flag any that do not belong.\n"
    "Apply the same test as merging: would ONE documentation page, fixing ONE "
    "root cause, resolve the whole group? List the numbers of questions that "
    "fall OUTSIDE that page.\n"
    "A question is an outlier only if it is about a DIFFERENT subject "
    "(different feature, different root cause) than the rest. Differences in "
    "wording, phrasing, or angle do NOT make a question an outlier.\n"
    "If every question belongs, or you are unsure, return an empty list.\n\n"
    "Example: 1. How do I install the metering agent? 2. How do I set up "
    "monitoring alerts? 3. Can monitoring alert on one application? "
    "Question 1 is about metering, the rest about monitoring/alerting: "
    "{\"outliers\": [1]}\n"
    "Example: 1. Any good examples of using e2e monitoring? 2. How are real "
    "clients using monitoring/alerting? Same subject, different wording: "
    "{\"outliers\": []}\n"
    "Example: 1. What is the best way to merge the content of two files? "
    "2. Can a control file trigger a transfer when it appears? Both mention "
    "files, but merging contents and trigger-on-control-file are different "
    "operations with different doc pages: {\"outliers\": [2]}\n\n"
    "Respond with JSON only."
)

ROUTE_SCHEMA = {
    'type': 'object',
    'properties': {'category': {'type': 'integer'}},
    'required': ['category'],
}

# The simplest possible task for a small model: a closed set, one item,
# one number, no inventing. Used only for questions whose top-2 anchor
# similarities are too close for embeddings to decide alone.
ROUTE_SYSTEM = (
    "If the question is empty, malformed, or you cannot decide with "
    "confidence, reply {\"category\": 0}. Do not guess.\n\n"
    "You sort ONE support question into exactly one category. Respond with "
    "the category number.\n"
    "These categories cover ONE specific product's support questions. A "
    "question about anything else — another tool or website, company "
    "logistics, vendor support hours, IT chat — is 0, even when a category "
    "NAME sounds related ('is the wiki down?' is not a product connectivity "
    "question).\n"
    "If the question fits two categories almost equally, fits none of them "
    "well, or is too vague to have a subject at all ('is there a limit?'), "
    "reply 0.\n"
    "Example: given file-transfer product categories, 'Is the Jenkins "
    "server down?' is about a DIFFERENT tool — {\"category\": 0}, even "
    "though a connectivity-sounding category exists.\n"
    "A question about a product capability AREA none of the categories "
    "describe (a brand-new kind of feature the list predates) is also 0 — "
    "never file it under the nearest operational category just because it "
    "mentions transfers or files. A growing 0-pile tells the team a "
    "category is missing; a forced fit hides that signal.\n"
    "Example: 'Can the AI assistant build a whole workflow from a "
    "plain-English description?' where no category mentions AI or "
    "assistants — the SUBJECT is the assistant, not the workflow: "
    "{\"category\": 0}.\n"
    "A STATUS or ACTION request asks a PERSON to act, not a question a "
    "category answers. Example: 'Can someone check on ticket INC-4821?' "
    "or 'please look at https://prod9.example.com' — there is nothing to "
    "file: {\"category\": 0}.\n"
    "Never invent a number. Never explain.\n"
    "Respond with JSON only: {\"category\": <number>}"
)

THEMES_SYSTEM = (
    "If the list is empty or malformed, return {\"themes\": []} and nothing "
    "else. Do not guess.\n\n"
    "You organize support-question topics into broad themes for an executive "
    "funnel view. Produce 2 to 6 themes (fewer is better) with short names "
    "(1-3 words, Title Case, drawn from the items' own vocabulary). Assign "
    "EVERY numbered item to exactly one theme by its number. Group by shared "
    "subject area (would the same team own them?), not surface word overlap. "
    "Do not invent themes for single stragglers if a broader existing theme "
    "fits.\n"
    "Respond with JSON only: {\"themes\": [{\"name\": \"...\", "
    "\"items\": [1, 4, 7]}]}"
)

SUMMARY_SYSTEM = (
    "If the topic list is empty or malformed, or you cannot summarize "
    "faithfully using only the listed topics and their exact counts, respond "
    "{\"summary\": \"NEEDS_REVIEW\"} and nothing else. Do not guess.\n\n"
    "You write a brief executive summary of support-question analytics for a team lead. "
    "1-2 sentences. No filler, no preamble, no advice.\n"
    "Rules:\n"
    "- Themes and question groups are DIFFERENT levels. The first sentence "
    "names THEMES ONLY, with their exact counts, largest first — never mix a "
    "question group into the theme list.\n"
    "- A recurring question group may get a SECOND sentence, introduced as "
    "'the most repeated question' — never presented as a theme.\n"
    "- Say the questions are 'evenly spread' ONLY if every theme count is "
    "within 1 of every other theme count. If the largest theme has 2 or "
    "more questions over the smallest, name the largest first instead.\n"
    "- Mention ONLY themes and topics that appear in the lists. NEVER invent "
    "one, even to fill out a sentence.\n"
    "- Use the exact counts as listed. Do not estimate, round, infer, or "
    "drop a theme's count. The total question count is NOT a theme's count.\n"
    "Example input: themes 'File Operations - 6, Performance - 6, "
    "Licensing - 3, Platform - 2' and top group 'Thread Scaling - 2'. "
    "Example output: {\"summary\": \"File Operations and Performance lead "
    "with 6 questions each, followed by Licensing (3) and Platform (2). The "
    "most repeated question was Thread Scaling (asked 2 times).\"}\n"
    "Respond with JSON only."
)

DETECT_SYSTEM = (
    "If the messages are empty, malformed, or you cannot proceed with "
    "confidence, return {\"questions\": []} and nothing else. Do not guess.\n\n"
    "You find IMPLICIT requests for help — problems stated as complaints or "
    "symptoms that imply the writer wants a solution, even with no question "
    "mark.\n"
    "An implicit request exists if the writer describes something broken, "
    "blocked, failing, or behaving unexpectedly in a way that implies they "
    "want it resolved.\n"
    "IS an implicit request: 'can't get the copy task to fire, stuck all "
    "day', 'metering numbers don't match the report', 'the transfer just "
    "hangs'.\n"
    "NOT an implicit request: announcements ('deployed the fix, all green'), "
    "opinions, status updates without a problem, social chat, answers.\n"
    "Rewrite each implicit request as ONE standalone question. Preserve "
    "technical tokens exactly; keep each question in its original language.\n"
    "Respond with JSON only: {\"questions\": [{\"index\": <message number>, "
    "\"question\": \"<the rewritten question>\"}]}. Use an empty list if none qualify."
)

EXTRACT_SYSTEM = (
    "If the messages are empty, malformed, or contain no request for help, "
    "return {\"questions\": []} and nothing else. Do not guess.\n\n"
    "You extract answerable questions from support chat messages.\n"
    "Classify each candidate sentence:\n"
    "- REAL: the writer wants information, a fix, a yes/no confirmation, OR "
    "a new capability. A relayed wish ('customer would like X', 'they want "
    "to be able to Y', 'feature request: ...') is a REAL ask — rewrite it "
    "as a question and tag it 'feature-request'. A described problem the "
    "writer is stuck on ('been stuck on this all morning', 'out of ideas') "
    "is an implicit help request — REAL even with no question mark. KEEP.\n"
    "- RHETORICAL: shaped like a question but seeks no answer ('Any "
    "thoughts?', 'Right?', 'Make sense?', 'Can you believe it?', "
    "'Anyone?'). Social banter is RHETORICAL even with a question mark "
    "('Anyone else excited for the holidays?') — it seeks solidarity, not "
    "information. DROP.\n"
    "- CONTEXT: a statement describing their setup, environment, or what they "
    "already tried. Not a question — DROP it, but USE its facts to make the "
    "REAL questions standalone.\n"
    "Most channel traffic is NOT a question: announcements, status updates, "
    "thanks/closures ('fixed now'), FYI links, and decisions yield NOTHING. "
    "The venting test: a complaint WITH a concrete symptom ('uploads hang "
    "at 99% for one customer, no errors anywhere, completely stuck') IS a "
    "REAL implicit ask — rewrite the symptom as a question. A complaint "
    "with only mood ('this rollout has been a mess from start to finish') "
    "has nothing to answer — extract NOTHING. A log paste with no request "
    "attached (posted 'for visibility' or noted as self-resolved) yields "
    "NOTHING — log text alone is not a question. Never rewrite a statement "
    "INTO a question.\n"
    "Rewrite every REAL question as a single standalone question.\n"
    "Standalone test: a reader who never saw the message can answer it "
    "without asking what 'it', 'this', 'they', or 'the platform' refers to — "
    "pull the subject into the question explicitly. But ONLY from the "
    "message itself: if the message names no subject ('Is there a limit?'), "
    "keep the question as written — NEVER invent specifics the message "
    "doesn't contain. Drop greetings, signatures, and bullet characters.\n"
    "Rules:\n"
    "- A message often contains MORE THAN ONE real question: output one entry "
    "per REAL question, repeating the message number. Implicit help requests "
    "count as REAL. Asks joined by 'and separately', 'also', 'second "
    "question', or a numbered list of UNRELATED requests are DISTINCT asks — "
    "output every one of them; never drop the later ones.\n"
    "- Output each distinct ask ONCE. Never output two rewrites of the same "
    "ask from one message (asking 'what is the error' and 'why does it fail' "
    "about the same failure is ONE ask).\n"
    "- An 'Or ...?' follow-up offering an ALTERNATIVE route to the SAME goal "
    "('Can X do this? Or do we have to build it ourselves?') is part of that "
    "one ask — output ONE question covering the goal, not two.\n"
    "- Enumerated steps of a single workflow or task list ('1. Find the file "
    "2. Move it 3. Merge them') are ONE question about whether the whole "
    "workflow is possible — never one question per step.\n"
    "- Sub-steps in service of a stated goal are NOT independent questions: "
    "if the goal is merging files and finding them is a step, output ONLY "
    "the goal question. Never emit a separate question for scaffolding the "
    "asker didn't independently ask about.\n"
    "- Preserve technical tokens exactly as written: error strings, API "
    "names, product names, version numbers, file names. Never normalize, "
    "paraphrase, or invent them.\n"
    "- Preserve the asker's intent verb exactly. HANDLE/CATCH/RESOLVE an "
    "error is NOT the same as BYPASS/DISABLE/IGNORE it. If you cannot tell "
    "what outcome they want, describe the symptom — do not name a "
    "resolution they didn't state.\n"
    "- Messages may be in any language: keep each question in its original "
    "language.\n"
    "- Tag each question with its type: 'how-to' (how do I do X), "
    "'troubleshooting' (something is broken or failing), 'is-it-possible' "
    "(does the product support X), 'feature-request' (asking for a new "
    "capability), 'defect-report' (reporting a bug).\n"
    "Respond with JSON only: {\"questions\": [{\"index\": <message number>, "
    "\"question\": \"<standalone question>\", \"type\": \"<type>\"}]}. "
    "Use an empty list if none qualify."
)

EXTRACT_FEW_SHOT = """Example messages:
0. Hi all! Quick one — can we trigger transfers via REST instead of the scheduler? Also is there a way to bulk-disable actions?
1. Deployed the fix to prod this morning, all green.
2. Re: wM MFT (SaaS), customer product feedback. They would like to 1. See a graphical representation of the Action log showing status of each task. 2. Is there a way to group Actions and filter by group name? Any thoughts?
3. Can we configure the following tasks to do this? 1. Find file001.txt in folder 2. Find file002.txt in folder 3. Invoke Flow Service to merge the files
4. copy task is failing during the virus scan, please advise

Correct answer: {"questions": [
{"index": 0, "question": "Can we trigger transfers via REST instead of the scheduler?", "type": "is-it-possible"},
{"index": 0, "question": "Is there a way to bulk-disable actions?", "type": "is-it-possible"},
{"index": 2, "question": "Can the wM MFT (SaaS) Action log show a graphical representation of each task's status?", "type": "feature-request"},
{"index": 2, "question": "Can Actions in wM MFT (SaaS) be grouped and filtered by group name?", "type": "is-it-possible"},
{"index": 3, "question": "Can tasks be configured to find file001.txt and file002.txt in a folder and merge them with a Flow Service?", "type": "is-it-possible"},
{"index": 4, "question": "Why does a copy task fail during the antivirus scanning phase?", "type": "troubleshooting"}]}

DO NOT answer message 2 like this: {"index": 2, "question": "They want to see the action log and group actions, any thoughts?"}
Wrong because: 'Any thoughts?' is RHETORICAL and must be dropped; the two REAL questions must be split into two entries; 'they' must be made explicit using the CONTEXT sentence.

DO NOT answer message 3 with three entries ("Find file001.txt in folder", "Find file002.txt in folder", "Invoke Flow Service to merge the files").
Wrong because: those are steps of ONE workflow — the only ask is whether the workflow can be configured.

DO NOT answer message 4 like this: {"index": 4, "question": "How do I bypass antivirus scanning errors?"}
Wrong because: the asker never said bypass — they reported a failure and want it explained or fixed. Describe the symptom; never name a resolution they didn't state.

DO NOT split "Can the tool export reports on a schedule? Or do we have to script that ourselves?" into two entries.
Wrong because: the 'Or ...?' offers an alternative route to the SAME goal (scheduled report export) — that is ONE ask, one entry.

DO NOT invent a question from a message like "Saw this in the logs but it self-resolved, posting for visibility: WARN retry 1 of 2 for endpoint X / INFO completed".
Wrong because: nobody asked anything — a log paste with no request attached yields NO entries.

DO NOT keep "Anyone else excited for the long weekend?" as a question.
Wrong because: social banter seeks solidarity, not information — RHETORICAL, no entry, even with the question mark.

Now extract from these messages."""

# The worked example's output questions. Small models sometimes copy them
# into real output; an extraction that reproduces one WITHOUT strong textual
# support in its claimed source message is prompt contamination, not content.
_EXAMPLE_QUESTION_TEXTS = (
    'Can we trigger transfers via REST instead of the scheduler?',
    'Is there a way to bulk-disable actions?',
    "Can the wM MFT (SaaS) Action log show a graphical representation of "
    "each task's status?",
    'Can Actions in wM MFT (SaaS) be grouped and filtered by group name?',
    'Can tasks be configured to find file001.txt and file002.txt in a folder '
    'and merge them with a Flow Service?',
    'Why does a copy task fail during the antivirus scanning phase?',
)


def _looks_like_example_leak(question: str, source_message: str) -> bool:
    """
    True when an extracted question reproduces a few-shot example (high token
    Jaccard) AND its claimed source message can't account for its vocabulary.
    A transcript that GENUINELY contains a similar question keeps it: nearly
    all of its content words appear in the source.
    """
    toks = set(re.findall(r'[a-z0-9]+', question.lower()))
    if not toks:
        return False
    if not any(len(toks & ex) / len(toks | ex) >= 0.75
               for ex in _EXAMPLE_TOKEN_SETS):
        return False
    content = [t for t in toks if len(t) > 3]
    if not content:
        return True
    msg = source_message.lower()
    supported = sum(1 for t in content if t in msg)
    return supported / len(content) < 0.8


_EXAMPLE_TOKEN_SETS = [set(re.findall(r'[a-z0-9]+', t.lower()))
                       for t in _EXAMPLE_QUESTION_TEXTS]

ANSWERED_SYSTEM = (
    "If the thread is empty or malformed, respond {\"verdict\": \"unknown\"} "
    "and nothing else. Do not guess.\n\n"
    "You decide whether a question in a chat thread was actually ANSWERED by "
    "the replies.\n"
    "A question is ANSWERED only if a reply contains specific, actionable "
    "information: a setting, a command, an API or endpoint, a config value, "
    "a yes/no WITH a reason, or a link to the exact relevant doc.\n"
    "The following are NOT answers, even if friendly or on-topic: "
    "acknowledgments ('thanks', 'good question', 'same here'); promises "
    "('let me check', 'I'll get back to you', 'looping in X'); asking for "
    "more details; restating or clarifying the problem; unrelated chatter.\n"
    "A thread may contain SEVERAL asks: judge only the question shown. "
    "Replies often answer by number ('for #2, yes — use X; not sure on "
    "#1') — use the first message to work out which numbered ask is the "
    "question being judged, and count only the part of the reply aimed at "
    "it.\n"
    "Example: question 'How do I reset my password?', reply 'Settings > "
    "Security > Reset, then check your email.': {\"verdict\": \"answered\"}\n"
    "Example: question 'Why did my transfer fail?', reply 'Hmm, let me check "
    "with the team.': {\"verdict\": \"unanswered\"}\n"
    "Respond with JSON only: {\"verdict\": \"answered\"}, "
    "{\"verdict\": \"unanswered\"}, or {\"verdict\": \"unknown\"}."
)


class GroupLabeler:
    """LLM prompting layer (labels, verification, summaries, detection)."""

    # 30m, not 10m: a big first-run embedding stage sits between the chat
    # warm-up and the first judgment call, and an expired keep_alive there
    # costs a full model reload at the worst time
    KEEP_ALIVE = '30m'
    SEED = 42
    CHAT_RETRIES = 3
    CHAT_BACKOFF_SECONDS = 1.0

    def __init__(self, provider: str = 'ollama'):
        load_dotenv()
        self.provider = provider
        # 8B models on CPU can take well over a minute per call; a short
        # timeout silently downgrades the whole pipeline to regex
        self.timeout = int(os.getenv('LLM_TIMEOUT', '180'))
        self._available: Optional[bool] = None

        if provider != 'ollama':
            raise ValueError(f"Unknown provider '{provider}'. This tool runs "
                             f"entirely on local Ollama.")
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
        self.model = default_generation_model()
        self._model_pinned = bool(os.getenv('OLLAMA_GENERATION_MODEL'))

        # Token-heavy work (extraction: hundreds of output tokens) runs on a
        # fast model; short judgment calls (verify/audit/label: a few tokens)
        # keep the quality model. On CPU an 8B model writes minutes per
        # extraction batch — quality belongs in the judgments, not the typing.
        self.fast_model = os.getenv('OLLAMA_FAST_MODEL') or self.model

        # Outputs are deterministic (temperature 0, fixed seed), so caching by
        # prompt makes re-analyzing the same transcript free
        cache_enabled = os.getenv('LLM_CACHE', 'on').lower() not in ('off', '0', 'false')
        self._cache = JsonDiskCache(provider, self.model or 'none',
                                    os.getenv('LLM_CACHE_DIR', '.llm_cache'),
                                    enabled=cache_enabled,
                                    max_entries=int(os.getenv('LLM_CACHE_MAX', '5000')))

        # Optional domain hint injected into every prompt: knowing the product
        # makes small models dramatically more specific
        self.domain_context = os.getenv('DOMAIN_CONTEXT', '').strip()

        # Abstain/verdict counters: the early-warning system. A spiking
        # verify-false or extract-empty rate means an upstream stage is
        # under-forming and a downstream pass is papering over it.
        self.stats: Dict[str, int] = {}

        # Cancellation hook: called before every LLM call so a cancel takes
        # effect between calls instead of waiting for a stage boundary
        # (a single call can run minutes on CPU). Raising is the contract.
        self.cancel_check = None

    def _count(self, key: str, n: int = 1):
        self.stats[key] = self.stats.get(key, 0) + n

    def flush_cache(self):
        """Write any throttled-but-unsaved LLM verdicts to disk. Called at
        the end of an analysis; per-call saves batch via min_dirty."""
        self._cache.save()

    def _system(self, base: str) -> str:
        """System prompt with the optional domain context appended."""
        if self.domain_context:
            return (f"{base}\nContext: the messages come from "
                    f"{self.domain_context}. Technical terms, error strings, "
                    f"API names, product names, and version numbers must be "
                    f"preserved exactly.")
        return base

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def available(self) -> bool:
        """Whether a generation model is actually usable (cached)."""
        if self._available is None:
            self._available = self._check_available()
        return self._available

    def _check_available(self) -> bool:
        try:
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            except requests.RequestException:
                # One slow answer (Ollama busy loading a model on the same
                # box) must not silently downgrade the WHOLE run to regex —
                # retry once with a patient timeout before giving up
                logger.warning("Ollama availability probe slow/failed; "
                               "retrying with a longer timeout...")
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=15)
            response.raise_for_status()
            names = [m.get('name', '') for m in response.json().get('models', [])]
            def downloaded(model):
                return any(n == model or n.startswith(f"{model}:") for n in names)
            if downloaded(self.model):
                if (not os.getenv('OLLAMA_FAST_MODEL')
                        and self.model != FALLBACK_GENERATION_MODEL
                        and downloaded(FALLBACK_GENERATION_MODEL)):
                    self.fast_model = FALLBACK_GENERATION_MODEL
                    logger.info("Model split: '%s' for question extraction "
                                "(token-heavy), '%s' for judgment calls",
                                self.fast_model, self.model)
                return True
            # Preferred model not downloaded: quietly drop to the small one
            # rather than losing all LLM features (env-pinned models never switch)
            if (not self._model_pinned and self.model != FALLBACK_GENERATION_MODEL
                    and downloaded(FALLBACK_GENERATION_MODEL)):
                logger.info("Chat model '%s' not downloaded; using '%s' instead",
                            self.model, FALLBACK_GENERATION_MODEL)
                self.model = FALLBACK_GENERATION_MODEL
                self.fast_model = FALLBACK_GENERATION_MODEL
                self._cache.save()  # flush throttled writes before the swap
                self._cache = JsonDiskCache(
                    self.provider, self.model,
                    os.getenv('LLM_CACHE_DIR', '.llm_cache'),
                    enabled=self._cache.enabled,
                    max_entries=int(os.getenv('LLM_CACHE_MAX', '5000')))
                return True
            return False
        except (requests.RequestException, ValueError):
            return False

    def warm_up(self, include_quality: bool = True) -> None:
        """
        Load the Ollama model(s) into memory before the first real call.

        Model load alone can exceed a per-call timeout, making the first
        calls fail and the pipeline silently fall back to regex. An empty
        chat request loads a model without generating anything. The fast
        model (needed first, for extraction) loads synchronously.

        include_quality=False defers the quality model: when a long
        fast-model pass (extraction, detection) runs next, background-
        loading the 8B alongside it makes the two models evict each other
        on tight-RAM machines — call warm_up_quality() when the fast pass
        is done instead.
        """
        if not self.model:
            return
        if self.cancel_check is not None:
            self.cancel_check()
        self.available()  # resolves the fast/quality model split

        def load(model):
            try:
                logger.info("Loading chat model '%s' into memory...", model)
                requests.post(f"{self.ollama_url}/api/chat",
                              json={'model': model, 'messages': [],
                                    'keep_alive': self.KEEP_ALIVE},
                              timeout=max(self.timeout, 600))
            except requests.RequestException as e:
                logger.warning("Chat model warm-up failed for '%s': %s", model, e)

        load(self.fast_model)
        if include_quality:
            self.warm_up_quality()

    def warm_up_quality(self):
        """Background-load the quality model. Called AFTER the fast-model
        work is dispatched: loading the 8B while the 3B is mid-extraction
        makes the two evict each other on 12-16GB machines — each load is
        10-60s from disk, paid several times at the worst moment."""
        if not self.model or self.model == self.fast_model:
            return

        def load(model):
            try:
                logger.info("Loading chat model '%s' into memory...", model)
                requests.post(f"{self.ollama_url}/api/chat",
                              json={'model': model, 'messages': [],
                                    'keep_alive': self.KEEP_ALIVE},
                              timeout=max(self.timeout, 600))
            except requests.RequestException as e:
                logger.warning("Chat model warm-up failed for '%s': %s", model, e)

        threading.Thread(target=load, args=(self.model,), daemon=True).start()

    # ------------------------------------------------------------------
    # Generation core
    # ------------------------------------------------------------------

    def _chat(self, messages: List[Dict], schema: Dict, max_tokens: int = 300,
              model: Optional[str] = None) -> str:
        """One chat completion with schema-enforced JSON output.

        Connection errors retry twice with backoff (like embeddings do) —
        a silently lost chat isn't an error here, it FLIPS two-judge
        outcomes and downgrades extraction batches to regex. Timeouts do
        NOT retry: the per-call timeout is already generous, and doubling
        a 180s wait helps nobody.
        """
        last_error = None
        for attempt in range(self.CHAT_RETRIES):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        'model': model or self.model,
                        'messages': messages,
                        'stream': False,
                        'format': schema,  # Ollama enforces the JSON schema while decoding
                        'options': {'temperature': 0, 'seed': self.SEED,
                                    'num_predict': max_tokens},
                        'keep_alive': self.KEEP_ALIVE,  # stay loaded across our call series
                    },
                    timeout=self.timeout,
                )
                # Ollama's own error sentence (e.g. "model requires more
                # system memory") must survive into the logged warning
                raise_with_detail(response)
                return response.json().get('message', {}).get('content', '')
            except requests.Timeout:
                raise
            except requests.RequestException as e:
                last_error = e
                if attempt < self.CHAT_RETRIES - 1:
                    logger.warning("Chat request failed (%s); retrying...", e)
                    time.sleep(self.CHAT_BACKOFF_SECONDS * (1 + attempt))
        raise last_error

    @staticmethod
    def _parse_json(raw: str) -> Optional[Dict]:
        """Parse model output; tolerate JSON wrapped in prose."""
        match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _generate_json(self, system: str, user: str, schema: Dict,
                       validator=None, corrective: str = '',
                       max_tokens: int = 300,
                       model: Optional[str] = None) -> Optional[Dict]:
        """
        Chat once; if validation fails, retry once with corrective feedback
        appended to the conversation. Returns None when both attempts fail.
        Successful (validated) results are cached on disk by prompt.
        """
        if self.cancel_check is not None:
            self.cancel_check()  # outside the try: cancellation must propagate
        cache_key = f"{model or self.model}\n{system}\n{user}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': user}]
        try:
            raw = self._chat(messages, schema, max_tokens=max_tokens, model=model)
            data = self._parse_json(raw)
            problem = ('Response was not valid JSON.' if data is None
                       else (validator(data) if validator else None))
            if problem is None:
                self._cache.set(cache_key, data)
                self._cache.save(min_dirty=8)
                return data

            if corrective:
                messages.append({'role': 'assistant', 'content': raw})
                messages.append({'role': 'user', 'content': f"{corrective} ({problem})"})
                raw = self._chat(messages, schema, max_tokens=max_tokens, model=model)
                data = self._parse_json(raw)
                if data is not None and (validator is None or validator(data) is None):
                    self._cache.set(cache_key, data)
                    self._cache.save(min_dirty=8)
                    return data
            return None
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_label(data: Dict) -> Optional[str]:
        """Return a problem description, or None when the label is good."""
        topic = str(data.get('topic', '')).strip()
        summary = str(data.get('summary', '')).strip()
        if not topic:
            return 'topic was empty'
        if topic.upper() == 'NEEDS_REVIEW':
            return None  # the model's honest abstain is a valid answer
        words = [w.strip('.,!?').lower() for w in topic.split()]
        if all(w in GENERIC_TOPIC_WORDS for w in words):
            return f"topic '{topic}' is too generic"
        if len(words) > MAX_TOPIC_WORDS:
            return (f"topic '{topic}' is {len(words)} words — use "
                    f"{MAX_TOPIC_WORDS} or fewer")
        if not summary:
            return 'summary was empty'
        return None

    def label_group(self, question_texts: List[str],
                    keywords: Optional[List[str]] = None) -> Optional[Dict[str, str]]:
        """
        Generate {'topic', 'summary'} for a group of similar questions.
        Returns None on any failure — callers should fall back gracefully.
        """
        sample = question_texts[:MAX_QUESTIONS_IN_PROMPT]
        parts = [LABEL_FEW_SHOT]
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords[:5])}")
        parts.append('Questions:\n' + '\n'.join(f"- {q}" for q in sample))
        user = '\n'.join(parts)

        data = self._generate_json(
            self._system(LABEL_SYSTEM), user, LABEL_SCHEMA,
            validator=self._validate_label,
            corrective='Your previous answer was rejected. The topic must name the '
                       'specific feature, system, or task in 2-4 words. Respond with '
                       'JSON only.',
        )
        if data is None:
            return None

        topic = str(data['topic']).strip()
        if topic.upper() == 'NEEDS_REVIEW':
            self._count('label_needs_review')
            return None  # abstained: callers fall back, nothing gets banked
        return {'topic': topic, 'summary': str(data['summary']).strip()}

    def verify_same_topic(self, questions_a: List[str],
                          questions_b: List[str]) -> Optional[bool]:
        """Decide whether two borderline groups ask about the same topic."""
        user = (
            'Group A:\n' + '\n'.join(f"- {q}" for q in questions_a[:3]) +
            '\n\nGroup B:\n' + '\n'.join(f"- {q}" for q in questions_b[:3]) +
            '\n\nDo Group A and Group B ask about the same underlying topic?'
        )
        data = self._generate_json(self._system(VERIFY_SYSTEM), user, VERIFY_SCHEMA, max_tokens=60)
        if data is None or not isinstance(data.get('same_topic'), bool):
            self._count('verify_uncertain')
            return None
        self._count('verify_true' if data['same_topic'] else 'verify_false')
        return data['same_topic']

    def confirm_same_ask(self, dropped: str, kept: List[str],
                         message: str = '') -> Optional[bool]:
        """
        Second judge for same-message consolidation: is the nominated drop
        REALLY the same ask as a kept question? Deliberately NOT
        verify_same_topic — its "one documentation page" test confirms
        topic-mates, and two distinct asks from one message often live on
        the same doc page (retention policy vs auto-delete). The yardstick
        here is CONSOLIDATE's own: would one answer fully resolve both?

        The judge sees the ORIGINAL MESSAGE: the first judge saw it, and
        stripping it here made this judge blind to exactly the evidence
        that matters — a trailing 'can agents be leveraged HERE?' loses its
        antecedent without the message, reads as wanting a different
        outcome, and a correct fold gets overruled (field finding,
        2026-07-07 run: five continuation fragments survived this way).
        """
        context = ''
        if message:
            context = ('Original message (both questions came from it):\n'
                       f'"{message[:400]}"\n\n')
        user = (
            context +
            'Question A (nominated as a duplicate):\n' + f"- {dropped}" +
            '\n\nQuestion B (kept):\n' +
            '\n'.join(f"- {q}" for q in kept[:3]) +
            '\n\nBoth came from the SAME message. Is A the SAME ask as one '
            'of B — meaning one single answer would fully resolve both? '
            'A short follow-up that leans on B for its meaning (a deictic '
            '"here"/"that", a request for the doc/link that answers B, a '
            '"so ...?" confirmation) is the SAME ask. Sharing a feature '
            'area is NOT enough: wanting different outcomes means '
            'different asks. If in doubt, answer false '
            '(keeping both is safe; deleting a real ask is not).'
        )
        data = self._generate_json(self._system(CONSOLIDATE_SYSTEM), user,
                                   SAME_ASK_SCHEMA, max_tokens=60)
        if data is None or not isinstance(data.get('same_ask'), bool):
            self._count('same_ask_uncertain')
            return None
        self._count('same_ask_true' if data['same_ask'] else 'same_ask_false')
        return data['same_ask']

    def audit_group(self, questions: List[str]) -> Optional[List[int]]:
        """
        Quality-check a formed group: 0-based indices of questions that
        clearly don't belong, [] when coherent, None when the LLM is
        unavailable/uncertain (callers keep the group as-is).
        """
        sample = questions[:MAX_QUESTIONS_IN_PROMPT]
        user = ('Group of matched questions:\n' +
                '\n'.join(f"{i + 1}. {q}" for i, q in enumerate(sample)) +
                '\n\nWhich question numbers (if any) are clearly about a '
                'different subject than the rest?')
        data = self._generate_json(self._system(AUDIT_SYSTEM), user, AUDIT_SCHEMA,
                                   max_tokens=60)
        if data is None or not isinstance(data.get('outliers'), list):
            self._count('audit_uncertain')
            return None
        outliers = [int(i) - 1 for i in data['outliers']
                    if isinstance(i, int) and 1 <= i <= len(sample)]
        # "Everything is an outlier" is not a meaningful audit verdict
        if len(outliers) >= len(sample):
            self._count('audit_uncertain')
            return None
        self._count('audit_clean' if not outliers else 'audit_evictions',
                    1 if not outliers else len(outliers))
        return outliers

    def choose_bucket(self, question: str,
                      candidates: List[Dict]) -> Optional[int]:
        """
        Closed-choice adjudication for an ambiguously routed question.
        candidates: [{'id': int, 'name': str}]. Returns the chosen id;
        0 when the model abstains (fits both/neither — quarantine it);
        None on failure/invalid answer (callers fall back to the embedding
        favorite).
        """
        listing = '\n'.join(f"{c['id']} {c['name']}" for c in candidates)
        # Abstain must be a VISIBLE option: given only real categories, a
        # small model picks from the list every time — off-topic questions
        # were force-routed with the abstain rule sitting unused in the
        # system prompt
        listing += ('\n0 NONE OF THESE (off-topic, not about this product, '
                    'too vague to place, or fits several equally)')
        user = f"Categories:\n{listing}\nQuestion: {question}\nAnswer with the JSON."
        data = self._generate_json(self._system(ROUTE_SYSTEM), user, ROUTE_SCHEMA,
                                   max_tokens=20)
        if data is None:
            return None
        chosen = data.get('category')
        if chosen == 0:
            return 0  # honest abstain — the caller's review pile, not a guess
        if isinstance(chosen, int) and any(c['id'] == chosen for c in candidates):
            return chosen
        return None

    def assign_themes(self, items: List[str]) -> Optional[List[Optional[str]]]:
        """
        Funnel roll-up: organize topics/questions into 3-6 broad themes.
        Returns one theme name per input item (None for items the model
        failed to place), or None when the call fails entirely.
        """
        numbered = '\n'.join(f"{i + 1}. {item[:160]}" for i, item in enumerate(items))
        data = self._generate_json(
            self._system(THEMES_SYSTEM),
            f"Topics and questions:\n{numbered}\n\nOrganize them into themes.",
            THEMES_SCHEMA, max_tokens=400)
        if data is None or not isinstance(data.get('themes'), list):
            return None

        assigned: List[Optional[str]] = [None] * len(items)
        for theme in data['themes']:
            name = str(theme.get('name', '')).strip()
            if not name:
                continue
            for i in theme.get('items', []):
                if isinstance(i, int) and 1 <= i <= len(items):
                    assigned[i - 1] = name
        return assigned if any(assigned) else None

    def consolidate_same_ask(self, message_text: str,
                             question_texts: List[str]) -> Optional[List[int]]:
        """
        Which of several questions extracted from ONE message are distinct
        asks? Returns 1-based indices to KEEP, or None on failure/abstain
        (callers keep everything).
        """
        numbered = '\n'.join(f"{i + 1}. {q}" for i, q in enumerate(question_texts))
        user = (f"Original message:\n{message_text[:400]}\n\n"
                f"Questions extracted from it:\n{numbered}\n\n"
                "Which numbers are distinct asks to keep?")
        data = self._generate_json(self._system(CONSOLIDATE_SYSTEM), user,
                                   CONSOLIDATE_SCHEMA, max_tokens=40)
        if data is None or not isinstance(data.get('keep'), list):
            return None
        keep = sorted({int(i) for i in data['keep']
                       if isinstance(i, int) and 1 <= i <= len(question_texts)})
        if not keep:
            return None  # "keep nothing" is not a meaningful verdict
        return keep

    def confirm_feature_request(self, question: str,
                                context: str = '') -> Optional[bool]:
        """
        Second opinion before a question leaves the support funnel: is this
        genuinely a request for a capability that doesn't exist? The asker's
        ORIGINAL message is supplied — the wish-phrasing ("would be great
        if...") lives there, not in the rewrite. Abstain or failure -> None
        (callers keep it in support — the safer home).
        """
        user = (f"Original message: {context[:300]}\n" if context else '') + \
               f"Question: {question}\nProduct feedback?"
        data = self._generate_json(self._system(FEEDBACK_SYSTEM), user,
                                   FEEDBACK_SCHEMA, max_tokens=20)
        if data is None or not isinstance(data.get('feature_request'), bool):
            return None
        self._count('feedback_confirmed' if data['feature_request']
                    else 'feedback_rejected')
        return data['feature_request']

    def summarize_analysis(self, groups: List[Dict], total_questions: int,
                           themes: Optional[List[Dict]] = None) -> Optional[str]:
        """Write a 1-2 sentence executive summary of the top question groups."""
        lines = [f"Total questions analyzed: {total_questions}"]
        if themes:
            lines.append('Themes (exact, deterministic counts): ' +
                         ', '.join(f"{t['name']} - {t['count']}" for t in themes))
        lines.append('Recurring question groups (sub-topics within themes, '
                     'NOT themes):')
        for i, group in enumerate(groups[:10], 1):
            topic = group.get('topic') or group['representative_question']
            lines.append(f"{i}. {topic} — asked {group['count']} times "
                         f"(e.g. \"{group['representative_question']}\")")
        data = self._generate_json(
            self._system(SUMMARY_SYSTEM), '\n'.join(lines), SUMMARY_SCHEMA,
            validator=lambda d: None if str(d.get('summary', '')).strip() else 'summary was empty',
            corrective='Respond with JSON only and a non-empty summary.',
        )
        if data is None:
            return None
        summary = str(data['summary']).strip()
        if summary.upper() == 'NEEDS_REVIEW':  # the model's honest abstain
            return None
        return summary

    def draft_answer(self, question: str, replies: List[str],
                     validator=None) -> Optional[str]:
        """
        Condense a thread's replies into a short draft FAQ answer.

        NOTHING is answered from the model's own knowledge: the prompt
        forbids it, and the caller's validator (deterministic grounding
        against the reply text) rejects any draft that smuggles in a fact
        the replies don't contain — the export then falls back to quoting
        the raw replies. Returns None when unavailable/rejected/empty.
        """
        if not replies:
            return None
        user = ('Question people keep asking:\n' + question +
                '\n\nReplies from the support thread(s):\n' +
                '\n'.join(f'- {r}' for r in replies[:4]) +
                '\n\nCondense the replies into a draft FAQ answer. Facts '
                'from the replies ONLY; empty answer if they do not answer '
                'the question.')
        data = self._generate_json(
            self._system(FAQ_DRAFT_SYSTEM), user, FAQ_DRAFT_SCHEMA,
            validator=validator, max_tokens=220,
            corrective='Use ONLY facts, names, and numbers that appear in '
                       'the replies. Remove anything they do not state.')
        if data is None:
            self._count('faq_draft_failed')
            return None
        answer = str(data.get('answer', '')).strip()
        if not answer:
            self._count('faq_draft_abstained')  # honest "replies answer nothing"
            return None
        self._count('faq_drafted')
        return answer

    def detect_questions(self, message_texts: List[str]) -> List[Dict]:
        """
        Find implicit help requests in messages the regex extractor skipped.
        Returns [{'index': int, 'question': str}], empty on failure.
        """
        return self._questions_from_llm(message_texts, self._system(DETECT_SYSTEM)) or []

    def extract_questions(self, message_texts: List[str],
                          thorough: bool = False) -> Optional[List[Dict]]:
        """
        LLM-first extraction (LLM_EXTRACTION=full): pull every question out of
        every message, cleaned and rewritten. Multiple questions per message
        are allowed. Returns [{'index': int, 'question': str}]; an empty list
        means "no questions here", None means the call failed.

        thorough=True re-runs with the quality model — used to double-check
        messages the fast model skipped that look like questions.
        """
        return self._questions_from_llm(message_texts, self._system(EXTRACT_SYSTEM),
                                        model=self.model if thorough else None)

    # Real messages are numbered from a range DISJOINT from the few-shot
    # examples' 0-4: with both numbered from 0 in one turn, a small model
    # can extract example content attributed to real indices — the exact
    # misattribution _verify_source exists to patch downstream
    MESSAGE_INDEX_BASE = 101

    def _questions_from_llm(self, message_texts: List[str], system: str,
                            model: Optional[str] = None) -> Optional[List[Dict]]:
        base = self.MESSAGE_INDEX_BASE
        numbered = '\n'.join(f"{base + i}. {text[:600]}"
                              for i, text in enumerate(message_texts))
        user = (f"{EXTRACT_FEW_SHOT}\n{numbered}" if system.startswith(EXTRACT_SYSTEM[:40])
                else f"Messages:\n{numbered}")
        data = self._generate_json(system, user, DETECT_SCHEMA,
                                   max_tokens=800, model=model or self.fast_model)
        if data is None or not isinstance(data.get('questions'), list):
            self._count('extract_failed_batches')
            return None
        self._count('extract_batches')
        if not data['questions']:
            self._count('extract_empty_batches')

        found = []
        for item in data['questions']:
            if not isinstance(item, dict):
                continue
            index = item.get('index')
            if isinstance(index, int):
                index -= base  # example indices (0-4) land negative: rejected
            question = str(item.get('question', '')).strip()
            if isinstance(index, int) and 0 <= index < len(message_texts) and question:
                if _looks_like_example_leak(question, message_texts[index]):
                    logger.info("Dropped a prompt-example leak (copied from "
                                "the few-shot, unsupported by its source): "
                                "%.80r", question)
                    self._count('extract_example_leaks')
                    continue
                entry = {'index': index, 'question': question}
                if item.get('type') in QUESTION_TYPES:
                    entry['type'] = item['type']
                found.append(entry)
        return found

    def is_answered(self, question: str, replies: List[str],
                    context: str = '') -> Optional[bool]:
        """Decide whether thread replies actually answered the question.

        context is the thread's first message: a multi-ask thread's replies
        often answer by NUMBER ('for #2, yes...'), which is unresolvable
        from the rewritten question alone."""
        intro = (f"Thread's first message (for context): {context[:300]}\n\n"
                 if context else '')
        user = (
            f"{intro}Question being judged: {question}\n\nThread replies:\n" +
            '\n'.join(f"- {r[:300]}" for r in replies[:5]) +
            '\n\nWas THIS question answered by these replies?'
        )
        data = self._generate_json(self._system(ANSWERED_SYSTEM), user, ANSWERED_SCHEMA, max_tokens=60)
        if data is None:
            return None
        verdict = str(data.get('verdict', '')).strip().lower()
        if verdict == 'answered':
            return True
        if verdict == 'unanswered':
            return False
        return None  # 'unknown' or malformed: abstain
