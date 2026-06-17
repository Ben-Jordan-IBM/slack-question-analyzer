# The Question Funnel — Pipeline Spec (v2.40.1, prompt pack 20, taxonomy v3)

> The stages below describe the CURRENT architecture. The change-by-change
> history of how it got here (14 measured eval rounds) is preserved in the
> [appendix](#appendix--design-history-by-eval-round) at the bottom.

How a Slack transcript becomes ranked topics. Design rule throughout: **the
language model is never asked to do the hard open-ended thing.** Embeddings
handle similarity, plain code handles counting/merging/collapsing, and the
LLM only answers small closed questions. Two models share the work:

| Role | Default model | Used for |
|---|---|---|
| Fast (typing) | `llama3.2` (3B) | Question extraction & detection — hundreds of output tokens |
| Quality (judging) | `llama3.1:8b` | Route adjudication, merge verify, group audit, labels, summary, answered — a few tokens each |

All LLM calls use Ollama structured output (`format=<JSON schema>`,
temperature 0, seed 42) — malformed output is impossible at decode time.
Verdicts are cached on disk keyed by (model + full prompt), so re-runs are
free and prompt edits self-invalidate the cache.

---

## Stage 0 — Parse & extract (fast model + code)

1. Transcript parsed into messages (Slack JSON with threads, dashed-separator
   text, or CSV). Markup, code blocks, mentions stripped. In plain text,
   '>'-quoted lines are thread replies: attached to the parent for answer
   detection, never extracted as questions (a responder's clarifying
   question is not an ask); '# ' heading lines are structural markup.
2. **LLM-first extraction** (transcripts ≤ 150 messages; batches of 8):
   the fast model classifies each sentence REAL / RHETORICAL / CONTEXT,
   rewrites every REAL ask as a standalone question, preserves the asker's
   intent verb (HANDLE ≠ BYPASS), and tags each question with its TYPE —
   the second axis, independent of subject: how-to / troubleshooting /
   is-it-possible / feature-request / defect-report (prompt: EXTRACT, below).
3. Safety nets, in order:
   - a failed LLM batch falls back to regex extraction (questions never lost);
   - any message the fast model skipped that regex flags as a question gets a
     second look from the **quality** model; if that fails too, the regex
     version is kept;
   - recoveries identical to an already-extracted question are dropped
     (guards against the fast model crediting the wrong message);
   - **same-message rephrasing collapse** (code): two *different* rewrites
     from one message with ≥ 50% content-word overlap are one ask
     (`SAME_MESSAGE_REPHRASE_OVERLAP=0.5`). Identical text is exempt — that's
     a genuine repeat and occurrence counting owns it.

## Stage 1 — Group globally by meaning (code + embeddings + LLM QC)

ALL questions cluster together, before any category exists — so a
recurrence can never be fragmented by routing noise, and an emerging topic
can surface as a coherent cluster. In order:

1. **Lexical dedup** (code, no AI): identical normalized text merges; then
   token-Jaccard ≥ `LEXICAL_DEDUP_THRESHOLD=0.9` merges rewordings. Merged
   duplicates COUNT as occurrences (that's the 2× ranking signal).
2. **Bank claims**: the learned topic bank (seeded from 150 curated MFT
   topics, grown by every run) claims questions whose embedding matches a
   known category centroid ≥ `BANK_MATCH_THRESHOLD=0.85`. Two questions
   claiming the same category group directly, regardless of their similarity
   *to each other*. Single-question claims are released back to clustering.
3. **Average-link clustering** at a FIXED bar `IN_BUCKET_THRESHOLD=0.8`
   (a user-pinned `SIMILARITY_THRESHOLD` overrides it). The LLM gates below
   guard every borderline merge, so the adaptive noise gate stays out of
   the way; it still applies when no taxonomy is configured.
4. **Borderline merge pass**: cluster pairs whose best cross-similarity is
   within `LLM_VERIFY_MARGIN=0.03` of the bar are judged by the quality
   model (prompt: VERIFY; conservative, doubt → false; numeric guard: the
   merged cluster's average must stay ≥ bar − margin; cap `LLM_VERIFY_MAX=10`).
5. **Singleton rescue (pair-completion only)**: a singleton within
   `LLM_RESCUE_MARGIN=0.1` (average-link) of an existing PAIR is adjudicated
   by the verifier against that nearest pair only — never against 3+ groups
   (`LLM_RESCUE_MAX_GROUP=2`: every historical mega-group grew by rescuing
   into an established group). Doubt/failure → stays a singleton.
6. **Group audit (final QC, two-judge + tiebreak)**: every formed group is
   checked by the quality model (prompt: AUDIT; doubt → keep). Eviction is
   destructive, so the auditor only nominates: the verifier must confirm
   (explicit false) — except a RESCUED member, where an audit flag ties the
   judges 1-1 and the rescue is simply undone. When the verifier overrules
   an audit flag, ROUTING acts as the third judge: confident category
   disagreement between the member and its group splits the member out.
7. Representative question = the member closest to the group centroid
   (extractive, never generated). Keywords = group word frequency × inverse
   frequency in the REST of the corpus (corpus-wide words score zero).

## Stage 2 — Route each cluster into a bucket (embeddings + closed-choice LLM)

`taxonomy.json` defines 8 buckets, each with an `anchor` paragraph and a
fixed `category` (the merge map). **The anchors are the single biggest
routing lever — they're what the embeddings actually match against — and
they are written in ASKERS' language (symptoms, goals, the messy way people
describe problems: "the file just sits there"), not documentation
language.**

- Each CLUSTER routes by its representative question: every representative
  and every anchor is embedded (`nomic-embed-text`, `clustering:` prefix),
  and the cluster goes to its nearest anchor.
- Best anchor below `ROUTE_OUTLIER_FLOOR=0.4` → the whole cluster is
  **needs review** (kept, amber-flagged; never forced into the closest
  wrong home). A MULTI-question review cluster is the emerging-category
  radar, logged loudly.
- Best anchor below `ROUTE_CONFIDENCE_FLOOR=0.55`, or top-2 anchors within
  `ROUTE_AMBIGUITY_MARGIN=0.05` → the quality model adjudicates a closed
  single-number choice (prompt: ROUTE, below; cap `ROUTE_LLM_MAX=20`), with
  "0 NONE OF THESE" presented as a listed option. Reply `0` = honest
  abstain → needs review.
- Health stats land in `metadata.routing` (routed / ambiguous /
  llm_adjudicated / needs_review) tagged with the taxonomy version. A rising
  review rate = the taxonomy is missing a bucket.

## Stage 3 — Name the groups (bank → LLM → keywords)

1. The **bank** names groups whose centroid matches a known category ≥ 0.85
   (stable names across runs, zero LLM calls; "recurring ×N" badge).
2. Unrecognized groups get an **LLM label** (prompt: LABEL; extractive only,
   `NEEDS_REVIEW` abstain). LLM-labeled groups are recorded into the bank —
   this is how the taxonomy of specific topics grows from real traffic.
3. Anything else falls back to keywords — and keyword names are NEVER
   banked (a junk name that sticks is worse than relabeling next time).

## Stage 4 — Merge map & summary (code + one LLM call)

- Each bucket collapses into its fixed `category` from `taxonomy.json` —
  the THEMES strip. Deterministic counts; no model involved.
- The executive summary (prompt: SUMMARY) receives those exact theme totals
  plus the ranked topic counts, and must lead with the themes. It may not
  invent topics, infer counts, or pick a winner among ties;
  `NEEDS_REVIEW` abstain → no summary shown.
- Optional: answer detection over thread replies (prompt: ANSWERED,
  three-way verdict) feeds the "Answered" stat.

---

## All knobs (.env)

| Variable | Default | Meaning |
|---|---|---|
| `SIMILARITY_THRESHOLD` | unset (auto) | pinning it overrides every bar, including the global one |
| `IN_BUCKET_THRESHOLD` | 0.8 | fixed GLOBAL grouping bar (name predates the reorder) |
| `BANK_MATCH_THRESHOLD` | 0.85 | question/group → known-category match floor |
| `EXTRACT_QUALITY_MAX` | 30 | transcripts up to this size use the quality model for extraction |
| `ROUTE_OUTLIER_FLOOR` | 0.4 | below this to every anchor → needs review |
| `ROUTE_CONFIDENCE_FLOOR` | 0.55 | weak best anchor → LLM adjudicates (may abstain) |
| `ROUTE_AMBIGUITY_MARGIN` | 0.05 | top-2 anchors closer than this → LLM adjudicates |
| `ROUTE_LLM_MAX` | 20 | adjudication call cap per run |
| `LLM_VERIFY_MARGIN` | 0.03 | borderline-merge window below the bar |
| `LLM_VERIFY_MAX` | 10 | verify cap and audit cap |
| `LLM_RESCUE_MARGIN` | 0.1 | singleton-rescue candidacy window below the bar |
| `LLM_RESCUE_MAX` | 10 | rescue adjudication cap per run |
| `LLM_RESCUE_MAX_GROUP` | 2 | rescue only completes PAIRS, never grows 3+ groups |
| `LLM_CONSOLIDATE_MAX` | 15 | same-ask consolidation call cap per run |
| `LEXICAL_DEDUP_THRESHOLD` | 0.9 | token-Jaccard for counting rewordings as repeats |
| `SAME_MESSAGE_REPHRASE_OVERLAP` | 0.5 | collapse bar for two rewrites of one ask |
| `OLLAMA_GENERATION_MODEL` / `OLLAMA_FAST_MODEL` | auto | quality / typing models |
| `LLM_TIMEOUT` | 180 | seconds per LLM call |
| `TAXONOMY_PATH` / `TAXONOMY` | taxonomy.json / on | bucket definitions |
| `DOMAIN_CONTEXT` | unset | appended to every prompt with a preserve-tokens clause |

## The prompts

The prompts evolve with every measured eval round (the pack version is
stamped into results metadata as `prompt_pack`), so they are not duplicated
here — **the single source of truth is `slack_question_analyzer/group_labeler.py`**,
where each prompt lives next to its JSON schema. The cast, and each one's
safe exit:

| Prompt | Model | Job | Abstain |
|---|---|---|---|
| EXTRACT | quality (small transcripts) / fast | rewrite every REAL ask standalone, tag its type | empty list |
| DETECT | fast | implicit help requests (regex-first mode only) | empty list |
| CONSOLIDATE | quality | within-message restatement pick (two-judge) | keep all |
| FEEDBACK | quality | support vs product-feedback second opinion | false (support) |
| ROUTE | quality | closed-choice bucket adjudication | 0 (review) |
| VERIFY | quality | one-doc-page same-topic test for borderline merges | false (no merge) |
| AUDIT | quality | nominate members that don't belong | empty list (keep) |
| LABEL | quality | 2-4 word topic + one-line summary | NEEDS_REVIEW |
| THEMES | quality | broad-theme roll-up (no-taxonomy fallback only) | unassigned |
| SUMMARY | quality | 2-3 sentence executive summary | none (optional) |
| ANSWERED | quality | did the thread replies answer THIS question? | unknown |

Conventions that hold across all of them: JSON-schema-enforced output,
temperature 0, fixed seed, an explicit do-not-guess safe exit FIRST, worked
examples over rules (small models follow examples, not policies), examples
always off-domain from the eval fixtures, and a code-level guard that drops
any output reproducing a few-shot example without source support.

## Regression fixtures & evaluation

After ANY prompt, anchor, or threshold change, run:

    slack-analyzer eval

With no arguments it runs EVERY fixture in `fixtures/` and exits 1 on any
miss. Two fixture types:

- **Question-level** (`field_run_2026-06-10.json`): frozen questions with
  correct buckets and groupings. Scores routing accuracy and grouping
  precision/recall, listing every mismatch, missed pair, wrong pair, and
  integrity violation.
- **Transcript-level** (`mft_synthetic_1..7.json`, `"type": "transcript"`):
  a raw transcript plus an answer key. Runs the FULL pipeline — extraction
  included — and asserts the key's headline numbers: total asks,
  per-message ask counts (collapse traps vs split controls), named
  recurrences with exact occurrence counts, genuine singletons the rescue
  pass must leave alone, exact feedback membership, verb-drift bans,
  false-merge twins, the Answered count with per-question membership, and
  occurrence integrity (every count provable with populated distinct-source
  rows; zero integrity repairs). Extraction bugs (silent drops,
  over-splitting, verb drift, feedback misrouting, reply leakage) happen
  before routing, so only this type can see them. The human-readable trap
  maps live next to them (`mft_test_answer_key*.md`). Fixture themes:
  1 = trap map (drops, scaffolding, verb drift), 2 = calibration pairs +
  feedback gate, 3 = occurrence/recurrence integrity, 4 = threads +
  Answered + tokenization stress, 5 = extraction PRECISION (a noisy
  channel where 10 of 14 messages must yield ZERO asks — fabrication
  guard), 6 = routing HUMILITY (off-topic / split / vague questions must
  reach the review pile via review_must_match, while clear questions —
  even ones quoting error strings — must route confidently via
  routed_must_match).

The topic bank is excluded from both types (transcript runs get a temp
empty bank) so the score measures the pipeline, not one machine's learned
history.

`metadata.llm_stats` tracks abstain/verdict rates per run (verify
true/false/uncertain, audit evictions, extract empty batches, label
abstains). If the rescue pass makes verify fire constantly, in-bucket
clustering is under-forming upstream — that's the real problem, not a
fuller-looking output.

## Appendix — design history by eval round

Newest first. Each entry summarizes what a measured eval round changed
and why; the prompts/stages above already reflect all of it.

> Deltas since the prompts/stages quoted below were written:
> - Eval round 14 (provenance diagnostics paid off immediately): a
>   transcript TITLE line glued to the first message inflated its source
>   past the 200-char identity cap AND '(test set 2)' matched the bare
>   digit-enumeration pattern - faking an asker-declared split, exempting
>   the message from the single-ask cap, and producing the long-mysterious
>   'enumerated' ejection. Fixed both ways: digit enumeration must appear
>   in enumeration POSITION (message start or after a colon/semicolon),
>   and a pre-date title line in the first block is stripped as file
>   furniture (same class as '# ' comments). ROUTE gains a worked
>   AI-assistant abstention example (the rule alone failed twice).
>   Deterministic impossibility feedback verified in the field (cron).
> - Eval round 13 (first group-then-route run; reorder VALIDATED:
>   fixture 7 hit its counts/ranking/precedence wins, fixtures 4/5/6
>   stayed perfect): the predicted cost appeared as cross-category
>   borderline merges the bucket walls used to block, all wearing the
>   audit-flagged-verifier-overruled signature. ROUTING IS NOW THE THIRD
>   JUDGE: a member the audit nominated and the verifier kept (judges
>   1-1) is split out when it and its group CONFIDENTLY route to
>   different buckets. ROUTE abstains on new-CAPABILITY areas no
>   category describes (not just off-topic tools). Wish + the asker's
>   own impossibility statement ('doesn't look possible today') diverts
>   to feedback deterministically. Eval failures now print row sources
>   and the full provenance trail.
> - Eval round 12 / fixture 7 (volume + ranking + emerging topic) — THE
>   REORDER: grouping now runs GLOBALLY FIRST, and each resulting
>   CLUSTER is routed to a bucket by its representative (Stage 1 and
>   Stage 2 below have swapped). Grouping used to live inside buckets,
>   downstream of routing, so identical asks that routed to different
>   buckets could never merge: a 4x recurrence caught 3, a 2x never
>   fired, and an emerging topic was scattered before its coherence
>   could be seen. Recurrence is a fact about MEANING; the bucket is
>   presentation. An unroutable CLUSTER now abstains as a unit — a
>   multi-question review cluster is the 'a category is missing' radar.
>   PRECEDENCE RULE: enumerated-split siblings ('1. ... 2. ...', 'and
>   separately') are locked separate — the asker's own split outranks
>   every collapse pass (consolidation once deleted 'max retry count'
>   as a 'rephrasing' of its enumerated sibling). Topic labels must be
>   GROUNDED: every content word of a label must occur in the group's
>   own question text, else keyword fallback ('Transfer Retries' once
>   named a group of failure-alert questions). Wish-phrasing gains
>   "doesn't look/seem possible".
> - Eval round 11 (181/184; fixtures 1, 2, 5, 6 ALL PERFECT; the
>   single-ask cap killed the 5GB dup first try): the cap's survivor is
>   now ranked against the '?'-SENTENCE (the lone question mark marks
>   the asker's actual question), not the whole message — both
>   candidates can be verbatim-supported, and the symptom rewrite once
>   beat the real ask on length. VERIFY gains a second TRUE example
>   (same end-to-end goal in different words) — it had six false
>   anchors and one true, and the genuinely-borderline onboarding pair
>   flip-flopped between rounds.
> - Eval round 10 (179/184; fixtures 1, 2, 5 PERFECT; host-key exactly
>   3x - the rescue cap worked): the single-ask cap - an UNENUMERATED
>   message containing at most one '?' asks at most one question; a
>   second distinct extraction is the model rewriting context into an
>   extra ask. Best-supported phrasing survives; identical-text repeats
>   and truncated sources are exempt. The virus-scan-vs-metering route
>   is now a DOCUMENTED convention (taxonomy comment + fixture asserts
>   the consistent route), per the answer key's sanctioned alternative.
> - Eval round 9 (177/184; fixtures 2 and 5 PERFECT): rescue only
>   completes under-grouped PAIRS (LLM_RESCUE_MAX_GROUP=2) — every
>   mega-group across nine rounds grew by rescuing a singleton into an
>   already-established 3+ group, which is itself evidence the singleton
>   differs. CONSOLIDATE gains the capability-vs-limit example ('is
>   there a max size?' / 'can it handle very large files?' = one ask).
> - Eval round 8 (first quality-extraction round; best yet): same-source
>   rows inside a group are now dispositioned by whether the MESSAGE
>   enumerates separate asks ('1. ... 2. ...', 'and separately', 'two
>   unrelated questions'): enumerated -> eject to its own row (T6 class),
>   not enumerated -> drop as a rephrase, with provenance. Deictic
>   meta-questions ('Is that the right approach, or is there a cleaner
>   pattern?') collapse as continuations of their message's other ask.
>   Rhetorical filler gains the solidarity-banter pattern ('anyone else
>   ready/excited/looking forward...'). CONSOLIDATE gains the
>   context-symptom restatement example; ROUTE gains an off-topic
>   worked example.
> - Eval round 7: small transcripts (<= EXTRACT_QUALITY_MAX, default 30
>   messages) hand PRIMARY extraction to the quality model — extraction
>   is the hardest open-ended job and seven rounds of 3B wobble say so.
>   Content-free rhetorical filler ('Anyone seen this before?') is
>   dropped in CODE (the prompt's own list, enforced; the two-judge
>   consolidation once protected one). A question LEADING with a
>   restatement marker ('I mean...', 'Basically...') collapses with its
>   message's other ask regardless of lexical overlap. The Kafka
>   half-loss root cause: 'Quick one - does X...' hid its question word
>   behind the opener, so regex counted 1 while the questions differed —
>   greetings are now stripped BEFORE the question test and the opener
>   list covers conversational prefixes. ROUTE gains the product-scope
>   rule; EXTRACT forbids inventing subjects ('Is there a limit?' must
>   stay subjectless); VERIFY gains the same-action-different-object
>   example.
> - Eval round 6: route adjudication shows abstain as a LISTED option
>   ('0 NONE OF THESE...') — given only real categories a small model
>   picks from the list every time, so off-topic questions were
>   force-routed with the abstain rule sitting unused. Extraction
>   few-shot gains DO-NOTs for log pastes and social banter (rules alone
>   don't bind a 3B), and the venting-vs-symptom contrast is stated as
>   one test with OFF-DOMAIN examples (two fixture-verbatim phrases that
>   had crept into the prompt were removed).
> - Eval round 5: the rescue-audit tie rule — a rescue is one verifier
>   YES on a borderline add; if the audit then flags that member, the
>   judges are 1-1 and the rescue is UNDONE (no verifier overrule round;
>   that loop built three mega-groups). Rescue nearness is now AVERAGE
>   similarity to the group (the clustering metric), not max. The
>   recovery regex fallback considers every batch message so a half-lost
>   two-part ask gets its missing '?' half back. is_answered receives the
>   thread's first message so replies that answer by number ('for #2,
>   yes') are resolvable. original_message is one canonical string
>   (cleaned + collapsed + capped) on every path — it is the message's
>   identity for collapse/ejection/integrity.
> - Eval round 4 (fixtures 5/6 baselines): routing gains a CONFIDENCE
>   floor (best anchor < 0.55 -> the closed LLM choice with abstain, so
>   off-topic/vague questions reach review instead of the closest wrong
>   bucket; ambiguity margin widened to 0.05). The recovery pass no
>   longer restores question-shaped statements over an explicit
>   quality-model "no ask" — only a literal '?' overrules two models.
>   Extraction prompt: venting without a symptom, log pastes with no
>   request, and social banter yield NOTHING; never rewrite a statement
>   into a question. Rephrase-collapse tokens get light suffix folding
>   (fails/failed/transfers share a stem) so reworded restatements
>   can't survive as fake 2x groups.
> - Eval round 3: feedback diversion is gated on DETERMINISTIC wish
>   phrasing in the source message — without it a question stays in
>   support no matter what any model says (the 8B had diverted plain
>   capability questions, killing a recurrence and the Answered count);
>   wish + an explicit "feature request"/"product feedback" label diverts
>   with no LLM at all; wish alone goes to the conservative confirmer.
>   The 3B's feature-request tag no longer gates anything. 'and
>   separately' splits compound sentences into distinct ask candidates so
>   the under-extraction safety net can count them. VERIFY gains the
>   credential-lifecycle vs identity-verification example.
> - Eval round 2: same-source occurrences inside a group are EJECTED to
>   their own singleton row, never deleted (a wrong eject = one extra
>   unique; a wrong delete = a silent drop). The extraction REAL
>   definition now includes capability wishes (tag feature-request) and
>   stuck-problems; the safety net also re-checks any wordy message that
>   produced ZERO asks with the quality model (regex can't see implicit
>   asks, so the fewer-than-regex trigger never fired for them).
>   Taxonomy v3: partner onboarding/provisioning language moved to
>   Install, Upgrade & Admin so setup questions stop sharing a bucket
>   with host-key/credential questions.
> - Prompt pack 9 (first eval round across all 4 transcript fixtures):
>   extraction gains the or-alternative rule (an 'Or...?' offering another
>   route to the same goal is ONE ask) and the explicit multi-part rule
>   ('and separately' / numbered unrelated requests are DISTINCT asks);
>   consolidation gains restatement cues ('Basically', 'I mean', forwarded
>   quote + paraphrase = one ask) and a different-outcomes guard; feedback
>   confirmation treats an explicit 'feature request'/'product feedback'
>   label in the source as decisive; verify gains the workflow-stage rule
>   (setting X up vs configuring one property of X = different topics).
> - Example-leak guard: an extraction that reproduces a few-shot example
>   question without strong textual support in its claimed source message
>   is prompt contamination — dropped and counted (extract_example_leaks).
> - Lexical rephrase-collapse counts CONTENT words only (>3 chars):
>   template boilerplate is zero same-ask evidence; gray zone falls through
>   to two-judge LLM consolidation. When a collapse fires, the survivor is
>   the best-SOURCE-SUPPORTED phrasing, not the first-seen one.
> - Plain-text transcripts: '>'-quoted lines are thread replies (attached
>   for answer detection, never extracted as questions); '# ' headings are
>   structural markup.
> - Two-judge rule for every DESTRUCTIVE action: audit evictions and
>   same-ask consolidation drops need independent verifier agreement.
> - Source-support invariant (extractions must be contained in their
>   claimed message; reassign or drop), date-integrity invariant, and the
>   exit invariant: a group may only render a count it can prove with rows
>   (2+ distinct sources for any 2+ count); totals derive from rendered rows.
> - Same-ask consolidation (within-message, quality model) and the
>   confirmed-only feedback lane (feature-request tag + intent-aware 8B
>   confirmation using the original message).
> - Type-family merge veto (capability never LLM-merges with breakage).
> - Provenance: results['dropped_questions'] records every removal with a
>   reason; results metadata carries app_version, prompt_pack, taxonomy
>   version, routing health, and LLM verdict rates.
> - Cancellation is checked before every LLM call, not just at stage
>   boundaries.
