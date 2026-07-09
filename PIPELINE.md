# The Question Funnel — Pipeline Spec (v2.60.3, prompt pack 26, taxonomy v5)

> The stages below describe the CURRENT architecture. The change-by-change
> history of how it got here (14 measured eval rounds through v2.40, plus
> the v2.41-v2.45 rounds summarized at the appendix top) is preserved in
> the [appendix](#appendix--design-history-by-eval-round) at the bottom.

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
   text, CSV, or text copied straight out of the Slack app — author +
   timestamp line pairs; an '<n> replies' divider marks a THREAD copy,
   whose first message becomes the ask and the rest its replies). Markup, code blocks, mentions stripped. In plain text,
   '>'-quoted lines are thread replies: attached to the parent for answer
   detection, never extracted as questions (a responder's clarifying
   question is not an ask); '# ' heading lines are structural markup.
2. **Announcement immunity** (code, no AI; `ANNOUNCEMENT_FILTER=on`):
   broadcast posts (webinar promos, sales-kit launches, win wires) and
   "please note" resource notices are CONTEXT, never question sources —
   left in, they yield phantoms (rhetorical headers, schedule lines,
   marketing copy inverted into asks) and their vocabulary bleeds into
   rewrites of real questions in the same batch. Detection is
   precision-first: any help-seeking phrasing vetoes outright; marketing
   shape needs a content signal (CTA / marketing phrasing / cc:-broadcast)
   plus one more signal — emoji density and bullet lists alone never
   qualify, because real bug reports use both. Skips are counted
   (`announcements_skipped`) and surfaced in the dashboard's extraction
   notes. Motivated by the 2026-07-07 field run (its fixture encodes
   every observed phantom).
3. **LLM-first extraction** (transcripts ≤ 150 messages; batches of 8):
   the fast model classifies each sentence REAL / RHETORICAL / CONTEXT,
   rewrites every REAL ask as a standalone question, preserves the asker's
   intent verb (HANDLE ≠ BYPASS), and tags each question with its TYPE —
   the second axis, independent of subject: how-to / troubleshooting /
   is-it-possible / feature-request / defect-report (prompt: EXTRACT, below).
4. Safety nets, in order:
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
   **Subject coherence** (v2.57): a claimed member sharing not one
   distinctive subject word with the rest of its claim is released back
   to clustering — a centroid blended from a past over-merge sits between
   several unrelated asks and would otherwise re-claim them as one group
   in every future analysis, poison that bypasses every downstream guard.
3. **Average-link clustering** at a FIXED bar `IN_BUCKET_THRESHOLD=0.8`
   (a user-pinned `SIMILARITY_THRESHOLD` overrides it). The LLM gates below
   guard every borderline merge, so the adaptive noise gate stays out of
   the way; it still applies when no taxonomy is configured. **Subject
   gate** (v2.49): a join that clears the bar only narrowly (within
   `LLM_VERIFY_MARGIN` above it) while sharing almost no DISTINCTIVE
   subject words with the group (`MERGE_SUBJECT_MIN=0.25`;
   corpus-common tokens like the product name carry no evidence — same
   principle as the keyword scorer) is DEFERRED to the borderline
   verifier instead of template-merged ('Does wM MFT support X?'
   scaffolding inflates cosine between unrelated asks). Only active when
   a verifier is available — without one, deferring would silently
   under-merge. 0 disables.
4. **Borderline merge pass**: cluster pairs whose best cross-similarity is
   within `LLM_VERIFY_MARGIN=0.03` of the bar are judged by the quality
   model (prompt: VERIFY; conservative, doubt → false; numeric guard: the
   merged cluster's average must stay ≥ bar − margin; cap `LLM_VERIFY_MAX=10`).
   **Subject guard** (v2.56): a cosine pair sharing NOT ONE distinctive
   subject word (df-aware, floor 4; raw tokens under 8 buckets) keeps its
   verifier hearing but loses the margin discount — the merged average
   must clear the FULL bar. The verifier is fooled by the same
   product-name scaffolding that inflates the cosine (field finding:
   a supported-token check, a vault-integration ask, and an access error
   merged at avg 0.77); genuine rewordings share at least the head noun.
   A SECOND candidate source widens what the verifier gets to see: cluster
   pairs sharing most of their distinctive content words
   (`LLM_VERIFY_LEXICAL_MIN=0.7`, the SAME prefix-tolerant overlap metric
   as the veto override, with their own `LLM_VERIFY_LEXICAL_MAX=5` budget)
   are candidates even when the embedding model underscores them (looser
   numeric floor: bar − `LLM_VERIFY_LEXICAL_SLACK=0.15`). The type-family veto (capability never
   merges with breakage) downgrades to "ask the verifier" when both clusters
   are about the SAME named subject (`SHARED_SUBJECT_MIN=0.5` overlap) —
   "raise the sftp timeout" answers both the how-to and the symptom.
5. **Singleton rescue**: a singleton near an existing group (average-link)
   is adjudicated by the verifier against that nearest group only. Pairs
   use the generous `LLM_RESCUE_MARGIN=0.1` window; 3+ groups (up to
   `LLM_RESCUE_MAX_GROUP=4`, live-size-checked) use the tighter
   `LLM_RESCUE_MARGIN_LARGE=0.05` — an established group that didn't catch
   the singleton during clustering is itself evidence it differs, and
   unbounded rescue into big groups is how mega-groups grew. Every rescue
   stays undoable by the audit (judges tied → undo). Doubt/failure → stays
   a singleton. **Subject guard** (v2.56, same finding as the borderline
   guard): rescue exists for REWORDINGS, and rewordings share at least one
   distinctive subject word — a singleton sharing none with its nearest
   group never reaches the verifier.
6. **Group audit (final QC, two-judge + tiebreak)**: every formed group is
   checked by the quality model (prompt: AUDIT; doubt → keep). Eviction is
   destructive, so the auditor only nominates: the verifier must confirm
   (explicit false) — except a RESCUED member, where an audit flag ties the
   judges 1-1 and the rescue is simply undone. When the verifier overrules
   an audit flag, ROUTING acts as the third judge — with two rules that
   both use MUTUAL BUCKET PREFERENCE (each side prefers its own best
   anchor over the other side's by the ambiguity margin, both above the
   outlier floor; global routing confidence is the wrong test, because a
   mixed group's representative always sits near some third anchor):
   a flagged MINORITY of 2+ members routes as a unit and splits out as
   its own group with counts intact; a single flagged member splits out
   as a singleton.
7. Representative question = the member closest to the group centroid
   (extractive, never generated). Keywords = group word frequency × inverse
   frequency in the REST of the corpus (corpus-wide words score zero).

## Stage 2 — Route each cluster into a bucket (embeddings + closed-choice LLM)

Two deterministic pre-filters run before any budget is spent (v2.49):
**status/action requests** ('can someone check on <url>?' — a request
phrase + a bare identifier + almost no other content words, checked on
the representative AND its source message) go straight to review, since
they ask a person to act, not a question a category answers; and
**distinctive-token tiebreaks** settle margin ambiguity without the LLM
when the representative mentions a token unique to the embedding
favorite's anchor and none unique to the runner-up (token sets computed
from whatever taxonomy the user loaded — data, not code).

`taxonomy.json` defines 9 buckets (v4 added **AI & Automation** when a
recurring AI-agent capability area kept force-routing into wrong homes —
confident singletons are invisible to both the coherence gate and the
review radar, so a recurring new capability area gets a bucket), each
with an `anchor` paragraph and a
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
  single-number choice (prompt: ROUTE, below; cap `ROUTE_LLM_MAX=40`), with
  "0 NONE OF THESE" presented as a listed option. Reply `0` = honest
  abstain → needs review. The budget is spent demotions-first, then
  biggest clusters first (a wrong route on a multi-question cluster
  mis-shelves N questions). Past the cap, the fallback respects WHY the
  cluster was ambiguous: a margin case (two strong anchors) takes the
  embedding favorite, but a confidence-floor case (weak best anchor) goes
  to review — force-routing it is exactly what the floor exists to
  prevent, and a saturated cap used to make the review pile structurally
  unreachable (2026-07-07 field finding).
- **Cluster-coherence gate** (`ROUTE_MEMBER_COHERENCE=on`): after the
  representative routes confidently, every MEMBER of a multi-question
  cluster votes its own nearest anchor. Members splitting with no majority,
  or outvoting the representative's bucket, demote the route to the LLM's
  closed choice — where abstain sends the whole cluster to review. A
  coherent cluster whose members smear across buckets is the "no bucket
  owns this" signal the radar wants, not a routing coin-flip. The gate
  only runs when an adjudicator is available, demotions are adjudicated
  before other ambiguities, and the original bucket is always the first
  candidate — past the budget (or on LLM failure) the route stays put.
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

> The adaptive noise gate, threshold auto-adjust, and the THEMES LLM call
> only execute when **no taxonomy is configured** — a default deployment
> (taxonomy.json present) pins a fixed bar and derives themes from the
> bucket merge map instead.

## Embedding model choice (measured A/B, July 2026)

`nomic-embed-text` is the calibrated default (grouping bar 0.80). A/B
against `mxbai-embed-large` on the full fixture suite:

| Model / bar | Score | Failure profile |
|---|---|---|
| nomic @ 0.80 | **247/253** | a handful of verifier judgment calls on same-domain pairs |
| mxbai @ 0.80 | 239/253 | fixes the judgment over-merges, but its lower cosine scale under-merges wanted paraphrase pairs at nomic-calibrated bars |
| mxbai @ 0.75 | 243/253 | recovers some pairs (fixture 3 perfect for the first time), but distant paraphrases stay unreachable and fixture 2's over-merge re-forms — no bar dominates |

Takeaway: mxbai separates same-domain topics better but scores true
paraphrases too low for threshold-based recall; nomic + the LLM judge
stack wins overall. Re-test when trying a new embedding model: pull it,
set `OLLAMA_MODEL`, run `eval --compare` (caches are keyed by model, so
the experiment is fully reversible).

## Pipeline knobs (.env)

> This table covers the GROUPING/ROUTING pipeline. Feature switches
> (`GROUP_LABELS`, `LLM_EXTRACTION`, `LLM_ANSWER_DETECTION`,
> `EXECUTIVE_SUMMARY`, `THEMES`), infrastructure (`OLLAMA_URL`,
> `OLLAMA_MODEL`, `OLLAMA_GENERATION_MODEL`, cache dirs/caps,
> `EMBEDDING_PREFLIGHT`, `LARGE_CLUSTERING_THRESHOLD`), and server vars
> (`API_HOST`/`API_PORT`, `MAX_CONCURRENT_JOBS`, `MAX_SAVED_ANALYSES`,
> `ANALYSES_DIR`/`JOBS_DIR`, `NO_BROWSER`) are documented in
> `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `SIMILARITY_THRESHOLD` | unset (auto) | pinning it overrides every bar, including the global one |
| `IN_BUCKET_THRESHOLD` | 0.8 | fixed GLOBAL grouping bar (name predates the reorder) |
| `BANK_MATCH_THRESHOLD` | 0.85 | question/group → known-category match floor |
| `EXTRACT_QUALITY_MAX` | 30 | transcripts up to this size use the quality model for extraction |
| `ROUTE_OUTLIER_FLOOR` | 0.4 | below this to every anchor → needs review |
| `ROUTE_CONFIDENCE_FLOOR` | 0.55 | weak best anchor → LLM adjudicates (may abstain) |
| `ROUTE_AMBIGUITY_MARGIN` | 0.05 | top-2 anchors closer than this → LLM adjudicates |
| `ROUTE_LLM_MAX` | 40 | adjudication call cap per run (spent demotions-first, then biggest clusters) |
| `LLM_VERIFY_MARGIN` | 0.03 | borderline-merge window below the bar |
| `LLM_VERIFY_MAX` | 10 | verify cap and audit cap |
| `LLM_VERIFY_LEXICAL_MIN` | 0.7 | shared-subject overlap (same prefix-tolerant metric as the veto override) that also surfaces a pair for verification (>1.0 disables) |
| `LLM_VERIFY_LEXICAL_MAX` | 5 | separate candidate budget for lexical pairs (they'd be truncated inside the cosine cap) |
| `LLM_VERIFY_LEXICAL_SLACK` | 0.15 | numeric tightness floor for lexical candidates (bar − slack) |
| `SHARED_SUBJECT_MIN` | 0.5 | same-subject overlap that downgrades the cross-type veto to "ask the verifier" (>1.0 disables) |
| `LLM_RESCUE_MARGIN` | 0.1 | singleton-rescue window below the bar (pair targets) |
| `LLM_RESCUE_MARGIN_LARGE` | 0.05 | tighter rescue window for 3+ targets |
| `LLM_RESCUE_MAX` | 10 | rescue adjudication cap per run |
| `LLM_AUDIT_MAX` | =verify | audit cap per run (own budget since v2.43) |
| `LLM_AUDIT_NOMINEE_MAX` | 3 | overrule-verify fan-out cap per audited group |
| `LLM_FEEDBACK_MAX` | 15 | feature-request confirm cap (overflow stays in support) |
| `LLM_CONSOLIDATE_CONFIRM_MAX` | 3 | same-ask confirm fan-out cap per message |
| `LLM_RESCUE_MAX_GROUP` | 4 | largest group rescue may complete (live-checked) |
| `ROUTE_MEMBER_COHERENCE` | on | member votes can demote an incoherent cluster's route |
| `LLM_CONSOLIDATE_MAX` | 15 | same-ask consolidation call cap per run |
| `LEXICAL_DEDUP_THRESHOLD` | 0.9 | token-Jaccard for counting rewordings as repeats |
| `SAME_MESSAGE_REPHRASE_OVERLAP` | 0.5 | collapse bar for two rewrites of one ask |
| `MERGE_SUBJECT_MIN` | 0.25 | narrow above-bar joins with less subject overlap than this defer to the verifier (0 = off) |
| `ANNOUNCEMENT_FILTER` | on | broadcast/notice messages are context-only, never question sources |
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
  routed_must_match), 7 = volume + exact recurrence counts + the
  emerging-topic radar (the fixture that drove the group-then-route
  reorder and the AI & Automation bucket),
  field_run_2026-07-07 = the real July 2026 channel dump — announcement
  immunity (broadcast promos and notices must yield ZERO asks and never
  contaminate rewrites), one-answer consolidation calls on messy
  multi-sentence messages, and anti-false-positive guards (an emoji-heavy
  bug report and a broadcast-adjacent real question must both survive).

The topic bank is excluded from both types (transcript runs get a temp
empty bank) so the score measures the pipeline, not one machine's learned
history.

`metadata.llm_stats` tracks abstain/verdict rates per run (verify
true/false/uncertain, audit evictions, extract empty batches, label
abstains). If the rescue pass makes verify fire constantly, in-bucket
clustering is under-forming upstream — that's the real problem, not a
fuller-looking output.

## Appendix — design history by eval round

> **v2.60.1 (integration-audit fixes):** a two-agent integration audit
> plus a full-browser smoke of everything working together. Verified
> clean: week semantics agree across weekly stats, topic history, and
> the dashboard; every field the UI reads exists in the API responses
> with the right name and type; the whole analyze-to-export flow runs
> green end to end; zero console errors across views, modals, and
> themes. Fixed: renaming a topic in the Learned Topics modal now
> patches the loaded analysis and remounts the views on close (the
> dashboard used to show the old name until re-analysis); week
> navigation keeps the page on screen and dims it while loading instead
> of tearing down the chart mid-click; chart week-navigation works by
> touch (index from click coordinates, not hover state) and keyboard
> (arrow keys + Enter, with an aria label); an empty past week no
> longer claims to be the 'first week of data'; weekly stats, CSV
> export, and the analyses list use defensive key access like the
> markdown exporters (legacy files degrade, never 500); Week view
> singleton rows drop the 'Avg. similarity —' filler.

> **v2.60 (calendar weeks + week navigation):** Week in Review switched
> from 7-day windows counted back from the newest message to true
> CALENDAR weeks (Monday through Sunday) — the same bucketing the
> topic-history chart already used, so the two views finally agree on
> what "a week" is. The weekly endpoint takes ?week=YYYY-MM-DD (any date
> inside the wanted week); compute_weekly_stats(week=...) anchors the
> trend, rankings, movement, and feedback pulse to the selected week,
> and returns week/latestWeek/trendWeeks navigation metadata. Chart dots
> are clickable (AreaChart onPointClick) and jump to that week; the
> Latest-week chip becomes a working back-to-latest button; a selected
> week with zero questions renders an inline message instead of the
> global empty state. Trailing feature requests (dated after the newest
> question) count toward the latest week only — they no longer leak into
> past weeks. Also fixed: the header toggle rendered 'Weekin Review'
> (an inline-flex button strips a flex item's leading space at its line
> start; NBSP survives).

> **v2.59 (deep-sweep fixes, pack 26):** a second audit pass over the
> modules earlier audits skipped. Two demonstrated defects: (1) a ReDoS
> in the status-request target regex — 'server1' followed by a dashed
> rule sent the host-like alternative into exponential backtracking (21s
> at 42 dashes), hanging the analysis worker on a common paste shape;
> the nested-hyphen ambiguity is gone and a timing regression test
> guards it. (2) The DETECT pass was silently fed the full EXTRACTION
> few-shot: DETECT_SYSTEM shares its first 40 characters with
> EXTRACT_SYSTEM and the routing sniffed the prompt prefix — an explicit
> few_shot flag replaces the sniffing (pack 26; detect verdicts re-judge
> once against the corrected prompt). Also: month-name date recognition
> anchored to real months (restoring the recognized-implies-parseable
> invariant), eval results carry the real fixture path, yearless copied
> dates never land in the future, integer env vars tolerate typos, a bad
> SIMILARITY_THRESHOLD no longer 500s /api/config, job recovery takes a
> cross-process leader lock, and the topic-history cache evicts deleted
> analyses. UI: the settings slider is visible in dark mode, range
> inputs keep a single focus affordance, disabled buttons use the
> semantic disabled surface, and a component or JSX load failure shows a
> visible error banner instead of a silent blank page.

> **v2.58.1 (pre-ship audit fixes):** the Slack-copy detector was
> hijacking pasted incident notes and service logs ('label / clock time /
> body' is not unique to Slack): seconds-precision times no longer match
> (Slack stamps are HH:MM), and a bare 24-hour time only counts as a
> boundary when the line above looks like a real name (multi-word, no
> digits) — AM/PM and day-prefixed stamps still accept single-word
> display names. Bank-claim coherence upgraded from member-vs-rest to
> CONNECTED COMPONENTS over pairwise shared-subject edges: a centroid
> blended from two topics claims two coherent halves that the old test
> never released; components split them into separate claims. UI: single
> keyboard focus ring on buttons (the component's inline ring stacked
> with the new global outline), Week in Review reflows at split width,
> search-field focus rings the field not the inner input, placeholders
> use the theme token, the header toggle collapses to 'Week' on narrow
> screens, and the bundle's drift-detection hashes were regenerated.

> **v2.58 (paste-a-thread + responsive UI):** a fourth input format —
> text copied straight out of the Slack app (author + timestamp-only line
> pairs), with Slack-style dates ('Jun 9th at 2:30 PM', 'Today at 9:15',
> day dividers) normalized to parseable dates (year defaults to the
> current one, as Slack omits it). An '<n> replies' divider after the
> first message marks a THREAD copy: root becomes the ask, the rest its
> replies, feeding answer detection; without it each message stands
> alone. The upload modal gained a paste box wired to the existing
> content API. UI: the dashboard reflows for split-screen widths
> (responsive classes over the inline base styles), button labels are
> centered (the Carbon asymmetric padding read as off-center text), the
> header collapses to icons under 720px, and every control shows a
> keyboard focus ring.

> **v2.57 (bank-claim poisoning fix):** the v2.56 guards closed both LLM
> merge paths, and the very next field run re-formed the same bad group
> anyway — through the topic BANK. Earlier analyses had recorded the
> over-merged group, so the bank entry's centroid is a blend of three
> unrelated asks; each scores ≥0.85 against that blend, and bank claims
> run BEFORE clustering and skip clustering, borderline verify, and
> rescue entirely. Self-sustaining poison: every re-analysis re-recorded
> the group and re-hardened the centroid. Fix: claimed groups get the
> same zero-shared-distinctive-subject-word test as the other guards —
> an incoherent member is released back to normal clustering (logged),
> and a claim reduced below 2 members dissolves. Existing poisoned
> entries stop claiming; deleting them from Learned Topics cleans up
> their name and history.

> **v2.56 (over-merge fix, pack 25):** the first post-migration field run
> re-exposed template merging one layer up: a supported-token check, a
> vault-integration ask, and an access error merged into one 3x group at
> avg 0.77 — below the bar, approved by the VERIFIER via the borderline
> margin discount. The clustering subject gate (v2.49) defers template
> joins to the verifier, but the verifier reads the same product-name
> scaffolding the embeddings do. Deterministic fix, both LLM merge paths:
> a cosine borderline pair sharing NOT ONE distinctive subject word
> (df-aware with floor 4, raw tokens under 8 buckets — three questions
> about threads must not make 'thread' corpus-common) loses the margin
> discount and must clear the full bar; a rescue candidate sharing none
> with its nearest group is skipped before spending a verifier call.
> Zero-shared-token is the right test, not an overlap ratio: genuine
> rewordings can share exactly one head noun ('thread'). Prompts (pack
> 25): VERIFY gains the supported-feature-vs-integration-workaround FALSE
> example; CONSOLIDATE gains the candidate-inside-a-request-for-
> alternatives example ('Can HashiCorp Agents be leveraged here?' is a
> facet of the vault question, not a second ask).

> **v2.55 (repo upgrade round):** two-agent documentation-vs-code audit of
> the whole repo. Code: `taxonomy.json` and `seed_topics.json` now resolve
> through a repo-root fallback (`default_data_path`) when the current
> directory has no copy — running the CLI or server from another directory
> no longer silently loses routing buckets or seed topics (an explicit
> `TAXONOMY_PATH`/`SEED_TOPICS_PATH` still wins, and a cwd copy still
> shadows the shipped one). Docs: README's model-download size split by
> machine RAM, fixture wording made precise (eight transcripts + one
> question-level set), stale design-system readme claims fixed (the
> dashboard IS this design system served by Flask; fonts and Lucide are
> vendored, not CDN-loaded), `.env.example` notes AI_PROVIDER is
> intentionally ollama-only, and ignore files use a generic hidden-dir
> pattern.

> **v2.54 (stale-answer nudge):** curated answers can rot — a topic that
> keeps getting NEWLY answered asks after the approved wording was saved
> probably has fixes in those newer thread replies that the doc lacks.
> The history endpoint counts unique answered occurrences dated after
> `answer_updated` (only when a curated answer exists) and returns an
> `answer_stale` nudge; the topics modal renders it as a warning line,
> and the FAQ export prints the same note under the curated answer, above
> the source replies that contain the potential updates. Deterministic,
> reuses the occurrence index (which now carries an answered flag).

> **v2.53 (FAQ v3: closing the publish loop):** the FAQ feature's job is a
> loop — questions recur, you publish a doc, asks fall — and this round
> makes the app answer the last step instead of leaving it to eyeballing.
> The history endpoint computes weekly ask rate before vs. after the
> publish date (baseline window up to 8 weeks, anchored to DATA time — the
> newest dated question anywhere in the corpus, never the server clock, so
> a topic VANISHING from later uploads counts as the success it is) and
> returns a verdict: working (asks fell ≥50%), helping (≥15%),
> not_working, too_early (<2 weeks of post-publish data), or no_baseline;
> the topics modal renders it as a plain-language line. Curated answers:
> a human-approved answer saved on the bank entry becomes canonical —
> every FAQ export uses it instead of re-drafting (retroactively, via the
> bank map, so old analyses export the approved wording too), the
> drafting pass skips those groups (no wasted LLM call), and merges carry
> the answer (target's own wins). The export's "needs an answer" list is
> now ranked by recency + frequency against the data's newest date, with
> per-line evidence — frequent AND recent is where a doc saves the most
> support time.

> **v2.52 (round A: the bug-fix sweep of the v2.48–v2.51 additions):**
> topic merges now carry the FAQ-published flag (latest date wins) and
> keep the absorbed topic's id as a `merged_ids` alias, so the history
> chart still finds volume recorded under the old id in older saved
> analyses; occurrence fingerprints dedupe identical text+date pairs and
> full-overlap re-uploads no longer bump `last_seen`; seeds (recorded
> with count 0) blend their curated centroid with the first real sighting
> instead of being overwritten; bank mutations are serialized behind a
> lock and the history endpoint caches per-analysis occurrence indexes by
> file mtime (no more re-parsing every saved analysis per click); the
> grounding check stops false-rejecting formatting artifacts ('1,000' vs
> '1000', 'e.g.', 'built-in') while still rejecting invented numbers and
> identifiers; the status-request gate reads the REPRESENTATIVE member's
> source message (the medoid is not always questions[0]); FAQ drafting
> reports its own 'drafting' progress stage; multiple zips in one upload
> share a single decompression budget; legacy analyses missing newer
> metadata keys export instead of 500ing; and the dashboard's unanswered
> lane now respects the date scope, theme chip, and search filter, with
> single-week topic history charts rendering instead of dividing by zero.

> **v2.51 (FAQ v2, pack 24):** the draft FAQ writes its own first draft —
> a new FAQ_DRAFT prompt condenses each top group's CONFIRMED thread
> replies into a 2-4 sentence answer, summarization with receipts: the
> prompt forbids outside facts and a deterministic grounding check
> (numbers and identifier tokens must appear in the replies verbatim,
> content stems ≥75%) rejects drafts that smuggle anything in — rejected
> drafts fall back to quoting the raw replies. Replies are ranked by the
> gratitude signal (the reply before "thanks, that worked" is the fix),
> answered singletons get an FAQ appendix, and topics can be marked "FAQ
> published" — the date becomes a marker on the topic-history chart, so
> the user can watch ask-volume fall after their doc went live.

> **v2.49 (round 2: the accuracy package, pack 23, taxonomy v5):**
> deterministic — subject-overlap tokens are now STEMMED like every other
> content metric ('timing'/'timeout' finally overlap); filler '?'s
> ('anyone have any thoughts?') no longer defeat the single-ask cap;
> bare-'so' declarative confirmations are restatement markers;
> status/action requests pre-route to review; distinctive anchor tokens
> break margin ties without the LLM; the subject gate defers narrow
> template-merges to the verifier; ROUTE_LLM_MAX 20→40. Prompts (pack
> 23) — the same-ask second judge now SEES the original message (it was
> overruling correct folds while blind to the antecedent of 'here'),
> CONSOLIDATE gained a trailing-continuation example, VERIFY gained the
> symptom-vs-fix TRUE example PIPELINE.md always claimed, ROUTE gained a
> status-request abstain example. Taxonomy v5 extends the Errors anchor
> with wrong/impossible-value language (negative transfer speed class).

> **v2.48 (round-1 fixes from the four-way improvement audit):** routing's
> cap-saturation fallback now honors the humility gates (floor-ambiguous →
> review, never force-routed; budget spent biggest-clusters-first); the
> topic bank fingerprints occurrences (text+date) so overlapping
> re-uploads — the natural "export the last 90 days monthly" workflow —
> no longer inflate question counts or the recurring-xN badge; Slack
> rich-text blocks keep emoji shortcodes and list bullets so the
> announcement filter sees the same signals as in plain text; Ollama's
> own error sentence (e.g. "model requires more system memory") survives
> to the user instead of a bare 500; the quality model's warm-up is
> deferred past fast-model extraction (no more mutual eviction on
> 12-16GB machines; keep_alive 10m→30m); and the verify/rescue/audit and
> routing phases report per-call progress instead of freezing the bar.

> **v2.47 (fixture 9, the 2026-07-07 field run):** the first live field
> dump graded against a human answer key exposed one narrow failure
> class — broadcast announcements as phantom sources ("Why should
> customers care?", "When: July 8", marketing copy inverted into asks,
> plus a "What's New *" contamination prefix on a real question).
> Announcement immunity shipped as a deterministic pre-extraction gate
> (help-seeking veto, content-signal requirement, notice rule), verified
> to flag exactly the 5 broadcast/notice messages and none of the 47
> real ones. The filler net also learned "anyone have any thoughts?"
> and "any other ideas?" variants.

> **v2.41–v2.45 (measured rounds 1–6 of the unified scoreboard,
> 239→247/253):** eval gained the scoreboard/--json/--compare/--runs
> harness; four accuracy mechanisms shipped (rescue into 3+ groups,
> shared-subject veto override, lexical verify candidates, cluster-
> coherence gate); the one-answer convention replaced one-doc-page as
> the verifier yardstick (pack 22); taxonomy v4 added AI & Automation;
> the routing tiebreak learned minority-unit splits and mutual bucket
> preference (killed the 7x mega-group); ranking gained the recency
> tiebreak; an embedding A/B kept nomic over mxbai-embed-large (247 vs
> 243). Rounds below are the original 14 that shaped v2.28→v2.40.

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
