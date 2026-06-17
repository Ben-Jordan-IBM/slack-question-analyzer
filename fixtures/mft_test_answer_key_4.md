# Test Transcript 4 — Answer Key & Trap Map (Threads + Answered)

Purpose: exercise the **Answered metric** for the first time (it has only ever shown "—"), verify **replies are not mis-extracted as questions**, and stress **abbreviation/tokenization** splitting (the next class of string bugs after "e.g.").

**8 threads → 9 distinct asks (parent messages only; replies are NOT questions).**

> **Threshold finding:** if `Answered` still shows "—" after running this, the parser does not associate replies with parents yet. That is the gap this fixture reveals — a real result, not a pass.

---

## Headline numbers a correct run should produce

| Metric | Correct value | Notes |
|---|---|---|
| **Total asks** | **9** | replies must NOT inflate this |
| **Recurring topics** | **0** | all distinct; recurrence must not fire, and must not invent groups from reply text |
| **Answered** | **3** | of 9 — see breakdown |
| **Product feedback** | **0** | none here; the section/metric should render cleanly at zero |

---

## Answered metric — the core test

A question is **answered** only if a reply contains specific, actionable information (a setting, command, yes/no with reason, or exact doc link). Acknowledgments and promises do NOT count.

| Thread | Date | Question | Reply type | Answered? |
|---|---|---|---|---|
| T1 | Jun 9 | How to increase SFTP connection timeout? | specific property + value + action | **YES** |
| T2 | Jun 8 | Throttle bandwidth per partner? | "let me check and get back to you" | NO (promise) |
| T3 | Jun 7 | SharePoint Online as destination? | (no reply) | NO |
| T4 | Jun 5 | Validate integrity with SHA-256? | "Yes" + how + doc link | **YES** |
| T5 | Jun 4 | Why intermittent failures to partner X? | a clarifying question back | NO (no answer given) |
| T6a | Jun 3 | Per-folder retention policies? | "not sure, I'll check" | NO |
| T6b | Jun 3 | Auto-purge files older than N days? | specific mechanism (cleanup scheduler) | **YES** |
| T7 | Jun 2 | Max single-transfer file size (~5GB)? | (no reply) | NO |
| T8 | Jun 1 | 403 after upgrade to 10.15 CF19? | (no reply) | NO |

**Answered = 3** (T1, T4, T6b). The two hard cases: **T5** (the reply is a clarifying *question*, not an answer → NO) and **T6** (answered is **per-question** — T6b yes, T6a no — not per-thread).

---

## Reply-not-a-question trap (must NOT inflate the count)

Replies contain question-shaped and info-shaped text that must never become logged questions:
- **T5 reply:** "Are they failing around the same time of day?" — a responder's clarifying question. Must NOT be extracted as a separate ask.
- **T1/T4/T6 replies:** contain settings, "yes", doc links. Must NOT be extracted as questions.

**Assertion: total logged questions = 9.** If it's higher, reply text is leaking into extraction.

---

## Multi-part-in-a-thread (T6)

T6's parent has two distinct asks → **split into 2** (per-folder retention; auto-purge old files). Each carries its own answered status (a → NO, b → YES). Tests that splitting still works inside a thread and that Answered is tracked per-question.

---

## Tokenization / abbreviation stress (must each stay ONE whole question)

| Thread | The hazard | Correct result |
|---|---|---|
| T7 | "10.15.2", "approx.", "i.e.", "e.g." all in one sentence | ONE intact question about max file size (~5GB). Must NOT fragment on any of those periods |
| T8 | a URL (`https://partner.example.com/inbound`) and a version ("10.15 CF19") | ONE intact question about the 403 after upgrade. Must NOT fragment on the URL dots or the version |

These are the successors to the "e.g." bug. If T7 or T8 shows up as fragments (e.g. "1am to 4am?"-style shards), the splitter is still breaking on abbreviations, decimals, or URLs — the fix is an abbreviation-aware segmenter, or letting the LLM set question boundaries instead of naive period-splitting.

---

## Expected question list (the 9)
1. Increase SFTP connection timeout? — *answered* (T1)
2. Throttle bandwidth per partner? — unanswered (T2)
3. SharePoint Online as destination? — unanswered (T3)
4. Validate integrity with SHA-256 on receipt? — *answered* (T4)
5. Why intermittent transfer failures to partner X? — unanswered (T5)
6. Per-folder retention policies? — unanswered (T6a)
7. Auto-purge files older than N days? — *answered* (T6b)
8. Max single-transfer file size? — unanswered (T7)
9. Why a 403 after upgrading to 10.15 CF19? — unanswered (T8)

---

## Scoring procedure
1. Total asks = 9 (not inflated by reply text)?
2. Answered = 3 (T1, T4, T6b)?
3. T5 counted as unanswered (reply is a clarifying question, not an answer)?
4. T5's reply question NOT logged as its own ask?
5. T6 split into 2, with a=unanswered and b=answered?
6. T7 is one whole question (no fragmenting on 10.15.2 / approx. / i.e. / e.g.)?
7. T8 is one whole question (no fragmenting on the URL or version)?
8. Recurring = 0, Product feedback = 0, both rendering cleanly?

If Answered comes back "—" or 0, stop and confirm whether the parser ingests replies at all — that determines whether this is a metric bug or an unbuilt feature.
