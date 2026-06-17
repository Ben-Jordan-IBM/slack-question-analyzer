# Test Transcript — Answer Key & Trap Map

Use this to score a pipeline run on `mft_test_transcript.txt`. The transcript has **15 messages** containing **17 distinct answerable asks**. Each message is a planted test case.

---

## Headline numbers a correct run should produce

| Metric | Correct value | Notes |
|---|---|---|
| **Total asks extracted** | **17** | 15 support questions + 2 product-feedback |
| **Support questions** (in buckets) | **15** | after routing the 2 feature requests out |
| **Recurring topics** | **1** | exactly one true recurrence (see below) |
| **Answered** | **—** (no data) | transcript has no reply text; must NOT show 0 |
| **Product feedback** | **2** | dark mode + audit-log CSV export, both from the June 2 message |
| **Top theme** | **Operations & Performance (6)** | |

### Expected theme counts (support questions only)
- **Operations & Performance — 6:** bandwidth throttle, DST scheduler failure, UI hang on login, connection-pool-exhausted error, purge DB records, max concurrent SFTP sessions
- **File Operations — 4:** resume interrupted transfer, route by filename prefix, handle checksum failure, zip-compress before transfer
- **Platform & Connectivity — 3:** SFTP key rotation *(the recurring one)*, PGP encryption, FIPS 140-2 ciphers
- **Licensing & Metering — 1:** per-partner volume API for chargeback

> Two routings are **debatable, not wrong**: PGP encryption could sit in File Operations instead of Connectivity, and "purge DB records" could sit in an admin bucket. These are convention calls — don't fail the run over them. The **counts and the trap behaviors below are the real assertions.**

---

## The one TRUE recurrence (must fire)

**SFTP SSH key rotation without downtime** — Msg 1 (June 9) + Msg 6 (June 4). Two distinct messages, two distinct dates, same intent. This is the single recurring group; it should show **2 occurrences**. If the pipeline reports 0 recurring, key-rotation detection is broken.

---

## Traps (what each message is testing)

| # | Date | Tests | Correct handling | Failure signature |
|---|---|---|---|---|
| 1 | Jun 9 | True recurrence (A) | Pairs with Msg 6 into one recurring group | — |
| 2 | Jun 9 | Multi-part, 2 buckets | Split into **2** questions; throttle → Ops, resume → File Ops; **both survive** | one of the two silently dropped |
| 3 | Jun 8 | Over-split / scaffolding | **ONE** question (route by prefix). Bullets are steps, NOT separate "how do I use a Find task" questions | extractor emits 2–3 fragments |
| 4 | Jun 8 | Rhetorical + filler | **ONE** question (DST scheduler). Drop "Anyone seen this?" and "Thanks!" | rhetorical asks kept as questions |
| 5 | Jun 5 | **Verb drift** | "**handle** checksum failure gracefully / route to review folder." Must NOT become "disable/bypass checksum validation" | intent verb changed to bypass/disable |
| 6 | Jun 4 | True recurrence (B) | Pairs with Msg 1 | — |
| 7 | Jun 4 | False-merge twin | "PGP encrypt before send" stays separate from Msg 12 | merged with Msg 12 on boilerplate |
| 8 | Jun 3 | Implicit request (no "?") | Extract "Why does UI hang on login for one user?" | dropped because no question mark |
| 9 | Jun 3 | Raw error paste | Extract as defect/troubleshooting; **don't drop** | stack-trace paste dropped |
| 10 | Jun 2 | Feature requests, 2 | **2** product-feedback items, routed **out** of support funnel; **both survive** | one dropped, or routed into a support bucket |
| 11 | Jun 2 | Unique singleton | Stays a singleton — no near neighbor | force-merged into something unrelated |
| 12 | May 30 | False-merge twin | "zip compress before transfer" stays separate from Msg 7 | merged with Msg 7 |
| 13 | May 30 | Same-message double phrasing | **ONE** question (purge >90-day records). The restatement is the same ask | counted as 2 occurrences / fake recurrence |
| 14 | May 29 | Tail-batch survival | Survives as distinct question | dropped, or date back-filled from a neighbor |
| 15 | May 28 | Tail-batch survival | Survives in Licensing & Metering | dropped, or text duplicated onto wrong date |

---

## The two regression checks this fixture exists to enforce

1. **Silent drops.** Msgs 2, 10, 14, 15 are the bait. After extraction, every source message must yield at least one ask OR an explicit logged "no question." If total asks < 17, something dropped — trace which message produced zero.

2. **Date integrity / fake recurrence.** Msgs 7+12 (boilerplate twins) and Msg 13 (one message, restated) are the bait. If any "recurring topic" other than SFTP key rotation appears, or any group shows 100%-identical text from a single source message, the date-backfill / dedup bug is present.

---

## Quick scoring procedure

1. Total asks = 17? (15 support + 2 feedback)
2. Recurring topics = exactly 1, and it's SFTP key rotation?
3. Product feedback = exactly 2 (dark mode, audit CSV)?
4. Licensing & Metering = 1 (the chargeback API)?
5. No verb drift on Msg 5 (says "handle", not "bypass/disable")?
6. Msg 3 is one question, not three?
7. Msgs 7 and 12 are separate, not merged?

All seven green = the bugs are fixed. Any red points straight at the stage that needs work.
