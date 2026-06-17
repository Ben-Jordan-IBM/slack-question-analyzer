# Test Transcript 3 — Answer Key & Trap Map

Purpose: hammer the **recurrence and occurrence-count logic** — the area that has broken in five straight runs (phantom recurrences, empty occurrence slots, within-message restatement counted as a recurrence). This fixture exists to be the regression test for the *occurrence-integrity invariant*: **a group's occurrence count must equal the number of populated, distinct-source rows it can show.**

**13 messages → 13 distinct asks.**

---

## Headline numbers a correct run should produce

| Metric | Correct value | Notes |
|---|---|---|
| **Total asks** | **13** | 12 support + 1 product-feedback |
| **Recurring topics** | **2** | Host Key Verification (3×) and Partner Onboarding (2×) — and ONLY these |
| **Answered** | **—** (no data) | no replies in this fixture |
| **Product feedback** | **1** | retry-with-backoff UI request |

> Theme placement is loose here and not the point — most of these land in Platform & Connectivity. **The assertions are the occurrence counts below, not the buckets.**

---

## The TWO true recurrences (must fire, with populated rows)

**1. Host Key Verification — 3 occurrences.** M1 (Jun 9) + M5 (Jun 5) + M10 (May 30). Three different messages, three dates, same intent. This must show **3×** with **three populated rows**. Tests that counts exceed 2 and every row renders (no empty slots), and that non-adjacent dates still merge.

**2. Partner Onboarding — 2 occurrences.** M2 (Jun 8) + M8 (Jun 2). Same intent ("onboard / set up a new partner"), worded differently. **2×**, two populated rows.

Everything else is asked once.

---

## Phantom-recurrence traps (must NOT become recurrences)

These are single messages that restate one ask — the exact bug that produced empty-slot "2×" groups. Each must be **one question, one occurrence**.

| # | Date | The restatement | Correct result |
|---|---|---|---|
| M3 | Jun 7 | "schedule a recurring transfer… **Basically** run every night at midnight" | ONE question, 1× — NOT a "Recurring Transfer" 2× |
| M6 | Jun 4 | "compress files before sending… **I mean** gzip or zip the payload" | ONE question, 1× |

If either shows as a 2× group (especially with an empty second row), the same-ask consolidation is still re-emitting collapsed text as a phantom occurrence.

---

## Distinct-source trap (identical text, ONE source → ONE occurrence)

| # | Date | What's in it | Correct result |
|---|---|---|---|
| M7 | Jun 3 | The resume-interrupted question appears **twice in the same message** (forwarded quote + restated) | ONE question, **1× — not 2×** |

This is the cleanest test of the distinct-source rule: the same text appears twice, but from **one source message**, so it is one occurrence. If it shows 2×, the recurrence counter is counting text repetition instead of distinct sources.

---

## False-recurrence resistance (look similar, genuinely different → stay separate)

| # | Date | Question | Must NOT merge with | Why |
|---|---|---|---|---|
| M4 | Jun 6 | rotate **PGP encryption** keys | M9 (SSH keys) | both "rotate keys", different subjects (encryption vs auth) |
| M9 | Jun 1 | rotate **SSH** keys for SFTP | M4 (PGP keys) | different root cause, different doc page |

These share heavy vocabulary ("rotate", "keys") and are each other's nearest neighbor — exactly the kind of pair the singleton-rescue pass might wrongly merge. VERIFY must return false (different root cause). They stay **two separate singletons**.

---

## Genuine singletons (must stay singletons, not get rescued into a group)
- **M11** (May 29) — Google Cloud Storage destination
- **M12** (May 28) — max single-transfer file size

The rescue pass must leave these alone. If either gets absorbed into a group, the rescue bias is still too aggressive.

---

## Occurrence-integrity invariant to assert
For every group on the page: **occurrence count == number of non-empty rows, and every row has a distinct source message.** On this fixture:
- Host Key Verification: count 3, three rows, three sources ✓
- Partner Onboarding: count 2, two rows, two sources ✓
- Any other "2×" group is a **bug** (phantom from M3/M6, or text-repeat from M7).

---

## Scoring procedure
1. Total asks = 13?
2. Recurring = exactly 2 (Host Key Verification 3×, Partner Onboarding 2×)?
3. Host Key Verification shows 3 populated rows from 3 distinct dates?
4. M3 and M6 are each one question, NOT a 2× group?
5. M7 is 1× (one occurrence) despite the text appearing twice in the message?
6. M4 (PGP rotate) and M9 (SSH rotate) are SEPARATE, not merged?
7. M11 and M12 are singletons, not rescued into groups?
8. Product feedback = 1 (retry-with-backoff)?
9. Every group's occurrence count equals its populated-row count (no empty slots)?

All nine green = recurrence and occurrence integrity are solid. The most likely red is #4 or #5 (phantom / text-repeat occurrences), and the empty-slot signature makes them easy to catch.
