# Test Transcript 5 — Answer Key & Trap Map (Noise / Precision)

Purpose: test **extraction precision** — the pipeline's ability to correctly extract *nothing* from statements, announcements, chatter, and log pastes. The other fixtures test whether real questions are found and grouped; this one tests whether **fake questions are invented from non-questions.** A channel like this is what a real Slack export mostly looks like.

**14 messages → only 4 real asks.** Ten messages are noise and must produce zero questions.

---

## Headline numbers a correct run should produce

| Metric | Correct value | Notes |
|---|---|---|
| **Total asks** | **4** | anything higher = fabricated questions from noise |
| **Recurring topics** | **0** | |
| **Answered** | **—** (no data) | no replies |
| **Product feedback** | **0** | |

**The single most important assertion: total asks = 4, not 5+.** If the count exceeds 4, the extractor is manufacturing questions from statements — the precision failure this fixture is built to catch.

---

## The 4 REAL questions (must be extracted)

| # | Date | Question | Notes |
|---|---|---|---|
| M2 | Jun 9 | Can retry count be set per partner, or is it global only? | buried in "morning all / before standup" filler |
| M5 | Jun 6 | Why do transfers to the EU partner silently stop after ~50 files with no error? | **implicit** request — no question mark, but a clearly stuck problem |
| M8 | Jun 5 | Does the cleanup scheduler support a dry-run mode? | clean question |
| M11 | Jun 3 | Can we get a daily email digest of failed transfers instead of per-failure alerts? | wrapped in pleasantries |

These mostly route to Operations & Performance. Routing is not the point here — the count is.

---

## The 10 NOISE messages (must produce ZERO questions)

| # | Date | Type | Why it's not a question |
|---|---|---|---|
| M1 | Jun 9 | announcement | maintenance-window notice |
| M3 | Jun 8 | resolved thread | "fixed now, thanks" — a closure, not an ask |
| M4 | Jun 8 | status update | "pushed config, will validate" |
| M6 | Jun 6 | **venting** | "Mondays… nightmare" — no resolvable problem |
| M7 | Jun 5 | social / +1 | "same here" pile-on |
| M9 | Jun 4 | FYI link | release-notes link |
| M10 | Jun 4 | **log paste** | a WARN+INFO log that self-resolved, no ask |
| M12 | Jun 3 | acknowledgment | "thanks, that worked" |
| M13 | Jun 2 | decision/announcement | "standardizing on AS2" |
| M14 | Jun 2 | social | "ready for the weekend?" — rhetorical |

---

## The precision traps (where fabrication happens)

These three are the messages most likely to be wrongly turned into questions:

1. **M5 (extract) vs M6 (don't).** Both are frustrated venting. But M5 contains a concrete, resolvable problem ("transfers silently stop after ~50 files") → it's an implicit question. M6 is pure mood ("Mondays… nightmare") with no problem to solve → not a question. **Extracting M6 is the classic precision failure.** The test: a resolvable symptom makes it a question; an emotion does not.

2. **M10 (log paste, don't extract).** A WARN/INFO log with no ask, explicitly "self-resolved, posting for visibility." The extractor must not turn a log dump into "Why is there a retry warning?" — there's no question being asked. Tests that log text alone ≠ a question.

3. **M14 / M7 (social, don't extract).** "Anyone else ready for the weekend?" is question-shaped but rhetorical/social. "Same here" is a +1. Both must be dropped as non-asks despite M14 ending in a question mark.

---

## Scoring procedure
1. **Total asks = exactly 4?** (the headline test — 5+ means fabrication)
2. M5 extracted (implicit, real stuck problem)?
3. M6 NOT extracted (venting, no resolvable problem)?
4. M10 NOT extracted (log paste, no ask)?
5. M14 NOT extracted (rhetorical/social despite the "?")?
6. M2 and M11 extracted despite their chatter/pleasantry wrappers?
7. Recurring = 0, Product feedback = 0?

All green = the extractor has good precision and won't flood the dashboard with phantom questions when pointed at a real, noisy channel. The likely red is #3 or #4 — over-extraction from venting or log pastes.
