# Test Transcript 2 — Answer Key & Trap Map

Targets the two failure modes still open after Fixture 1: **within-message over-splitting** and the **product-feedback dumping-ground**. The defining feature of this fixture is the **calibration pair** — collapse traps (must become ONE) set against split controls (must stay TWO). A good fix passes both; an over-aggressive fix fails the controls; a weak fix fails the traps.

**14 messages → 16 distinct asks.**

---

## Headline numbers a correct run should produce

| Metric | Correct value | Notes |
|---|---|---|
| **Total asks** | **16** | 14 support + 2 product-feedback |
| **Support questions** (in buckets) | **14** | |
| **Recurring topics** | **1** | concurrency cap (M10 + M11) |
| **Answered** | **—** (no data) | no reply text in this fixture |
| **Product feedback** | **2** | bulk-edit (M7) + failure heat-map (M14) — and ONLY these |
| **Top theme** | **Operations & Performance** | |

### Expected theme counts (support only)
- **Operations & Performance — 8 topics:** retry policy, Slack/Teams alert, maintenance-window scheduling, Kafka transfer-complete event, bulk-deactivate (existing), 0-byte empty-file behaviour, concurrency cap *(the recurring one)*, reject-malformed-and-continue
- **File Operations — 2:** post-transfer archiving, encoding conversion
- **Platform & Connectivity — 3:** partner IP-range restriction, mutual TLS, IBM MQ transport
- **Licensing & Metering — 0** *(intentional — verifies the summary handles an empty theme without inventing a count)*

> Debatable-not-wrong routings: Slack/Teams alert and Kafka event could be a monitoring/integration bucket; reject-malformed could be File Operations. Don't fail the run over these.

---

## The one TRUE recurrence (must fire)
**Concurrency cap per node** — M10 (June 4) + M11 (May 31). Same intent, two messages, two dates → **2 occurrences**, one recurring group. This is also a cross-message recurrence guard: it must NOT be confused with a within-message duplicate.

---

## Calibration pair — the core of this fixture

### Collapse traps — each must produce EXACTLY ONE question
| # | Date | Message gist | Correct single question | Why it's one |
|---|---|---|---|---|
| M1 | Jun 9 | retry policy + "basically I want auto-retry" | How do I configure automatic retries for failed transfers? | restatement of the same ask |
| M2 | Jun 9 | archive after transfer + 3 bullet steps | Does MFT support timestamped post-transfer archiving? | bullets are steps, not separate asks |
| M3 | Jun 8 | "Slack/Teams alert on failure? Or wire up own webhook?" | Can MFT send a Slack/Teams alert on transfer failure, or is a custom webhook required? | the "or…?" is the same ask, not a 2nd question *(softest trap — if only this one splits, it's minor)* |
| M4 | Jun 8 | wrong-encoding context + "how to convert/normalize encoding" | How can we convert/normalize file encoding during transfer? | context + one restated ask |

### Split controls — each must STAY TWO questions
| # | Date | Message gist | Correct two questions | Why two |
|---|---|---|---|---|
| M5 | Jun 6 | "Two unrelated questions: 1. IP ranges 2. maintenance window" | (a) restrict partner connections to IP ranges? (b) schedule a transfer to a maintenance window? | genuinely unrelated; explicitly "two unrelated" |
| M6 | Jun 6 | "mutual TLS … and separately … Kafka event" | (a) mutual TLS for HTTPS partner endpoints? (b) publish transfer-complete event to Kafka? | two distinct asks joined by "and separately" |

**If M1–M4 produce 2+ each → over-split still broken. If M5 or M6 collapse to 1 → the fix over-corrected.** Both directions are failures.

---

## Product-feedback gate — build-it vs does-it-exist
| # | Date | Message | Correct destination | The distinction |
|---|---|---|---|---|
| M7 | Jun 5 | "bulk **edit** scheduled actions in the UI" (new capability) | **Product feedback** | asks to BUILD something |
| M8 | Jun 5 | "bulk-**deactivate** — is there an **existing** API or UI option?" | **Support** (Ops) | asks if a capability EXISTS |
| M14 | May 28 | "failure heat-map view — doesn't exist today, would help" | **Product feedback** | asks to BUILD something |
| M9 | Jun 4 | "0-byte silent success expected?" | **Support** (Ops) | troubleshooting/clarification — misroute bait |

M7 and M8 are the key pair: both are about "bulk operations on scheduled actions," but one is a feature request and one is a support question. **Product feedback must contain exactly {M7, M14}.** If M8 or M9 lands in feedback, the gate is still a dumping ground.

---

## Regression guards (confirm Fixture-1 fixes held)
- **M10 + M11** — true cross-message recurrence still fires (concurrency).
- **M12** (IBM MQ transport) — genuine unique singleton, must NOT be force-merged.
- **M13** (malformed file) — verb-drift bait: must say "**reject** the bad file and continue", NOT "disable validation" or "stop the listener".

---

## Scoring procedure
1. Total asks = 16?
2. M1, M2, M3, M4 → one question each? (collapse works)
3. M5, M6 → two questions each? (no over-correction)
4. Product feedback = exactly 2, and = {bulk-edit, heat-map}?
5. M8 (bulk-deactivate) and M9 (0-byte) in support buckets, not feedback?
6. Recurring = 1 (concurrency cap)?
7. M13 says "reject/continue", not "disable"?
8. Licensing & Metering = 0, and the summary doesn't invent a count for it?

All eight green = over-split and feedback-gate are fixed without collateral damage. A red on #2 = still over-splitting; a red on #3 = the fix went too far.
