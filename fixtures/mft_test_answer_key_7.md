# Test Transcript 7 — Answer Key & Trap Map (Volume + Ranking + Emerging Topic)

Purpose: the only fixture at **realistic volume**. It tests three things the small fixtures can't: **exact counts under load**, **correct ranking of several recurring groups of different sizes**, and **emerging-category detection** (a coherent cluster of in-domain questions that fit no existing bucket and should surface as uncertain / a new-category signal — the outlier-pile-as-radar behavior). Noise-rejection and multi-part splitting are also re-checked under load.

**24 messages → 24 distinct asks.**

---

## Headline numbers a correct run should produce

| Metric | Correct value | Notes |
|---|---|---|
| **Total asks** | **24** | 22 support + 2 product-feedback |
| **Recurring topics** | **4** | sizes 4 / 3 / 2 / 2 |
| **Answered** | **—** (no data) | no replies |
| **Product feedback** | **2** | drag-drop builder + cron expressions |
| **Emerging / uncertain** | **3** | the AI-agent questions — should NOT be force-bucketed |

---

## The 4 recurrences — RANKING is the test

The "Ranked · most frequent first" list must order these by occurrence count, with exact counts and populated rows:

| Rank | Topic | Count | Source dates |
|---|---|---|---|
| **1** | SFTP connection timeout | **4×** | Jun 11, Jun 8, Jun 5, Jun 1 |
| **2** | Transfer-failure alerting | **3×** | Jun 10, Jun 6, Jun 2 |
| **3 (tie)** | New partner onboarding | **2×** | Jun 9, Jun 3 |
| **3 (tie)** | Purge old transfer records | **2×** | Jun 7, Jun 1 |

Assertions: timeout (4×) ranks above alerting (3×) ranks above the two 2× groups; the two 2× groups are a **tie** (don't fabricate an order between them, or note it as a tie); every group's count equals its populated-row count; all four merged despite **non-adjacent dates** spread across the channel.

---

## The EMERGING topic — must NOT be force-bucketed

Three questions about the **AI / agentic** capability, which maps to none of the existing buckets (Antivirus, Metering, Scheduling/Performance, File handling, Connectivity, Monitoring, Errors, Install/Admin):

- M4 (Jun 10): AI Integration Agent building a flow from a plain-English prompt
- M12 (Jun 6): how the agentic AI chooses transformation steps
- M18 (Jun 3): AI Observability Agent auto-remediating a failed transfer

**Correct behavior:** these surface as uncertain / review, ideally recognizable as a coherent emerging cluster — the signal that an **"AI / Agentic" bucket** should be added. They must NOT be scattered into operational buckets to look "handled."

> Watch the forced-fit temptation: a weak router may grab M18 for *Monitoring* ("observability") or *Errors* ("remediate"), and M12 for *File handling* ("transformation"). Those are forced fits. The subject is the AI agent, which is new — so abstain, don't force. If all three land in normal buckets, the emerging-category radar is off.

---

## Volume re-checks (must still hold under load)

| # | Date | Type | Correct result |
|---|---|---|---|
| M8 | Jun 8 | multi-part | split into **2** (max retry count; custom transfer label) |
| M16 | Jun 4 | noise | office-hours reminder → **0 questions** |
| M10 / M24 | Jun 7 / May 30 | feature requests | both → **product feedback**, routed out (drag-drop builder, cron expressions) |

The 6 singletons (FTPS explicit TLS, Azure Data Lake Gen2, chmod permissions, exclude file extensions, audit-log retention, webhook-triggered transfer) each stay singletons — none should be swept into a recurrence or the emerging cluster.

---

## Why volume matters here
At 24 messages the failure modes differ from the small fixtures: counts drift (an off-by-one is easy to miss), the summary may mis-state which theme leads or invent a distribution, ranking ties get silently broken, and a 4× recurrence with one row sourced from a far-apart date can lose that occurrence. This fixture is the closest thing to "point it at a real month of Slack" — if the counts and ranking hold here, the arithmetic is trustworthy at production scale.

---

## Scoring procedure
1. Total asks = 24?
2. Recurring = exactly 4, with counts 4 / 3 / 2 / 2?
3. Ranked order = timeout (4×) → alerting (3×) → {onboarding, purge} as a 2× tie?
4. Every recurring group's count equals its populated-row count (no empty slots), including the 4× sourced from 4 non-adjacent dates?
5. The 3 AI-agent questions surface as uncertain/emerging, NOT force-bucketed into Monitoring/Errors/File handling?
6. M8 split into 2; M16 produced 0; the 6 singletons stayed singletons?
7. Product feedback = 2 (drag-drop, cron)?
8. Summary states the leading theme and counts accurately (no invented distribution)?

All green = counts, ranking, and emerging-category detection hold at volume. Likely reds: #3 (tie broken or mis-ordered), #5 (AI cluster force-fit), or #4 (a far-dated occurrence dropped from the 4× group).
