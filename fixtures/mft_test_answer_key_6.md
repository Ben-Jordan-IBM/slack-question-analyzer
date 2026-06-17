# Test Transcript 6 — Answer Key & Trap Map (Routing Edges / Humility)

Purpose: test **routing humility** — whether the pipeline sends genuinely unplaceable questions to the **uncertain/review** pile instead of forcing them into a bucket, while still routing clear questions confidently. This exercises the `0`/abstain path from the ROUTE prompt that no other fixture stresses. It also tests that a question isn't dumped into "uncertain" merely because it contains an error string.

**8 messages → 8 asks. The split that matters: 4 route confidently, 4 should NOT be force-bucketed.**

---

## Headline numbers a correct run should produce

| Metric | Correct value | Notes |
|---|---|---|
| **Total asks** | **8** | all are questions |
| **Cleanly routed to MFT buckets** | **4** | M3, M6, M7, M8 |
| **Uncertain / review / off-topic** | **4** | M1, M2, M4, M5 |
| **Recurring topics** | **0** | |
| **Product feedback** | **0** | |

There is no single "right" label for the uncertain pile (it may be called review, `0`, off-topic, or held). **The assertion is that M1, M2, M4, M5 do NOT land in a normal MFT theme** (Operations, File Operations, Platform & Connectivity, Licensing & Metering).

---

## The 4 that must route CONFIDENTLY (clear home)

| # | Date | Question | Bucket | Note |
|---|---|---|---|---|
| M3 | Jun 7 | Configure SFTP connection to a new partner | Platform & Connectivity | unambiguous |
| M6 | Jun 4 | PGP encryption for outbound files | File Operations / Connectivity | unambiguous |
| M8 | Jun 2 | Why does SFTP drop with "algorithm negotiation failed"? | Platform & Connectivity | **has an error string, but a clear subject (SFTP negotiation)** |
| M7 | Jun 3 | Does a moved file get archived or deleted by default? | File Operations *(or Ops)* | mild File-vs-Ops lean; pick one, don't abstain |

**M8 is the key control:** it contains an error message, but its subject is unmistakably SFTP connectivity. It must route to Connectivity, **not** get dumped into uncertain just because an error is present. "Has an error word" ≠ "ambiguous."

---

## The 4 that must NOT be force-bucketed

| # | Date | Question | Why it shouldn't get an MFT bucket | Correct destination |
|---|---|---|---|---|
| M1 | Jun 9 | Is the Confluence wiki down? | **off-topic** — not MFT, not even the product | excluded or uncertain |
| M5 | Jun 5 | IBM holiday support hours? | **off-topic** — not a product/support question at all | excluded or uncertain |
| M2 | Jun 8 | Does a virus-scan failure count against metered transactions? | **genuinely spans two buckets** — Antivirus AND Metering, roughly equally | uncertain / `0` (or a documented convention call) |
| M4 | Jun 6 | "Is there a limit?" | **too vague** — limit on what? unroutable as written | uncertain / review |

---

## The traps, by type

1. **Off-topic rejection (M1, M5).** A healthy pipeline either excludes these or parks them in uncertain. The failure is forcing "Confluence wiki down" or "IBM holiday hours" into, say, Operations & Performance just because they're questions in the channel. Off-topic items are also your radar — a pile of them means the channel scope is wider than the taxonomy.

2. **Genuinely split (M2).** Virus-scan-vs-metering is the rare question that truly sits between two buckets (it's *about* the interaction of antivirus and metering). This is what the top-2-anchors-within-margin → `0` path exists for. Either uncertain, or a documented convention ("billing questions win"), but **not** a silent coin-flip into one bucket.

3. **Too vague (M4).** "Is there a limit?" has no routable subject. It should go to review, not be force-fit. Tests that the router abstains on insufficient signal rather than guessing.

4. **False-uncertain control (M8).** The inverse trap: a question with a clear home that *looks* hard because it quotes an error. Must route confidently. If M8 lands in uncertain, the router is over-abstaining — treating any error mention as ambiguous.

---

## Scoring procedure
1. Total asks = 8?
2. M1 and M5 NOT in an MFT theme bucket (excluded or uncertain)?
3. M2 in uncertain/`0` (or a documented convention), NOT silently dropped into one of Antivirus/Metering?
4. M4 in review/uncertain, NOT force-bucketed?
5. M3, M6, M7 routed confidently to their buckets?
6. **M8 routed to Connectivity, NOT dumped into uncertain** (the over-abstain check)?
7. Recurring = 0, Product feedback = 0?

All green = the router knows when it doesn't know — it abstains on off-topic, split, and vague questions, but stays confident on clear ones even when an error string is present. The two likely reds are #2/#3/#4 (force-bucketing what should abstain) or #6 (over-abstaining on M8).
