# Answer key — field run 2026-07-07 (`field_run_2026-07-07.txt`)

The real July 7, 2026 channel dump: 52 raw messages. This is the messiest
fixture in the suite on purpose — it is the transcript that motivated
**announcement immunity** (v2.47.0). Conventions follow the one-answer
yardstick adopted with prompt pack 22: two extractions are one ask when a
single answer would fully resolve both.

## Noise inventory (must produce ZERO questions)

| Message | Why it is context, not a question |
|---|---|
| July 7 "Please use this documentation page…" | Resource NOTICE. Contains the statement "MFT is not yet available on Azure for Gen2" — the observed phantom inverted this into "Is MFT available on Azure for Gen2?". Banned via `gen ?2` / `capability and feature parity`. |
| July 1 webinar promo (":alert-blue: Webinar Alert…") | Broadcast. Observed phantoms: "Why should customers care?" (rhetorical header) and "When: July 8, 2026 \| 9:00 AM EDT" (schedule line). Both banned. |
| June 30 Sales Kit post (×2, double-posted) | Broadcast. Observed phantom: the FASP "question" fabricated from marketing copy; also the source of the "What's New *" contamination prefix that leaked onto a real question. `fasp`, `what's new`, `sales kit` banned. |
| June 2 Bupa win wire | Broadcast. Its migration story must not leak into rewrites (`bupa` banned). |
| July 6 "Message text is not available." | Stub. |
| June 2 (empty message) | Empty. |
| June 12 "deployment documentation will be available…" | Statement/reply; no ask. Not an announcement — just contains no question. |

## Recurrences (exactly these two)

1. **Sizing guidance, 2×** — June 30 double-ask ("do we have any sizing
   guidance or calculation documents for webMethods MFT… CP4I add-on
   sizing based on VPCs"). Two near-identical messages, one group.
2. **Transaction estimation, 2×** — June 2 ("how can customer estimate
   their potential number of mft transactions… entitlements") + May 30
   ("check their own transaction statistics similar to what the metering
   report provides"). Different wording, same ask: one answer (how to
   measure transfer counts without/beyond the metering server) resolves
   both.

Everything else is a genuine singleton. In the observed run, "Supported
Protocols" showed LinuxONE at 2× — the second member can only be junk (the
Azure-Gen2 phantom or the Windows-Server cert question). Hence
`must_stay_singleton: linuxone` and the `must_not_group` pairs.

**Borderline, deliberately unpinned**: June 5 "email notification when a
virus is detected" vs June 2 "quarantine + mail on failed scan" — one
config answer plausibly covers both, so a 2× virus-notification group is
acceptable; so is keeping them separate. `recurring_topics` is therefore
NOT pinned in this fixture.

## Consolidation calls (`message_asks`)

- **HashiCorp Vault (June 29) = 1 ask.** "Can HashiCorp Agents be
  leveraged here?" and "Any other ideas?" are facets of the interim-vault
  question — one answer enumerates the options. ("Any other ideas?" is
  also content-free rhetorical filler on its own.)
- **Box folder / RFP responses (June 2) = 1 ask.** "All together in one
  area?" is a continuation fragment, not a second question.
- **IWHI APIs (June 30) = 1 ask.** "Any IBM KC link would help?" asks for
  the same answer (the API docs link).
- **Metering Agent (June 9) = 1 ask.** "so customer just need to configure
  to point to location of on-prem metering server?" is the same ask
  restated — the answer to one IS the answer to the other.
- **"Coming back to this use case" (June 2) = 2 asks.** Prevalence ("is it
  common…") and references ("do we have customer references?") have
  different answers. The observed run produced 3 (an invented
  migration-variant plus both raw forms) — 2 is correct.
- **Danone bug report (June 10) = 3 asks.** Explicitly bulleted: same
  defect? / does Fix14 fix it? / workaround? The asker enumerated them.
  This message is also the anti-false-positive guard for the announcement
  filter: it is emoji- and bullet-heavy but help-seeking.
- **Windows Server 2025 (June 16) = 2 asks.** Numbered: WS2025
  certification and Azure SQL 2025 compatibility.
- **Product feedback (June 5) = 2 items.** The message says "customer
  product feedback" and enumerates: (1) graphical Action-log view,
  (2) group/filter Actions. BOTH belong in the feedback lane — the
  observed run lost item (1).
- **MFA webPortal (June 19, double-posted)** — one ask by the one-answer
  test (MFA support + doc/screenshot request share one answer). Left out
  of `message_asks` because the byte-identical double-post makes row
  counting ambiguous; presence is guarded via `support_must_match`.
- **Timezone/UTC message (June 16)** — 1 or 3 asks are both defensible
  (three enumerated facets; one underlying design question). Deliberately
  unpinned; survival guarded via `support_must_match`.

## Routing spot checks

- LinuxONE support → an install/platform bucket, never review.
- Negative transfer speed (>2GB) → errors/monitoring territory.
- Transaction estimation → Metering & Licensing.
- `prod537147` status check → **review pile** (a status request, not a
  knowledge question — nothing should claim it).

## Answered

The dashed-text format carries no thread replies: `answered_count: 0` and
the dashboard should show the "unmeasurable" state, not "0 answered".
