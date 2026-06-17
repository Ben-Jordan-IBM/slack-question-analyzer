# Question Analyzer — Design System

A lean, IBM Carbon-true design system for the **Slack Question Analyzer**, an internal
tool for **IBM webMethods**. A companion Slack bot scans channels and extracts the
questions people ask; this application ingests that export and condenses it into a
**ranked, de-duplicated list of the most common questions**, grouping near-identical
questions by semantic similarity so teams know what to document first.

This system gives design agents everything needed to build on-brand screens, components,
and assets: Carbon color/type/spacing foundations, IBM Plex typography, reusable React
primitives, and a full interactive recreation of the analyzer app.

---

## Sources

- **Codebase:** `dashboard/` (read-only mount) — a Streamlit app
  (`dashboard/app.py`, `dashboard/README.md`) that already themes itself on the IBM Carbon
  Design System. It defines the data contract this system designs against:
  `results.groups[]` with `representative_question`, `count`, `avg_similarity`,
  `keywords[]`, and `questions[]` (each `{ text, date }`), plus
  `total_questions`, `total_groups`, and `metadata.similarity_threshold`.
- **Design language:** IBM Carbon Design System (open source — Apache 2.0 / SIL OFL for
  fonts). Recreated faithfully here; nothing proprietary is reproduced.

> The current Streamlit UI is functional but generic. This system is the redesign:
> simpler, sharper, and unmistakably IBM.

---

## Content fundamentals

How the product writes copy:

- **Voice:** plain, direct, second person ("your channel", "you know what to document").
  Confident but not salesy. No exclamation marks in product chrome.
- **Casing:** **Sentence case** everywhere — headings, buttons, menu items
  ("Analyze questions", "Most-asked questions", "New analysis"). Reserve UPPERCASE only
  for small metric labels and section eyebrows, tracked at `0.32px`.
- **Numbers & data:** always in **IBM Plex Mono** (`14×`, `91%`, `threshold=0.85`).
  This is a load-bearing convention — data reads as data, prose reads as prose.
- **Terminology:** "questions", "groups", "occurrences", "similarity threshold",
  "Slack export". A *group* is a cluster of similar questions; its *count* is how many
  times it was asked. Product-domain nouns come from webMethods (Integration Server,
  JDBC adapter, flow service, Universal Messaging).
- **Tone examples:** _"Find the questions your channel keeps asking."_ /
  _"Upload a Slack export and the analyzer condenses every question into a ranked,
  de-duplicated list — so you know what to document first."_
- **Emoji:** **none** in the product UI. (The legacy Streamlit app used them; the redesign
  drops them for a cleaner, enterprise tone.)

---

## Visual foundations

- **Colors:** IBM Blue `#0f62fe` (Blue 60) is the single brand/interactive color — used
  for primary actions, links, focus, and the top rank in the frequency heat-bar. Neutrals
  are the Carbon **Gray** ramp (10–100) and do almost all the work: gray-10 panels,
  gray-20 borders, gray-100 text and the dark UI-shell header. Status uses red-60 / green-60
  / yellow-30. A small accent set (purple, teal, cyan, magenta) is reserved for data-viz and
  tags. See `tokens/colors.css`.
- **Type:** **IBM Plex Sans** for everything UI; **IBM Plex Mono** for all numerals and
  technical strings; IBM Plex Serif available but rarely used. Large headings are **Light
  (300)** with tight `-0.02em` tracking — the signature Carbon "expressive" look. UI
  headings are Semibold (600) at 14–16px. See `tokens/typography.css`.
- **Spacing:** strict **8px grid** (Carbon mini-units, `--spacing-01`…`13`). Layouts breathe
  via 16/24/32/48px rhythm. See `tokens/spacing.css`.
- **Corners:** **sharp.** `border-radius: 0` is the default and the brand signal. Only tags
  are pill-shaped; an opt-in 2–4px radius exists but is rarely used. Never round cards or buttons.
- **Borders:** 1px `gray-20` hairlines separate everything (cards, rows, metric tiles sit on
  a 1px grid). Left **3px accent bars** signal status/selection (Carbon hallmark) rather than
  full colored borders.
- **Backgrounds:** flat, solid fills only. White content surfaces on a gray-10 page; a dark
  gray-100 UI-shell header. **No gradients, no imagery, no textures, no illustrations** —
  Carbon is resolutely flat and functional.
- **Elevation:** restrained, cool-gray shadows (`--shadow-sm/md/lg`). Most surfaces are
  borders-only with **zero** shadow; shadow appears only on hover of interactive cards and on
  overlays. See `tokens/elevation.css`.
- **Cards:** white fill, 1px gray-20 border, no radius, no shadow at rest. Optional left
  accent bar; optional hover lift (`--shadow-md`) when interactive.
- **Focus:** Carbon's signature 2px focus treatment — `--focus-ring` (2px blue outline) or the
  inset double-ring for fields/buttons. Always visible, never removed.
- **Hover / press:** hovers darken one step (blue-60 → blue-70; gray rows → `--layer-hover`).
  No scale/bounce. Presses go one step darker again (blue-80). Transitions are short
  (70–150ms) on Carbon's productive easing `cubic-bezier(0.2,0,0.38,0.9)`.
- **Motion:** purposeful and quick. Fades and width/height transitions only — no springs, no
  decorative looping. Entrance `cubic-bezier(0,0,0.38,0.9)`, exit `cubic-bezier(0.2,0,1,0.9)`.
- **Transparency / blur:** essentially unused. Surfaces are opaque; the system favors solid
  color and hairlines over glass effects.
- **Imagery vibe:** N/A — this product uses **no photography or illustration**. The only
  "imagery" is data: the frequency heat-bar, where rank 1–3 ride the blue ramp
  (60 → 50 → 40) and lower ranks fade to gray-40.

---

## Iconography

- **System:** the product targets **@carbon/icons** (IBM's 2px-stroke, 16px-optical-grid,
  square-terminal icon set). Carbon icons ship as individual SVGs, not a CDN font.
- **Substitution (please confirm):** this system loads **[Lucide](https://lucide.dev)**
  from CDN (`unpkg.com/lucide`) as the closest readily-available match — same 2px stroke and
  geometric construction. Lucide uses *rounded* caps where Carbon uses square ones, so for
  production you should swap in the real `@carbon/icons` SVGs. Used in cards/UI kit via the
  `data-lucide="<name>"` attribute and `lucide.createIcons()`.
- **Common icons:** `search`, `upload`, `filter`, `arrow-up-down`, `sliders-horizontal`,
  `hash`, `download`, `chevron-down`, `trending-up`, `sparkles`, `check`, `x`.
- **Sizing:** 16px in dense UI, 18–22px for emphasis. Stroke width fixed at 2.
- **Emoji / unicode as icons:** not used.

---

## Logo & brand

There is no provided product logo, so this system ships an **original** wordmark:
a black square **ranked-bars glyph** (three descending blue bars — the core metaphor of the
app) paired with an IBM Plex "**Question** Analyzer" lockup (Semibold + Light). Files:
`assets/logo.svg` (lockup), `assets/mark.svg` (glyph only). The IBM 8-bar logo is **not**
reproduced.

---

## Index / manifest

**Root**
- `styles.css` — global entry point (consumers link this one file); `@import`s all tokens + fonts.
- `tokens/` — `colors.css`, `typography.css`, `spacing.css`, `elevation.css`, `fonts.css`.
- `assets/` — `logo.svg`, `mark.svg`.

**Components** (`components/<Name>/` — `.jsx` + `.d.ts`; compiled into
`_ds_bundle.js`, which the UI kit loads)
- `Button` — primary/secondary/tertiary/ghost/danger; sharp, right-pinned icon.
- `Tag` — Carbon pill for keywords & status; 8 color pairs, outline, dismissible.
- `Card` — layered surface; accent bar, hover lift, selected state.
- `MetricTile` — big-number summary stat (light numeral, uppercase label).
- `Slider` — range control with live readout; the similarity threshold.
- `FileDropzone` — drag-drop upload field with selected-file chip.
- `QuestionGroup` — **hero** ranked, expandable group of similar questions.

**UI kit** (`ui_kits/analyzer/`) — the real dashboard app: `index.html` runs the
end-to-end flow (upload, analyze, ranked results) against the backend (`api.js`),
assembled in `App.jsx` from `AppHeader`, `DashboardView`, `WeekView`, `Modals`,
`RankedRow`, `Icon`, and `anim`.

---

## Notes & caveats

- **Fonts** load from Google Fonts CDN (IBM Plex is SIL OFL). To self-host, replace the
  `@import` in `tokens/fonts.css` with local `@font-face` rules + `.ttf`/`.woff2` binaries.
- **Icons** are a Lucide stand-in for `@carbon/icons` — see Iconography above.
