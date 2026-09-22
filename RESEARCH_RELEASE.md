# Research homepage — September 22, 2026

The root page combines the companion paper, six recorded course-design comparisons,
the electrical/cooling/hand findings, and access to all existing interactive experiments.
The longer article remains at `report.html`. The existing `#demo-update` link opens
the embedded replay. `lab/?design=kv_high#terrain` selects the matching course run;
unsupported design identifiers fall back to the native reference.

## Evidence and current scope

- `assets/research/data/candidates.json` copies the challenge manifest and joins its
  evaluated design parameters. The six tiles show actual videos, with one shared clock.
  These are fixed-policy, single-seed development comparisons, not a completed search.
- `assets/research/data/cooling-source.json` preserves the completed Allegro sweep.
  The summary and figure report 32 replicas per design over 120 seconds. Temperature
  is a recorded simulation state; startup replication is not policy-training replication.
- Atlas has reported a newer physical thermal bench test. Its source record is still
  being identified. The site reserves a separate place for those results and does not
  label the simulation sweep as a physical bench measurement.
- `paper/` contains the manuscript source, online version, PDF, figures, figure data
  and proposed experiment protocol. The independent design-search comparison and
  motor-placement experiment remain explicitly pending.

## Updating the static release

```bash
python tools/build_cooling_results.py
python tools/build_research_home.py
```

The cooling builder uses NumPy and matplotlib. It checks replica counts, recomputes
throughput from raw goal counts, and writes the source hash to the summary. The homepage
builder uses the standard library. Poster images are frame grabs from the existing videos.
The paper's Markdown and HTML are included for editing alongside the review PDF.

Replay component changes are built with `npm run typecheck` and `npm run build` in
`lab-src`. New homepage media reuse existing full videos; no duplicated video library
is added to the Pages artifact.

## Validation for this release

Type checking and the production build pass. Browser checks cover synchronized video
play/pause/seeking, early-ended videos, candidate selection and parameters, matching
candidate video/3D deep links, the legacy embed anchor, experiment switching, mobile
overflow, local links and assets, the PDF and byte-range video responses. Desktop and
mobile screenshots were inspected. The nine-page PDF includes the completed cooling
sweep and identifies physical-bench data as a separate update.
