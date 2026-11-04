# Track 3 technical report

`main.tex` is the two-page technical report (plus references) the organisers' form asks for, on the
anonymous `nvidiaextabstract` template. The committed PDF the form links to is
`docs/report/final/track3-technical-report.pdf`; `scripts/reproduce/release.json` pins the same build
(`report.sha256`, `report.epoch` = the commit time of the last change under `docs/report/track3`).

Build (CPU, deterministic for a given commit):

```bash
bash scripts/reproduce/d3_report.sh --run    # prints where main.pdf lands
bash scripts/reproduce/check_release.sh --combined <release manifest>   # rebuilds and compares the hash
```

`main.tex` is written in plain prose with every number inline (no toggle or macro block since 4 November
15:00 UTC). After any wording change: commit, run `d3_report.sh --run`, move `release.json`'s `epoch` and
`sha256` to the new build, copy the PDF to `docs/report/final/` and commit. The uploaded moving pack's
admission record pins the hash of the `main.tex` it was admitted against (at its URL commit `c8ff8c1`), so a
later wording change is a documentation update, not a change to the packed file. The report names no AI
tools or agents (owner, 4 November).

`figures/` holds the figure sources of the earlier extended abstract (packed CD-O by object, slot-1
offsets); the current report has no figures, and the non-technical report under
`docs/report/nontechnical/` carries the graphs and overlays.
