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

The block at the top of `main.tex` holds every number that can change at release: the `\ifmoving`
toggle (true: episode 38's learned moving rollout is in the file, nineteen episodes static; false: the
static fallback), the measured macros listed in `macros.txt` with their records, and the six
leaderboard macros `\LBCDO` to `\LBMPPE`, which read `[pending]` until the file is scored. After the
scores exist: fill them, commit, run `d3_report.sh --run`, move `release.json`'s `epoch` and `sha256`
to the new build, copy the PDF to `docs/report/final/` and commit. The moving pack's admission record
pins the hash of the `main.tex` it was admitted against, so a later wording change is a documentation
update, not a change to the packed file.

`figures/` holds the figure sources of the earlier extended abstract (packed CD-O by object, slot-1
offsets); the current report has no figures, and the non-technical report under
`docs/report/nontechnical/` carries the graphs and overlays.
