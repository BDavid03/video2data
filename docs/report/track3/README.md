# Track 3 technical report

`main.tex` is the source of our two-page technical report plus references (README step 6). It uses the
anonymous `nvidiaextabstract` template the organisers' form asks for. The PDF the form links to is
`docs/report/final/track3-technical-report.pdf`.

## Build

CPU only. The same commit gives the same bytes.

```bash
bash scripts/reproduce/d3_report.sh --run    # prints where main.pdf lands
bash scripts/reproduce/check_release.sh --combined <release manifest>   # rebuilds and compares the hash
```

`d3_report.sh` still runs `figures/make_figures.py`, but the current report has no figures. Graphs and
video overlays are in the plain-language report, `docs/report/nontechnical/`.

## After a wording change

1. Commit.
2. Run `bash scripts/reproduce/d3_report.sh --run`.
3. Set `report.epoch` and `report.sha256` in `scripts/reproduce/release.json` to the new build.
   `epoch` is the commit time of the last change under `docs/report/track3`.
4. Copy the PDF to `docs/report/final/` and commit.

The submitted moving pack's admission record hashes the `main.tex` it was checked against (at commit
`c8ff8c1`). A later wording change updates the report only. It does not change the submitted file.
