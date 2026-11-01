"""Render the extended-abstract LaTeX report from the ledger and (optionally) a finalize result.

The template at templates/V2D_Challenge_Extended_Abstract_Template is copied
verbatim (class, style, bibliography, assets) and main.tex is generated. If
pdflatex is on PATH the PDF is built; otherwise the source is left ready to build.
"""

from dataclasses import dataclass
import shutil
import subprocess

from track3.paths import project_path
from ._common import ROOT, authorize, new_run_id, now, read_json, relative, save, state_dir
from . import ledger

TEMPLATE = "templates/V2D_Challenge_Extended_Abstract_Template"


@dataclass(frozen=True)
class ReportRequest:
    finalize_result: str | None = None
    team: str = "video2data-local"
    title: str = "Closed-Loop Scratch Depth Recipes Against MoGe2 Pseudo-Targets"

    @classmethod
    def from_file(cls, path):
        return cls(**read_json(path))


def _tex(value):
    return str(value).replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def _fmt(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "--"


def _results_table(entries, final):
    rows = []
    best_per = {}
    for e in ledger.completed(entries):
        if e["technique"] not in best_per or e["tuning_log_mae"] < best_per[e["technique"]]["tuning_log_mae"]:
            best_per[e["technique"]] = e
    holdout = {}
    if final:
        for c in final["candidates"]:
            holdout[c["name"]] = c
    for technique, e in sorted(best_per.items(), key=lambda kv: kv[1]["tuning_log_mae"]):
        name = f"iteration_{e['iteration']}_{e['technique']}"
        h = holdout.get(name, {}).get("holdout", {})
        rows.append((f"{_tex(technique)} (iter {e['iteration']})", _fmt(e["tuning_log_mae"]), _fmt(e["abs_rel"]),
                     _fmt(e["delta1"], 3), _fmt(h.get("score"))))
    if final:
        for c in final["candidates"]:
            if c["technique"] == "ensemble":
                h = c.get("holdout", {})
                rows.append((f"ensemble {_tex(c['name'])}", _fmt(c["tuning"].get("score")), "--", "--", _fmt(h.get("score"))))
    body = "\n".join(f"    {name} & {t} & {ar} & {d1} & {ho} \\\\" for name, t, ar, d1, ho in rows)
    return body


def write_report(request: ReportRequest):
    authorize("report")
    entries = ledger.load()
    if not entries:
        raise RuntimeError("No ledger entries to report")
    final = read_json(request.finalize_result) if request.finalize_result else None
    directory = state_dir() / "report" / new_run_id()
    directory.mkdir(parents=True)
    template = project_path(ROOT, TEMPLATE)
    for name in ("nvidiaextabstract.cls", "nvidiatechreport.cls", "natbib.sty", "references.bib"):
        shutil.copy2(template / name, directory / name)
    shutil.copytree(template / "assets", directory / "assets")
    for figure in ("dashboard.png", "best-samples.png"):
        source = state_dir() / figure
        if source.exists():
            shutil.copy2(source, directory / "assets" / figure)
    done = ledger.completed(entries)
    best = ledger.best(entries)
    techniques = sorted({e["technique"] for e in entries})
    sources = {s: sum(e["source"] == s for e in entries) for s in ("baseline", "codex", "heuristic")}
    winner_line = ""
    if final:
        sel = next(c for c in final["candidates"] if c["name"] == final["selected_candidate"])
        winner_line = (f"The frozen tuning-selected candidate was \\texttt{{{_tex(sel['name'])}}} with tuning MAE "
                       f"{_fmt(sel['tuning'].get('score'))} and holdout MAE {_fmt(sel.get('holdout', {}).get('score'))} "
                       f"on episodes 43/44, assessed once after the selection was frozen.")
    tex = f"""\\pdfoutput=1
\\documentclass[10pt]{{nvidiaextabstract}}
\\usepackage{{microtype}}
\\usepackage{{multirow}}
\\usepackage{{subcaption}}
\\usepackage{{xspace}}
\\venue{{From Human Videos to Dexterous Robot Capability (CoRL 2026)}}
\\submissionID{{{_tex(request.team)}}}
\\title{{{_tex(request.title)}}}
\\author[1]{{{_tex(request.team)}}}
\\affil[1]{{Local RTX 5080 research loop}}
\\begin{{abstract}}
\\textbf{{Abstract:}}
We study how far small depth networks trained from random initialisation on MoGe2 pseudo-depth can go on
egocentric Track~3 video, using a closed experiment loop in which each recipe is run, recorded, visualised and
examined by an LLM advisor before the next is proposed. Over {len(entries)} iterations
({sources['baseline']} baselines, {sources['codex']} advisor proposals, {sources['heuristic']} heuristic mutations)
the best tuning log-depth MAE reached {_fmt(best['tuning_log_mae']) if best else '--'}
({_tex(best['description']) if best else ''}). Selection used tuning episodes 39/42 only. {winner_line}
\\end{{abstract}}
\\definecolor{{NvidiaGreen}}{{RGB}}{{118, 185, 0}}
\\newcommand{{\\method}}{{\\textsc{{AutoLoop}}\\xspace}}
\\begin{{document}}
\\maketitle

\\section{{Introduction}}
\\label{{sec:intro}}
Metric depth from a single egocentric RGB stream is the first stage of the video-to-data pipeline for Track~3.
Rather than depend on a large pretrained estimator at runtime, we ask what accuracy small original networks reach when
distilled from MoGe2 pseudo-labels, and we automate the experimental method itself: \\method closes the loop between
running a recipe, recording its metrics, visualising the ledger and asking an advisor model to examine the evidence
and propose the next bounded change. Contributions:
\\begin{{itemize}}
  \\item a bounded recipe space over {', '.join(_tex(t) for t in techniques)} models with validated hyperparameter,
        optimisation and data-preparation ranges;
  \\item a circular pipeline that records equal-episode log-depth MAE together with AbsRel, RMSE-log, SILog,
        $\\delta_1$, bias, support, generalisation gap, convergence speed and cost for every iteration;
  \\item a frozen tuning-only selection with a single holdout assessment after freezing.
\\end{{itemize}}

\\section{{Approach}}
\\label{{sec:approach}}
Each iteration takes a recipe $r=(\\text{{technique}}, \\theta_{{\\text{{arch}}}}, \\theta_{{\\text{{opt}}}}, \\theta_{{\\text{{data}}}})$,
generates (once per data setting) MoGe2 teacher depth at up to 768\\,px on sparse frames of the ten training/tuning
episodes, prepares calibrated RGB/depth pairs at the requested side, trains from random weights and predicts the tuning
frames. The selection objective is
\\begin{{equation}}
\\label{{eq:objective}}
J(r) = \\frac{{1}}{{|E|}} \\sum_{{e \\in E}} \\frac{{1}}{{|P_e|}} \\sum_{{p \\in P_e}} \\big| \\log \\hat d_p - \\log d_p \\big|,
\\end{{equation}}
with $E=\\{{39,42\\}}$ and $P_e$ the supported pixels of episode $e$. After each run the ledger and dashboard are
regenerated and an advisor (Codex, high reasoning, read-only) returns a hypothesis and one to three proposals, validated
against the bounds before execution; a local mutation search is the fallback. Holdout episodes 43/44 are never read
during the loop.

\\section{{Experiments and Results}}
\\label{{sec:experiments}}
{len(entries)} recipes ran on one RTX~5080; {len(done)} completed and {len(entries) - len(done)} failed and were recorded as
outcomes. Table~\\ref{{tab:results}} lists the best recipe per technique and the tuning-selected ensembles;
Figure~\\ref{{fig:overview}} shows the loop trajectory and companion metrics. All scores compare to MoGe2
pseudo-labels, not measured ground truth, and two tuning episodes give weak statistical support.

\\begin{{table}}[t]
  \\centering
  \\begin{{tabular}}{{lcccc}}
    \\toprule
    \\textit{{Recipe}} & \\textit{{Tuning log-MAE}} & \\textit{{AbsRel}} & \\textit{{$\\delta_1$}} & \\textit{{Holdout log-MAE}} \\\\
    \\midrule
{_results_table(entries, final)}
    \\bottomrule
  \\end{{tabular}}
  \\caption{{Best completed recipe per technique on tuning episodes 39/42; holdout 43/44 only for frozen candidates.}}
  \\label{{tab:results}}
\\end{{table}}

\\begin{{figure}}[t]
  \\centering
  \\includegraphics[width=0.95\\linewidth]{{assets/dashboard.png}}
  \\caption{{Loop dashboard: best-so-far trajectory by technique, accuracy versus training cost, per-episode MAE and
  companion metrics of the leading recipes.}}
  \\label{{fig:overview}}
  \\Description{{Four-panel dashboard of the automated research loop.}}
\\end{{figure}}

\\balance

\\section{{Conclusion}}
\\label{{sec:conclusion}}
{_tex(best['description']) if best else 'No recipe completed.'} gave the lowest tuning log-depth MAE
({_fmt(best['tuning_log_mae']) if best else '--'}). {winner_line} The loop's value is in the recorded, examined
trajectory rather than any single number: the next direction is replacing pseudo-labels with independent geometric
evidence and scaling teacher frames beyond sparse sampling.

\\bibliographystyle{{plainnat}}
\\bibliography{{references}}

\\clearpage
\\appendix
\\section{{Appendix}}
\\label{{sec:appendix}}
\\subsection{{Iteration ledger}}
\\begin{{tabular}}{{llp{{6.5cm}}cc}}
\\toprule
iter & source & recipe & tuning log-MAE & status \\\\
\\midrule
""" + "\n".join(f"{e['iteration']} & {e['source']} & \\small {_tex(e['description'])} & {_fmt(e.get('tuning_log_mae'))} & {e['status']} \\\\"
                for e in entries) + """
\\bottomrule
\\end{tabular}
\\end{document}
"""
    (directory / "main.tex").write_text(tex, encoding="utf-8")
    record = {"run_id": directory.name, "created_utc": now(), "request": request.__dict__, "ledger_entries": len(entries),
              "finalize_result": request.finalize_result, "main_tex": relative(directory / "main.tex"), "pdf": None}
    if shutil.which("pdflatex"):
        for command in (["pdflatex", "-interaction=nonstopmode", "main"], ["bibtex", "main"],
                        ["pdflatex", "-interaction=nonstopmode", "main"], ["pdflatex", "-interaction=nonstopmode", "main"]):
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=600)
            record.setdefault("build", []).append({"command": command, "exit_code": result.returncode})
        if (directory / "main.pdf").exists():
            record["pdf"] = relative(directory / "main.pdf")
    else:
        record["build"] = "pdflatex not on PATH; main.tex is ready to build"
    save(directory / "run.json", record)
    return directory / "main.tex"
