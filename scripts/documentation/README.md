# Compile project documentation

`compile.py` combines the project's Markdown documentation into one file using
only the Python standard library (Python 3.9 or newer). It imports no project
modules, starts no subprocesses, installs nothing and makes no network requests.
The phase-2 documentation restructure ran this compiler successfully; this
provides documentation-generation evidence only.

From the project root, the reference invocation is:

```text
python scripts/documentation/compile.py
```

The default output is `docs/compiled/all-documentation.md`. A custom name can be
specified inside the same generated-document directory:

```text
python scripts/documentation/compile.py --output docs/compiled/project-handbook.md
```

Use `python3` instead of `python` where that is the installed interpreter name.
The script finds the project root relative to its own location, so an absolute
script invocation also works from another working directory. Relative output
paths always refer to the project root. No pipeline operation or runtime flag
is involved.

## Included documentation

- All root-level `.md` files, including the README and current instructions.
- The historical plan at `docs/design/legacy-plan.md` and the research backlog
  under `docs/experiments/`, discovered recursively through `docs/`.
- Markdown recursively under `docs/`, `src/`, `data/configs/`, `scripts/` and
  `data/reports/`, including feature READMEs and historical review snapshots.
- This script's own documentation.

The main README and design pages appear first in a fixed reading order. Remaining
documents are sorted by their project-relative path. Historical copies remain
separate sections labelled with their full source paths; their inclusion does
not make them current guidance.

Vendored upstream documentation, datasets, models, environments, caches,
coursework notebooks, generated pipeline outputs and non-Markdown files are
outside the input scope. Hidden directories, symlinks, `compiled/` directories
and previous files bearing this compiler's marker are skipped. Adding another
project documentation tree requires updating `DOCUMENT_DIRECTORIES` in this
same slice. The scope deliberately means project-owned documentation, not every
README shipped by a third-party dependency.

## Output contract

The collection contains a table of contents with stable per-file anchors, source
links relative to the output file, source byte hashes and the document bodies.
It has no generated timestamp, so unchanged source bytes and the same output
location produce stable content. Sources are read in sequence, not locked as a
transactional repository snapshot; avoid concurrent documentation edits while
compiling if one consistent revision is required.

Bodies retain their headings, Markdown links, code fences and historical content.
This is a concatenation utility, not a Markdown parser or renderer: relative body
links, reference labels and duplicate heading anchors are not rewritten or
namespaced. Use each section's **Open original document** link for navigation in
the original document context. The table of contents uses independent anchors.
Images remain external references; their bytes are not embedded.

UTF-8 BOMs and line endings are normalized. NUL padding is removed from the
collected text with a visible per-source note; unreadable/empty content gets a
placeholder instead of disappearing. Invalid UTF-8 fails the operation. The
recorded SHA-256 covers the original bytes, before these text transformations.
Source files are never modified.

Output must remain under this checkout's `docs/compiled/`, inside the agreed
project storage (`Z:/hugging-face/video2data` or its workspace mapping). A symlink
destination or a pre-existing file without the compiler marker is refused.
Repeated compilation can replace this utility's own output. The new file is
written to a temporary file in the destination directory and published with an
atomic replacement after all sources have been read; a failed read leaves the
previous compilation intact. Temporary files are removed on ordinary exceptions.

Compiling documentation provides no testing, validation, model, training,
simulation or submission evidence. Current project execution restrictions remain
unchanged.
