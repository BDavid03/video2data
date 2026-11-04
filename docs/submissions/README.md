# Counted submission files

`counted.json` records each distinct file uploaded to Kaggle, including rejected
uploads and the uploader's separate CD-O extracts. Repeated uploads share one
entry with all retained Kaggle IDs. The upload history is `docs/chain/SUBMISSION.md`;
receipt files supply the additional moving-v5 upload IDs. Missing IDs are `null`,
not guessed. Times are recorded instants, minute precision, or explicit intervals.

Each entry pins the original file's size and SHA-256, original manifest, an exact
manifest copy under `manifests/<pack>/`, recorded code URL, and an existing policy
`SHA256SUMS` where available. Missing historical file sizes come from the upload
log. Manifest paths inside old JSON are historical; the ledger names the current
disk locations. Manifest originals are never edited.

Run before each context's hand-off:

```bash
bash scripts/reproduce/check_counted.sh
```

The CPU-only checker prints `OK`, `MISSING`, or `CHANGED` for each unique file and
exits nonzero on a problem. It checks both manifest copies, policy checksum lists,
and every checkpoint named in them. It never repairs or repacks anything, and does
not check historical source-code pins or replace the release checks.

Initial inventory: six uploaded files, four manifest copies, 20 archived policies.
The D1 v5 Parquet and D2 v5 CSV are missing from `submission/`; the checker reports
two problems. The combined and moving files, both CD-O extracts, manifests and
moving-policy archive are present with the recorded hashes. Older packs have no
per-pack policy checksum list. The moving archive retains its existing name
`models/policies/track3-moving-v5/`.

After a joint candidate passes packing and manifest verification,
`src/features/pose_insurance/pack_combined_joint.py` automatically copies all its
counted policies to `models/policies/<pack>/`, writes `SHA256SUMS`, preserves its
manifest here, and appends `packed, not uploaded`. Differing existing archive
files are refused; identical repeats preserve any upload history. Uploading is
separate: update that entry's status and receipts only after a real upload.

Keep this directory and checker in Git; the lead commits new files. Large packed
files and checkpoints remain ignored on disk. The ledger detects their loss; Git
preserves the small manifests and expected hashes, not the missing binary bytes.
