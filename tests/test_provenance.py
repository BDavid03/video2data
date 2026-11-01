import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from track3.provenance import create_manifest, sha256_file, write_json
from track3.paths import output_path


class ProvenanceTests(unittest.TestCase):
    def test_cli_rejects_overwriting_source(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-m", "track3", "provenance", "--output",
                                 "configs/sources.lock.json"], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("beneath manifests/runs", result.stderr)

    def test_generated_artifacts_cannot_escape_or_overwrite_sources(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(output_path(root, "reports/test.json"), root / "reports/test.json")
        for forbidden in ("../outside.json", "datasets--example/data.json", "upstream/x", ".git/x", "."):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                output_path(root, forbidden)

    def test_hash_and_atomic_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.json"
            write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):
                write_json(path, {"invalid": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["sample.json"])

    def test_pins_and_missing_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "configs/sources.lock.json", {"upstream": {"path": "upstream", "commit": "abc"}})
            with patch("track3.provenance.git_evidence", return_value={"commit": "abc"}):
                result = create_manifest(root)
                self.assertTrue(result["upstream_checkout"]["matches_pin"])
                self.assertEqual(result["files"][0]["path"], "configs/sources.lock.json")
                with self.assertRaises(FileNotFoundError):
                    create_manifest(root, ["missing.bin"])
                with self.assertRaises(ValueError):
                    create_manifest(root, ["../outside.bin"])

    def test_matching_commit_with_dirty_worktree_is_not_clean_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "configs/sources.lock.json", {"upstream": {"path": "upstream", "commit": "abc"}})
            with patch("track3.provenance.git_evidence", return_value={"available": True, "commit": "abc", "status_porcelain": " M source.py"}):
                source = create_manifest(root)["upstream_checkout"]
            self.assertTrue(source["matches_pin"])
            self.assertFalse(source["source_is_clean_and_pinned"])


if __name__ == "__main__":
    unittest.main()
