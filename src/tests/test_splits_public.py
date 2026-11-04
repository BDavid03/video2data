"""Verify the frozen split against pinned public metadata and actual filenames.

No trajectory rows, camera pixels or evaluation files are read.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "data/configs/splits/public-v1.json"


class PublicSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.snapshot = ROOT / "data/datasets--nvidia--video_to_data_challenge/snapshots" / cls.config["dataset_revision"]

    def test_partition_task_rules_and_no_duplicate_episodes(self):
        cfg = self.config
        dev, val = cfg["development_episode_indices"], cfg["validation_episode_indices"]
        self.assertEqual(len(dev), 13)
        self.assertEqual(len(val), 7)
        self.assertEqual(len(set(dev + val)), 20)
        self.assertEqual(dev, sorted(dev))
        self.assertEqual(val, sorted(val))
        self.assertEqual(sorted(i for task in cfg["tasks"] for i in task["development"]), dev)
        self.assertEqual(sorted(i for task in cfg["tasks"] for i in task["validation"]), val)
        singletons = []
        for task in cfg["tasks"]:
            ids = task["development"] + task["validation"]
            if len(ids) > 1:
                self.assertEqual(task["validation"], [max(ids)])
            else:
                self.assertEqual(task["validation"], [])
                singletons.append(task["task"])
        self.assertEqual(sorted(singletons), sorted(cfg["singleton_tasks_without_within_task_validation"]))
        self.assertEqual(cfg["source_split"], "public")
        self.assertIn("source_split", cfg["group_key"])
        self.assertEqual(cfg["camera_streams"], ["ego_cam_a", "ego_cam_b", "ego_cam_c"])

    def test_reproduce_from_pinned_metadata_and_filenames(self):
        if not self.snapshot.is_dir():
            self.skipTest("pinned snapshot unavailable; metadata/file integration check not run")
        cfg = self.config
        for path, digest in cfg["source_metadata_sha256"].items():
            self.assertEqual(hashlib.sha256((self.snapshot / path).read_bytes()).hexdigest(), digest)
        public = self.snapshot / "track_3/public"
        rows = [json.loads(line) for line in (public / "meta/episodes_metadata.jsonl").read_text().splitlines()]
        groups = defaultdict(list)
        for row in rows:
            groups[row["task"]].append(row["episode_index"])
        ids = {row["episode_index"] for row in rows}
        self.assertEqual(len(ids), len(rows))
        reproduced_val = sorted(max(group) for group in groups.values() if len(group) > 1)
        self.assertEqual(reproduced_val, cfg["validation_episode_indices"])
        self.assertEqual(sorted(ids - set(reproduced_val)), cfg["development_episode_indices"])
        info = json.loads((public / "meta/info.json").read_text())
        self.assertEqual(ids, set(info["episode_indices"]))
        task_indices = {task["task"]: task["task_index"] for task in info["tasks"]}
        for task in cfg["tasks"]:
            self.assertEqual(sorted(groups[task["task"]]), sorted(task["development"] + task["validation"]))
            self.assertEqual(task_indices[task["task"]], task["task_index"])
        self.assertEqual({int(p.stem.removeprefix("episode_")) for p in public.glob("data/chunk-*/*.parquet")}, ids)
        for camera in cfg["camera_streams"]:
            files = list(public.glob(f"videos/chunk-*/observation.images.{camera}/episode_*.mp4"))
            self.assertEqual(len(files), len(ids))
            self.assertEqual({int(p.stem.removeprefix("episode_")) for p in files}, ids)


if __name__ == "__main__":
    unittest.main()
