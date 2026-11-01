import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from track3.data.audit import audit_failures, timestamp_summary, verify_file
from track3.data.episodes import discover_episodes, iter_public_frames, validate_pose


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(__file__).resolve().parents[1] / "reports/data-audit/test-tmp"
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=self.temp_root)

    def test_sparse_episode_discovery_does_not_expand_split_range(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            meta = root / "track_3/public/meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text(json.dumps({"splits": {"train": "0:46"}}))
            (meta / "episodes_metadata.jsonl").write_text('\n'.join(json.dumps({"episode_index": i, "task": "Pan_Solo"}) for i in [0, 42]))
            for index in [0, 42]:
                trajectory = root / f"track_3/public/data/chunk-000/episode_{index:06d}.parquet"
                trajectory.parent.mkdir(parents=True, exist_ok=True)
                trajectory.touch()
                for camera in "abc":
                    video = root / f"track_3/public/videos/chunk-000/observation.images.ego_cam_{camera}/episode_{index:06d}.mp4"
                    video.parent.mkdir(parents=True, exist_ok=True)
                    video.touch()
            records = discover_episodes(root)
            self.assertEqual([r["episode_index"] for r in records], [0, 42])
            self.assertTrue(all(not r["issues"] for r in records))

    def test_metadata_only_episode_reports_missing_files(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            meta = root / "track_3/public/meta"
            meta.mkdir(parents=True)
            (meta / "episodes_metadata.jsonl").write_text('{"episode_index": 9, "task": "Cup"}\n')
            record = discover_episodes(root)[0]
            self.assertIn("missing_public_trajectory", record["issues"])
            self.assertIn("missing_camera:ego_cam_a", record["issues"])

    def test_evaluation_never_reads_trajectory_or_assigns_task(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            video = root / "track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a/episode_000003.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
            forbidden = root / "track_3/evaluation/data/chunk-000/episode_000003.parquet"
            forbidden.parent.mkdir(parents=True)
            forbidden.write_bytes(b"should not be read")
            record = discover_episodes(root)[0]
            self.assertIsNone(record["trajectory"])
            self.assertIsNone(record["task"])
            with self.assertRaises(ValueError):
                list(iter_public_frames(root, record))

    def test_unknown_camera_is_reported(self):
        with self.temporary_directory() as directory:
            root = Path(directory)
            video = root / "track_3/evaluation/videos/chunk-000/observation.images.unknown/episode_000003.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
            self.assertIn("unknown_camera:unknown", discover_episodes(root)[0]["issues"])

    def test_lfs_checksum_and_corruption(self):
        with self.temporary_directory() as directory:
            path = Path(directory) / "asset"
            path.write_bytes(b"data")
            expected = {"size": 4, "lfs_sha256": hashlib.sha256(b"data").hexdigest(), "blob_id": "pointerhash"}
            self.assertEqual(verify_file(path, expected)["status"], "ok")
            path.write_bytes(b"oops")
            self.assertEqual(verify_file(path, expected)["status"], "mismatch")
            path.unlink()
            self.assertEqual(verify_file(path, expected)["status"], "missing")

    def test_git_blob_hash_includes_header(self):
        with self.temporary_directory() as directory:
            path = Path(directory) / "asset"
            path.write_bytes(b"data")
            expected = {"size": 4, "blob_id": hashlib.sha1(b"blob 4\0data").hexdigest()}
            self.assertEqual(verify_file(path, expected)["status"], "ok")
            expected["blob_id"] = hashlib.sha1(b"data").hexdigest()
            self.assertEqual(verify_file(path, expected)["status"], "mismatch")

    def test_invalid_zero_pose_never_valid(self):
        self.assertEqual(validate_pose([0.] * 7, True)["reason"], "zero_pose")
        self.assertFalse(validate_pose([0.] * 7, False)["valid"])
        self.assertFalse(validate_pose([1, 2, 3, 0, 0, 0, 0], True)["valid"])
        self.assertFalse(validate_pose([1, 2, 3, float("nan"), 0, 0, 0], True)["valid"])
        self.assertTrue(validate_pose([0, 0, 0, 1, 0, 0, 0], True)["valid"])
        self.assertFalse(validate_pose([0, 0, 0, 1, 0, 0, 0], False)["valid"])
        self.assertFalse(validate_pose([0, 0, 0, 1.005, 0, 0, 0], True)["valid"])
        self.assertFalse(validate_pose([0, 0, 0, True, 0, 0, 0], True)["valid"])
        self.assertFalse(validate_pose([0, 0, 0, 1, 0, 0, 0], "false")["valid"])

    def test_timestamp_duplicates_and_reversal(self):
        result = timestamp_summary([0, 0.05, 0.05, 0.02])
        self.assertEqual(result["duplicate_count"], 1)
        self.assertEqual(result["backwards_count"], 1)

    def test_incomplete_and_broken_audits_fail_closed(self):
        good = {"track3_file_count": 1, "integrity_counts": {"ok": 1}, "episode_issues": [],
                "trajectory_status": "audited", "trajectory_issues": [], "video_requested": True,
                "video_status": "completed", "video_count": 3, "expected_video_count": 3, "video_issues": [],
                "episode_counts": {"public": 1}, "pairing": [{"frame_counts_equal": True, "container_pts_equal": True, "public_rows_match_video_frames": True}]}
        self.assertEqual(audit_failures(good), [])
        for field, broken in [("trajectory_status", "blocked_missing_pyarrow"), ("video_status", "blocked_missing_ffmpeg"),
                              ("video_count", 2), ("video_issues", [{"errors": ["bad timestamp"]}]),
                              ("trajectory_issues", [{"issues": ["invalid_visible_poses:1"]}]), ("pairing", [])]:
            value = copy.deepcopy(good)
            value[field] = broken
            self.assertTrue(audit_failures(value), field)
        value = copy.deepcopy(good)
        value["pairing"][0]["container_pts_equal"] = False
        self.assertIn("frame_pairing", audit_failures(value))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Optional installed parquet reader unavailable")
    def test_adapter_refuses_ambiguous_pose_convention(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with self.temporary_directory() as directory:
            root = Path(directory)
            relative = "track_3/public/data/chunk-000/episode_000000.parquet"
            path = root / relative
            path.parent.mkdir(parents=True)
            table = pa.table({"frame_index": [0], "episode_index": [0]})
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, "world_T_object"):
                list(iter_public_frames(root, {"split": "public", "trajectory": relative, "episode_index": 0}))


if __name__ == "__main__":
    unittest.main()
