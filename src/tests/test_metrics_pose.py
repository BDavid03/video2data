"""Analytic transform/validity checks, independent of challenge labels."""

import math
import unittest

from src.metrics import Pose, coverage, pose_error, relative_pose


class PoseMetricsTests(unittest.TestCase):
    def setUp(self):
        self.identity = Pose((0, 0, 0), (1, 0, 0, 0))
        self.quarter_turn = Pose((1, 2, 3), (math.sqrt(0.5), 0, 0, math.sqrt(0.5)))

    def assertPoseEqual(self, a, b):
        translation, angle = pose_error(a, b)
        self.assertAlmostEqual(translation, 0, places=12)
        self.assertAlmostEqual(angle, 0, places=12)

    def test_zero_invalid_rows_filtered_before_quaternion_use(self):
        self.assertIsNone(Pose.from_row([0] * 7, valid=False))
        with self.assertRaises(ValueError):
            Pose.from_row([0] * 7, valid=True)
        self.assertPoseEqual(Pose.from_row([0, 0, 0, 1, 0, 0, 0], valid=True), self.identity)

    def test_malformed_flagged_valid_rows_rejected(self):
        for pose in ([0] * 6, [0, 0, float("nan"), 1, 0, 0, 0],
                     [0, 0, 0, float("inf"), 0, 0, 0],
                     [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, "1", 0, 0, 0]):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                Pose.from_row(pose, valid=True)
        with self.assertRaises(ValueError):
            Pose.from_row([0] * 7, valid=1)

    def test_small_norm_roundoff_is_normalized(self):
        self.assertPoseEqual(Pose((0, 0, 0), (1.0001, 0, 0, 0)), self.identity)

    def test_quaternion_sign_equivalence_and_half_turn(self):
        self.assertEqual(pose_error(self.identity, Pose((0, 0, 0), (-1, 0, 0, 0))), (0, 0))
        distance, angle = pose_error(self.identity, Pose((3, 4, 0), (0, 1, 0, 0)))
        self.assertEqual(distance, 5)
        self.assertEqual(angle, math.pi)

    def test_small_angle_precision(self):
        angle = 1e-10
        q = (math.cos(angle / 2), 0, math.sin(angle / 2), 0)
        self.assertAlmostEqual(pose_error(self.identity, Pose((0, 0, 0), q))[1], angle, places=20)

    def test_composition_direction_rotates_translation(self):
        local = Pose((1, 0, 0), (1, 0, 0, 0))
        result = self.quarter_turn.compose(local)
        expected = Pose((1, 3, 3), self.quarter_turn.quaternion)
        self.assertPoseEqual(result, expected)
        self.assertGreater(pose_error(result, local.compose(self.quarter_turn))[0], 1)

    def test_inverse_both_directions(self):
        self.assertPoseEqual(self.quarter_turn.compose(self.quarter_turn.inverse()), self.identity)
        self.assertPoseEqual(self.quarter_turn.inverse().compose(self.quarter_turn), self.identity)

    def test_relative_motion_invariant_to_fixed_world_gauge(self):
        end = Pose((2, 0, 0), (0, 0, 1, 0))
        original = relative_pose(self.identity, end)
        transformed = relative_pose(self.quarter_turn.compose(self.identity), self.quarter_turn.compose(end))
        self.assertPoseEqual(original, transformed)

    def test_relative_motion_exposes_end_drift(self):
        reference = relative_pose(self.identity, Pose((1, 0, 0), (1, 0, 0, 0)))
        prediction = relative_pose(self.identity, Pose((1.5, 0, 0), (1, 0, 0, 0)))
        self.assertEqual(pose_error(reference, prediction), (0.5, 0))

    def test_coverage_denominator_and_empty_support(self):
        self.assertEqual(coverage(3, 4), 0.75)
        self.assertEqual(coverage(0, 4), 0)
        self.assertIsNone(coverage(0, 0))
        for counts in ((1, 0), (-1, 1), (0, -1), (True, 1), (1.5, 2)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                coverage(*counts)


if __name__ == "__main__":
    unittest.main()
