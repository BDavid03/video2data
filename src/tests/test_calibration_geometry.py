import copy
import unittest
from pathlib import Path

from src.calibration import (load_calibration, validate_calibration, matmul, rigid_inverse,
                               transform_point, pixel_transform, transformed_intrinsics,
                               rotate_camera_frame_180)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/datasets--nvidia--video_to_data_challenge/snapshots/5f68335f3acc802033d1e80728c1633197521de8/track_3/public/meta/camera_calibration.json"


class GeometryTests(unittest.TestCase):
    def test_180_corners_and_involution(self):
        h = pixel_transform(1280, 800, rotation_deg=180)
        self.assertEqual(matmul(h, [[0], [0], [1]]), [[1279.], [799.], [1.]])
        self.assertEqual(matmul(h, [[1279], [799], [1]]), [[0.], [0.], [1.]])
        self.assertEqual(matmul(h, h), [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def test_resize_pixel_centers(self):
        self.assertEqual(matmul(pixel_transform(1280, 800, output_size=(640, 400)), [[0], [0], [1]]), [[-.25], [-.25], [1.]])

    def test_projection_commutes_with_image_transform(self):
        k = [[567., 0., 635.], [0., 568., 397.], [0., 0., 1.]]
        point = [[.2], [-.1], [1.]]
        h = pixel_transform(1280, 800, rotation_deg=180, output_size=(640, 400))
        actual = matmul(transformed_intrinsics(k, 1280, 800, rotation_deg=180, output_size=(640, 400)), point)
        self.assertEqual(actual, matmul(h, matmul(k, point)))

    def test_positive_focal_rotation_requires_optical_basis_change(self):
        k = [[567., 0., 635.], [0., 568., 397.], [0., 0., 1.]]
        hk = transformed_intrinsics(k, 1280, 800, rotation_deg=180)
        self.assertEqual(matmul(hk, [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), [[567., 0., 644.], [0., 568., 402.], [0., 0., 1.]])

    def test_transform_inverse_and_camera_basis(self):
        t = [[0, -1, 0, .2], [1, 0, 0, -.3], [0, 0, 1, .4], [0, 0, 0, 1]]
        p = [.5, .6, 2.]
        restored = transform_point(rigid_inverse(t), transform_point(t, p))
        for x, y in zip(p, restored):
            self.assertAlmostEqual(x, y)
        self.assertEqual(rotate_camera_frame_180(rotate_camera_frame_180(t)), t)
        changed = rotate_camera_frame_180(t)
        self.assertEqual([row[3] for row in changed[:3]], [-.2, .3, .4])

    def test_bad_inputs(self):
        for dims in ((0, 800), (1280.0, 800), (True, 800)):
            with self.assertRaises(ValueError):
                pixel_transform(*dims)
        with self.assertRaises(ValueError):
            pixel_transform(1280, 800, rotation_deg=90)
        with self.assertRaises(ValueError):
            rigid_inverse([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        with self.assertRaises(ValueError):
            matmul([[float("nan")]], [[1]])


@unittest.skipUnless(SOURCE.exists(), "pinned dataset not available")
class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.calibration = load_calibration(SOURCE)

    def test_pinned_stereo_order_and_baseline(self):
        evidence = validate_calibration(self.calibration)
        ego = evidence["stereo_pairs"]["ego"]
        self.assertEqual((ego["left"], ego["right"]), ("ego_cam_c", "ego_cam_b"))
        self.assertAlmostEqual(ego["extrinsic_translation_norm_m"], .07547102070195857, places=12)
        self.assertGreater(ego["stored_rectification_relative_rotation_max_abs_from_identity"], .03)

    def test_pinned_intrinsics_and_distortion_preserved(self):
        for name in ("ego_cam_a", "ego_cam_b", "ego_cam_c"):
            camera = self.calibration["cameras"][name]
            self.assertEqual(camera["rotation_deg"], 180)
            self.assertEqual(len(camera["distortion_coefficients"]), 14)
            self.assertTrue(any(camera["distortion_coefficients"][12:]))
        k = self.calibration["cameras"]["ego_cam_a"]["intrinsic_matrix"]
        changed = transformed_intrinsics(k, 2028, 1520, rotation_deg=180)
        self.assertAlmostEqual(changed[0][2], 1032.1956787109375)
        self.assertAlmostEqual(changed[1][2], 751.455322265625)

    def test_rejects_corrupt_calibration(self):
        for mutate in (
            lambda c: c.update(translation_units="centimeters"),
            lambda c: c["cameras"]["ego_cam_a"].update(fx=1),
            lambda c: c["cameras"]["ego_cam_a"].update(distortion_coefficients=[0]*5),
            lambda c: c["cameras"]["ego_cam_a"].update(rotation_deg=0),
        ):
            c = copy.deepcopy(self.calibration)
            mutate(c)
            with self.assertRaises(ValueError):
                validate_calibration(c)


if __name__ == "__main__":
    unittest.main()
