import unittest

import cv2
import numpy as np

from app.align import align_face, align_faces, detect_and_align
from app.detector import Detection


TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)


class AlignmentTests(unittest.TestCase):
    def test_template_positions_at_both_sizes(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        matrix = cv2.getRotationMatrix2D((56, 56), 20, 1.6)
        matrix[:, 2] += 65
        landmarks = cv2.transform(TEMPLATE[None], matrix)[0]
        for point in landmarks:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 4, (255, 0, 0), -1)
        for size in (112, 224):
            with self.subTest(size=size):
                crop = align_face(image, landmarks, size)
                self.assertEqual(crop.shape, (size, size, 3))
                for x, y in np.rint(TEMPLATE * (size / 112)).astype(int):
                    np.testing.assert_array_equal(crop[y, x], [255, 0, 0])

    def test_visibility_batch_and_offline_selection(self):
        image = np.full((160, 160, 3), 50, dtype=np.uint8)
        frontal = Detection(np.array([0, 0, 100, 110]), 0.9, TEMPLATE)
        smaller = Detection(np.array([10, 10, 90, 95]), 0.9, TEMPLATE)
        side = TEMPLATE.copy()
        side[2, 0] += 15
        rejected = [
            Detection(np.array([0, 0, 39, 110]), 0.9, TEMPLATE),
            Detection(frontal.bbox, 0.49, TEMPLATE),
            Detection(np.array([0, 0, 150, 150]), 0.99, side),
            Detection(frontal.bbox, 0.9, np.zeros((5, 2))),
        ]

        class FakeDetector:
            def __init__(self, detections):
                self.detections = detections

            def detect(self, frame):
                return self.detections

        detections = [smaller, *rejected, frontal]
        batch = align_faces(image, detections, 224)
        self.assertEqual([face.analysable for face in batch], [True, False, False, False, False, True])
        self.assertTrue(all(face.crop is None for face in batch[1:-1]))
        offline = detect_and_align(image, 224, detector=FakeDetector(detections))
        self.assertIs(offline.detection, frontal)
        np.testing.assert_array_equal(offline.crop, batch[-1].crop)
        self.assertIsNone(detect_and_align(image, 112, detector=FakeDetector(rejected)))
        self.assertEqual(align_faces(image, []), [])
        with self.assertRaises(ValueError):
            align_face(image, TEMPLATE, 128)
