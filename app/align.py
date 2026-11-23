"""Alinhamento ArcFace partilhado pela aplicação e pelo processamento offline."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from insightface.utils.face_align import norm_crop

from app.detector import Detection, FaceDetector

CROP_SIZE = 112
MIN_BOX_SIZE = 40
MIN_SCORE = 0.5
MAX_NOSE_OFFSET = 0.25


@dataclass
class FaceCrop:
    detection: Detection
    analysable: bool
    crop: np.ndarray | None


def align_face(image: np.ndarray, landmarks5: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Devolve um crop BGR no template ArcFace, escalado para 112 ou 224 px."""
    if size not in (112, 224):
        raise ValueError("size must be 112 or 224")
    points = np.asarray(landmarks5, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Expected five finite landmarks with two coordinates")
    return norm_crop(image, points, image_size=size)


def align(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    return align_face(frame, landmarks)


def _visible(detection: Detection) -> bool:
    x1, y1, x2, y2 = detection.bbox
    if not (min(x2 - x1, y2 - y1) >= MIN_BOX_SIZE and detection.score >= MIN_SCORE):
        return False
    points = np.asarray(detection.landmarks)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return False
    left_eye, right_eye, nose = points[:3]
    axis = right_eye - left_eye
    squared_distance = float(axis @ axis)
    if squared_distance <= 0:
        return False
    offset = abs(float((nose - (left_eye + right_eye) / 2) @ axis)) / squared_distance
    return offset <= MAX_NOSE_OFFSET


def align_faces(image: np.ndarray, detections: list[Detection],
                size: int = CROP_SIZE) -> list[FaceCrop]:
    """Devolve um resultado por deteção, na mesma ordem."""
    if size not in (112, 224):
        raise ValueError("size must be 112 or 224")
    results = []
    for detection in detections:
        visible = _visible(detection)
        crop = align_face(image, detection.landmarks, size) if visible else None
        results.append(FaceCrop(detection, visible, crop))
    return results


@lru_cache(maxsize=1)
def _default_detector() -> FaceDetector:
    import yaml

    root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))["detector"]
    settings["model_path"] = str(root / settings["model_path"])
    return FaceDetector(**settings)


def detect_and_align(image: np.ndarray, size: int, pick: str = "largest",
                     *, detector: FaceDetector | None = None) -> FaceCrop | None:
    """Seleciona a maior face analisável. Sem faces visíveis, devolve None."""
    if pick != "largest":
        raise ValueError("pick must be 'largest'")
    if size not in (112, 224):
        raise ValueError("size must be 112 or 224")
    detector = _default_detector() if detector is None else detector
    visible = (d for d in detector.detect(image) if _visible(d))
    selected = max(visible, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                   default=None)
    if selected is None:
        return None
    return FaceCrop(selected, True, align_face(image, selected.landmarks, size))
