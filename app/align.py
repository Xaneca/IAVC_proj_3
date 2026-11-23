"""Alinhamento ArcFace partilhado pela aplicação e pelo processamento offline."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
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


def _largest_visible(image: np.ndarray, detector: FaceDetector) -> Detection | None:
    visible = (d for d in detector.detect(image) if _visible(d))
    return max(visible, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
               default=None)


def detect_and_align(image: np.ndarray, size: int, pick: str = "largest",
                     *, detector: FaceDetector | None = None,
                     retry_pad: float = 0.0, pad_first: bool = False) -> FaceCrop | None:
    """Seleciona a maior face analisável. Sem faces visíveis, devolve None.

    retry_pad > 0: se nada for encontrado, repete com uma margem preta de
    retry_pad × lado em cada borda. Serve para imagens já recortadas (UTKFace,
    AffectNet), onde a face ocupa a imagem toda e o SCRFD a 640 px falha.
    pad_first=True salta a tentativa sem margem (metade do tempo quando se
    sabe que a imagem é um recorte apertado).
    A deteção devolvida fica nas coordenadas da imagem original.
    """
    if pick != "largest":
        raise ValueError("pick must be 'largest'")
    if size not in (112, 224):
        raise ValueError("size must be 112 or 224")
    if retry_pad < 0:
        raise ValueError("retry_pad must be non-negative")
    if pad_first and retry_pad == 0:
        raise ValueError("pad_first requires retry_pad > 0")
    detector = _default_detector() if detector is None else detector
    if not pad_first:
        selected = _largest_visible(image, detector)
        if selected is not None:
            return FaceCrop(selected, True, align_face(image, selected.landmarks, size))
        if retry_pad == 0:
            return None
    margin = max(1, round(retry_pad * max(image.shape[:2])))
    padded = cv2.copyMakeBorder(image, margin, margin, margin, margin,
                                cv2.BORDER_CONSTANT, value=0)
    selected = _largest_visible(padded, detector)
    if selected is None:
        return None
    # Margem preta também no crop, como faz o norm_crop junto à borda da imagem.
    # Reflexão (olhos espelhados) e replicação (riscas) criam padrões falsos.
    crop = align_face(padded, selected.landmarks, size)
    shifted = Detection(selected.bbox - margin, selected.score, selected.landmarks - margin)
    return FaceCrop(shifted, True, crop)
