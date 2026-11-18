"""Deteção de caras com SCRFD (InsightFace)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray        # [x1, y1, x2, y2]
    score: float
    landmarks: np.ndarray   # (5, 2): olho esq., olho dir., nariz, boca esq., boca dir.


class FaceDetector:
    def __init__(self, model_path: str, input_size: int = 640, min_score: float = 0.5):
        from insightface.model_zoo import get_model

        self.model = get_model(model_path)
        self.model.prepare(ctx_id=-1, input_size=(input_size, input_size), det_thresh=min_score)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """frame: imagem BGR. Devolve todas as caras detetadas."""
        bboxes, kpss = self.model.detect(frame, max_num=0)
        return [
            Detection(bbox=b[:4], score=float(b[4]), landmarks=k)
            for b, k in zip(bboxes, kpss)
        ]
