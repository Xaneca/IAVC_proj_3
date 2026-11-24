"""Deteção de caras com SCRFD (InsightFace)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from onnxruntime import get_available_providers
from insightface.model_zoo import get_model


@dataclass
class Detection:
    bbox: np.ndarray        # [x1, y1, x2, y2]
    score: float
    landmarks: np.ndarray   # (5, 2): olho esq., olho dir., nariz, boca esq., boca dir.


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Recebe uma imagem BGR e devolve as deteções na escala original."""
        ...


class SCRFDDetector(Detector):
    def __init__(self, model_path: str, input_size: int = 640,
                 min_score: float = 0.5, nms_threshold: float = 0.4):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Pesos em falta: {model_path}. Executa python -m scripts.download_models.")
        if input_size <= 0 or input_size % 32:
            raise ValueError("input_size tem de ser um múltiplo positivo de 32")
        if not (0 <= min_score <= 1 and 0 <= nms_threshold <= 1):
            raise ValueError("Os limiares de score e NMS têm de estar em [0, 1]")


        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.model = get_model(str(model_path), providers=providers)
        if getattr(self.model, "taskname", None) != "detection" or not self.model.use_kps:
            raise ValueError(f"O modelo {model_path} não é um detetor com cinco landmarks")
        self.provider = self.model.session.get_providers()[0]
        self.model.prepare(
            ctx_id=0 if self.provider == "CUDAExecutionProvider" else -1,
            input_size=(input_size, input_size),
            det_thresh=min_score,
            nms_thresh=nms_threshold,
        )
        if self.model.input_size != (input_size, input_size):
            raise ValueError(f"O modelo {model_path} não suporta resolução {input_size} × {input_size}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """frame: imagem BGR. Devolve todas as caras detetadas."""
        bboxes, kpss = self.model.detect(frame, max_num=0)
        return [
            Detection(bbox=b[:4], score=float(b[4]), landmarks=k)
            for b, k in zip(bboxes, kpss, strict=True)
        ]


FaceDetector = SCRFDDetector
