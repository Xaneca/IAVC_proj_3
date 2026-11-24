"""Backbone ArcFace partilhado (iResNet-50, pesos pré-treinados, congelado).

Usa models/arcface_r50_l3.onnx, gerado por scripts/export_backbone.py a partir
do w600k_r50.onnx (InsightFace). Uma única passagem por cara devolve:
  - "layer3":    features do fim do stage 3 (256 × 14 × 14) para as cabeças de atributos
  - "embedding": vetor de identidade (512-d, normalizado L2) para a galeria
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F


class SharedBackbone:
    def __init__(self, weights_path: str, device: str = "cpu"):
        if not Path(weights_path).is_file():
            raise FileNotFoundError(
                f"Backbone em falta: {weights_path}. Executa python -m scripts.download_models "
                "e python -m scripts.export_backbone.")
        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(weights_path), providers=providers)
        outputs = {o.name for o in self.session.get_outputs()}
        if outputs != {"layer3", "embedding"}:
            raise ValueError(f"{weights_path} não expõe layer3 e embedding (saídas: {sorted(outputs)})")

    def run(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Crops BGR 112×112 → (layer3 (B,256,14,14), embedding (B,512) sem normalizar)."""
        layer3, embedding = self.session.run(["layer3", "embedding"], {"input": preprocess(crops)})
        return layer3, embedding

    @torch.no_grad()
    def __call__(self, crops: list[np.ndarray]) -> dict[str, torch.Tensor]:
        layer3, embedding = self.run(crops)
        return {"layer3": torch.from_numpy(layer3),
                "embedding": F.normalize(torch.from_numpy(embedding))}


def preprocess(crops: list[np.ndarray]) -> np.ndarray:
    """Crops BGR uint8 (112×112) → array (B, 3, 112, 112) normalizado como no ArcFace."""
    x = np.stack([c[:, :, ::-1] for c in crops]).astype(np.float32)   # BGR → RGB
    x = x.transpose(0, 3, 1, 2)
    return np.ascontiguousarray((x / 255.0 - 0.5) / 0.5)
