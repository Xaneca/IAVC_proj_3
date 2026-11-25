"""Backbone partilhado: ArcFace iResNet-50 com dois ramos a partir do primeiro stage afinado.

    crop 112×112 → stem + stages congelados (partilhados)
                     ├─ stages restantes originais + fc → embedding (identidade)
                     └─ stages afinados → cabeças de idade, género e expressão

O ramo de identidade fica com os pesos originais, para o fine-tuning não
estragar o embedding. Custo: os stages afinados correm duas vezes (só o
stage 4, ~0,8 GMAC, se for o único afinado; o stage 3, ~3,4 GMAC, se for afinado).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.heads import MultiTaskModel
from app.iresnet import load_arcface


def preprocess(crops: list[np.ndarray]) -> torch.Tensor:
    """Crops BGR uint8 (112×112) → tensor (B, 3, 112, 112) normalizado como no ArcFace."""
    x = np.stack([c[:, :, ::-1] for c in crops]).astype(np.float32)   # BGR → RGB
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
    return (x / 255.0 - 0.5) / 0.5


class SharedBackbone:
    def __init__(self, arcface_weights: str, heads_weights: str | None, device: str = "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA indisponível: a usar CPU.")
            device = "cpu"
        self.device = device
        self.arcface = load_arcface(arcface_weights, device)
        self.heads = None
        if heads_weights and Path(heads_weights).is_file():
            self.heads = MultiTaskModel.from_checkpoint(heads_weights, self.arcface, device)
        elif heads_weights:
            print(f"Cabeças em falta ({heads_weights}): só identidade. "
                  "Treina com python -m train.train_multitask.")

    @torch.no_grad()
    def __call__(self, crops: list[np.ndarray], identity: bool = True,
                 attributes: bool = True) -> dict[str, torch.Tensor]:
        """Devolve embedding (L2) e, se houver cabeças, idade (anos) e probabilidades."""
        last_shared = 2 if self.heads is None else self.heads.first_stage - 1
        trunk = self.arcface.trunk(preprocess(crops).to(self.device), last_shared)
        out = {}
        if identity:
            out["embedding"] = F.normalize(self.arcface.embed(trunk, last_shared)).cpu()
        if attributes and self.heads is not None:
            pred = self.heads(trunk)
            out["age"] = self.heads.age_years(pred["age_z"]).cpu()
            out["gender"] = pred["gender"].softmax(1).cpu()
            out["expression"] = pred["expression"].softmax(1).cpu()
        return out
