"""Backbone ArcFace partilhado (iResNet-50, congelado).

Uma única passagem por cara devolve:
  - "layer3":    features intermédias (256 × 14 × 14) para as cabeças de atributos
  - "embedding": vetor de identidade (512-d, normalizado) para a galeria
"""
import numpy as np
import torch
import torch.nn.functional as F


class SharedBackbone:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = device

        # TODO: copiar backbones/iresnet.py do repositório insightface/recognition/arcface_torch
        #       e carregar os pesos:
        #   from app.iresnet import iresnet50
        #   self.model = iresnet50()
        #   self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model = None
        raise NotImplementedError("Carregar a iResNet-50 do arcface_torch")

        self.model.eval().to(device)
        self._feats = {}
        self.model.layer3.register_forward_hook(lambda m, i, out: self._feats.update(layer3=out))

    @torch.no_grad()
    def __call__(self, crops: list[np.ndarray]) -> dict[str, torch.Tensor]:
        x = preprocess(crops).to(self.device)
        emb = self.model(x)
        return {"layer3": self._feats["layer3"], "embedding": F.normalize(emb)}


def preprocess(crops: list[np.ndarray]) -> torch.Tensor:
    """Crops BGR uint8 (112×112) → tensor (B, 3, 112, 112) normalizado como no ArcFace."""
    x = np.stack([c[:, :, ::-1] for c in crops])            # BGR → RGB
    x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2).float()
    return x.div(255).sub(0.5).div(0.5)
