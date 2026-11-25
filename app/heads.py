"""Ramo de atributos: stages afinados do ArcFace e três cabeças MLP.

    trunk (stem + stages congelados, partilhado com a identidade)
      → encoder (cópia afinável dos stages first_stage..feature_stage)
      → MLP idade (regressão), MLP género, MLP expressão

first_stage é o primeiro stage treinável: os anteriores ficam no trunk e são
calculados uma só vez para identidade e atributos. Exemplos:
    feature_stage 3, trainable [3] → trunk = stages 1-2, encoder = stage 3
    feature_stage 4, trainable [4] → trunk = stages 1-3, encoder = stage 4
    feature_stage 3, trainable []  → trunk = stages 1-3, encoder vazio (só as MLP treinam)

O encoder é uma cópia: os stages originais continuam intactos no ArcFace, para
o embedding de identidade.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from app.iresnet import CHANNELS, IResNet50

EXPRESSIONS = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]
GENDERS = ["male", "female"]   # categorias do dataset (UTKFace), não identidade de género

# Os mesmos bins de data/splits/utkface.csv (scripts/data/prepare_utkface.py)
AGE_EDGES = [5, 10, 20, 30, 40, 50, 60, 70, 80]
AGE_BINS = ["0-4", "5-9"] + [f"{a}-{a + 9}" for a in range(10, 80, 10)] + ["80+"]


def age_to_class(age: float | np.ndarray) -> int | np.ndarray:
    """Idade (anos) → índice do bin (age_class do CSV)."""
    return np.digitize(age, AGE_EDGES)


def age_to_range(age: float) -> str:
    return AGE_BINS[int(age_to_class(age))]


def _mlp(in_channels: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(3),   # mantém a disposição espacial (cima/baixo da cara)
        nn.Flatten(),
        nn.Linear(in_channels * 9, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, out_dim),
    )


class MultiTaskModel(nn.Module):
    """Recebe a saída do trunk (stages 1..first_stage-1) e devolve as três previsões."""

    def __init__(self, arcface: IResNet50, feature_stage: int = 3,
                 trainable_stages: list[int] | None = None):
        super().__init__()
        if feature_stage not in (3, 4):
            raise ValueError("feature_stage tem de ser 3 ou 4")
        trainable_stages = sorted([feature_stage] if trainable_stages is None else trainable_stages)
        if not set(trainable_stages) <= set(range(3, feature_stage + 1)):
            raise ValueError("trainable_stages tem de estar entre 3 e feature_stage")
        self.feature_stage = feature_stage
        self.trainable_stages = trainable_stages
        # Sem stages treináveis ([]), o encoder fica vazio e as cabeças leem
        # diretamente o stage feature_stage do ArcFace (só as MLP treinam).
        self.first_stage = trainable_stages[0] if trainable_stages else feature_stage + 1
        stages = arcface.stages[self.first_stage - 1:feature_stage]
        self.encoder = nn.Sequential(*(copy.deepcopy(s) for s in stages)).requires_grad_(True)
        channels = CHANNELS[feature_stage - 1]
        self.age = _mlp(channels, 1)
        self.gender = _mlp(channels, len(GENDERS))
        self.expression = _mlp(channels, len(EXPRESSIONS))
        # A cabeça da idade prevê (idade - média) / desvio; média e desvio do treino.
        self.register_buffer("age_mean", torch.tensor(0.0))
        self.register_buffer("age_std", torch.tensor(1.0))

    def forward(self, trunk: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.encoder(trunk)
        return {"age_z": self.age(f).squeeze(1),            # idade normalizada (para a loss)
                "gender": self.gender(f),                    # logits
                "expression": self.expression(f)}            # logits

    def age_years(self, age_z: torch.Tensor) -> torch.Tensor:
        return (age_z * self.age_std + self.age_mean).clamp(0, 100)

    def checkpoint(self) -> dict:
        return {"feature_stage": self.feature_stage, "trainable_stages": self.trainable_stages,
                "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, path, arcface: IResNet50, device: str = "cpu") -> "MultiTaskModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        # Checkpoints antigos (sem trainable_stages) afinavam desde o stage 3.
        trainable = ckpt.get("trainable_stages", list(range(3, ckpt["feature_stage"] + 1)))
        model = cls(arcface.cpu(), ckpt["feature_stage"], trainable)
        model.load_state_dict(ckpt["state_dict"])
        arcface.to(device)
        return model.eval().to(device)
