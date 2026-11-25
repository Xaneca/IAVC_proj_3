"""Converte o ArcFace R50 (w600k_r50.onnx, InsightFace) para PyTorch.

O ONNX não se treina, e o treino afina o stage 3. Este script copia os pesos
para a iResNet-50 de app/iresnet.py e verifica que o embedding é igual ao do
ONNX original (onnxruntime).

Correr: python -m scripts.export_backbone
Saída:  models/arcface_r50.pt (state_dict)
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from app.iresnet import IResNet50, weights_from_onnx

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "models/w600k_r50.onnx"
TARGET = REPO / "models/arcface_r50.pt"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Falta {SOURCE}. Executa python -m scripts.download_models.")
    model = IResNet50()
    weights_from_onnx(model, SOURCE)
    model.eval()

    x = np.random.default_rng(0).uniform(-1, 1, (4, 3, 112, 112)).astype(np.float32)
    session = ort.InferenceSession(str(SOURCE), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    reference = np.concatenate([session.run(None, {name: x[i:i + 1]})[0] for i in range(len(x))])
    with torch.no_grad():
        embedding = model(torch.from_numpy(x)).numpy()
    diff = float(np.abs(embedding - reference).max())
    if diff > 1e-3:
        raise AssertionError(f"Embedding diferente do ONNX (máx. {diff:.2e})")

    torch.save(model.state_dict(), TARGET)
    print(f"Guardado: {TARGET} (diferença máx. para o ONNX {diff:.1e})")


if __name__ == "__main__":
    main()
