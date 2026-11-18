"""Galeria de identidades: registar, apagar, reconhecer.

Guarda APENAS embeddings (não fotografias), um ficheiro .npy por pessoa.
"""
from pathlib import Path

import numpy as np

UNKNOWN = "Unknown"


class Gallery:
    def __init__(self, directory: str, threshold: float):
        self.dir = Path(directory)
        self.dir.mkdir(exist_ok=True)
        self.threshold = threshold
        self.templates: dict[str, np.ndarray] = {
            p.stem: np.load(p) for p in self.dir.glob("*.npy")
        }

    def enrol(self, name: str, embeddings: np.ndarray) -> None:
        """embeddings: (N, 512) de várias imagens da mesma pessoa → template médio."""
        template = embeddings.mean(axis=0)
        template /= np.linalg.norm(template)
        self.templates[name] = template
        np.save(self.dir / f"{name}.npy", template)

    def delete(self, name: str) -> None:
        self.templates.pop(name, None)
        (self.dir / f"{name}.npy").unlink(missing_ok=True)

    def clear(self) -> None:
        for name in list(self.templates):
            self.delete(name)

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Devolve (nome, similaridade), ou (Unknown, similaridade) abaixo do threshold."""
        if not self.templates:
            return UNKNOWN, 0.0
        names = list(self.templates)
        sims = np.stack([self.templates[n] for n in names]) @ embedding
        best = int(np.argmax(sims))
        score = float(sims[best])
        return (names[best] if score >= self.threshold else UNKNOWN), score
