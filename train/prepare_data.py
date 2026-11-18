"""Prepara os datasets para treino e avaliação.

1. Alinha as caras de UTKFace / AffectNet / BTFER com o MESMO detector + alinhamento da app
   (data/raw/<dataset> → data/aligned/<dataset>).
2. Cria os splits train/val/test em data/splits/<dataset>.csv (seed fixa, sem imagens repetidas).

Correr: python -m train.prepare_data
"""
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from app.align import align
from app.detector import FaceDetector

RAW, ALIGNED, SPLITS = Path("data/raw"), Path("data/aligned"), Path("data/splits")
SEED = 42


def align_folder(src: Path, dst: Path, detector: FaceDetector) -> None:
    """Deteta a maior cara de cada imagem e guarda o crop alinhado 112×112."""
    for path in tqdm(sorted(src.rglob("*.jpg")), desc=src.name):
        out = dst / path.relative_to(src)
        if out.exists():
            continue
        img = cv2.imread(str(path))
        dets = detector.detect(img) if img is not None else []
        if not dets:
            continue   # TODO: registar quantas imagens falharam (vai para o relatório)
        d = max(dets, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), align(img, d.landmarks))


def make_splits(dataset: str) -> None:
    """TODO: ler as labels de cada dataset e escrever data/splits/<dataset>.csv
    com colunas: path, label(s), split.

    - UTKFace: labels no nome do ficheiro ([age]_[gender]_[race]_[date].jpg); split 70/15/15
    - AffectNet: usar as partições oficiais (train / val)
    - BTFER: só teste (avaliação cross-dataset)
    """
    raise NotImplementedError


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    detector = FaceDetector(**cfg["detector"])
    for dataset in ["utkface", "affectnet", "btfer"]:
        align_folder(RAW / dataset, ALIGNED / dataset, detector)
        make_splits(dataset)


if __name__ == "__main__":
    main()
