"""Gera os crops de treino redimensionando as imagens para size×size.

Os datasets já vêm recortados e alinhados, por isso não há deteção nem
alinhamento: cada imagem de data/splits/<dataset>.csv é apenas redimensionada
(112×112 para ArcFace, 224×224 para ViT) e gravada em
data/processed/<dataset>_<size>/<path>.

Saídas:
  - data/splits/<dataset>.csv ganha a coluna processed_<size> com o caminho
    do crop (relativo à raiz do repositório). Fica vazia quando a imagem não
    abre, e essas imagens não entram no treino.
  - data/processed/<dataset>_<size>/removed.csv lista as imagens que não abrem
    (path, split, reason=unreadable).

Os crops já gravados são saltados, por isso o script pode ser interrompido e
retomado.

Correr: python -m scripts.data.preprocess_faces --dataset utkface [--size 112|224]
"""
import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data/processed"
SPLITS = REPO / "data/splits"
JPEG_QUALITY = 95

# dataset -> pasta onde os caminhos do CSV são relativos
DATASETS = {
    "utkface": REPO / "data/UTKFace/UTKFace",
    "affectnet": REPO / "data/AffectNET",
    "btfer7": REPO / "data/BTFER7",
}
REMOVED_FIELDS = ["path", "split", "reason"]


def resize(image, size: int):
    """Redimensiona para size×size (INTER_AREA a reduzir, INTER_CUBIC a ampliar)."""
    shrink = max(image.shape[:2]) > size
    return cv2.resize(image, (size, size),
                      interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)


def preprocess(name: str, size: int) -> None:
    root = DATASETS[name]
    splits_csv = SPLITS / f"{name}.csv"
    if not root.is_dir() or not splits_csv.is_file():
        raise FileNotFoundError(f"{name}: falta {root} ou {splits_csv}")

    df = pd.read_csv(splits_csv)
    out_dir = PROCESSED / f"{name}_{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    column = f"processed_{size}"

    processed: list[str | None] = []
    removed: list[dict] = []
    for rel, split in tqdm(zip(df["path"], df["split"]), total=len(df), desc=f"{name} {size}"):
        out = (out_dir / rel).with_suffix(".jpg")
        if not out.is_file():
            image = cv2.imread(str(root / rel))
            if image is None:
                removed.append({"path": rel, "split": split, "reason": "unreadable"})
                processed.append(None)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out), resize(image, size), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                raise OSError(f"Falha ao gravar {out}")
        processed.append(out.relative_to(REPO).as_posix())

    removed_csv = out_dir / "removed.csv"
    with removed_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, REMOVED_FIELDS)
        writer.writeheader()
        writer.writerows(removed)

    df[column] = processed
    df.to_csv(splits_csv, index=False)

    kept = df[column].notna()
    print(f"{name} {size}: {kept.sum()}/{len(df)} crops ({kept.mean():.1%}), "
          f"{(~kept).sum()} ilegíveis -> {removed_csv.relative_to(REPO)}")
    print(df.assign(kept=kept).groupby("split")["kept"].agg(["sum", "count"]).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--size", type=int, choices=(112, 224), default=112,
                        help="112 para ArcFace, 224 para ViT (DINOv2/CLIP)")
    args = parser.parse_args()
    preprocess(args.dataset, args.size)


if __name__ == "__main__":
    main()
