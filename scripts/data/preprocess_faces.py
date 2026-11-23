"""Gera os crops de treino com o MESMO detetor e alinhamento da aplicação.

Para cada imagem de data/splits/<dataset>.csv:
  1. junta uma margem preta à volta (os datasets já vêm recortados e, sem
     margem, o SCRFD não deteta faces que ocupam a imagem toda);
  2. deteta a maior face com o SCRFD (5 landmarks);
  3. alinha com o template ArcFace (app.align.align_face);
  4. grava o crop em data/processed/<dataset>_<size>/<path>.

Saídas:
  - data/splits/<dataset>.csv ganha a coluna processed_<size> com o caminho
    do crop (relativo à raiz do repositório). Fica vazia quando não há face,
    e essas imagens não entram no treino.
  - data/processed/<dataset>_<size>/removed.csv lista as imagens eliminadas
    (path, split, reason), com reason:
      no_face    - o SCRFD não encontra nenhuma face, mesmo com margem;
      pose       - há face, mas o filtro de visibilidade da aplicação rejeita-a
                   (perfil: nariz desviado > 0,25 da distância interocular,
                   caixa < 40 px ou score < 0,5). A app também não a analisaria;
      unreadable - o ficheiro não abre.

Os crops já gravados e as imagens já eliminadas são saltados, por isso o
script pode ser interrompido e retomado.

Correr: python -m scripts.data.preprocess_faces --dataset utkface [--size 112|224]
"""
import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

from app.align import detect_and_align
from app.detector import FaceDetector

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data/processed"
SPLITS = REPO / "data/splits"
PAD = 0.5            # margem preta = metade do lado da imagem em cada borda
JPEG_QUALITY = 95

# dataset -> (pasta onde os caminhos do CSV são relativos, já vem recortado?)
# Nos recortados a deteção sem margem falha sempre, por isso a margem é logo aplicada.
DATASETS = {
    "utkface": (REPO / "data/UTKFace/UTKFace", True),
    "affectnet": (REPO / "data/AffectNET", True),
    "btfer7": (REPO / "data/BTFER7", False),
}
REMOVED_FIELDS = ["path", "split", "reason"]


def read_removed(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["path"]: row for row in csv.DictReader(stream)}


def rejection_reason(image, detector: FaceDetector) -> str:
    """Distingue 'sem face' de 'face rejeitada pelo filtro de visibilidade'."""
    margin = max(1, round(PAD * max(image.shape[:2])))
    padded = cv2.copyMakeBorder(image, margin, margin, margin, margin,
                                cv2.BORDER_CONSTANT, value=0)
    return "pose" if detector.detect(padded) else "no_face"


def preprocess(name: str, size: int, detector: FaceDetector) -> None:
    root, cropped = DATASETS[name]
    splits_csv = SPLITS / f"{name}.csv"
    if not root.is_dir() or not splits_csv.is_file():
        raise FileNotFoundError(f"{name}: falta {root} ou {splits_csv}")

    df = pd.read_csv(splits_csv)
    out_dir = PROCESSED / f"{name}_{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    removed_csv = out_dir / "removed.csv"
    removed = read_removed(removed_csv)
    column = f"processed_{size}"

    processed: list[str | None] = []
    new_file = not removed_csv.is_file()
    with removed_csv.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, REMOVED_FIELDS)
        if new_file:
            writer.writeheader()
        for rel, split in tqdm(zip(df["path"], df["split"]), total=len(df), desc=f"{name} {size}"):
            out = (out_dir / rel).with_suffix(".jpg")
            if out.is_file():
                processed.append(out.relative_to(REPO).as_posix())
                continue
            if rel in removed:
                processed.append(None)
                continue

            image = cv2.imread(str(root / rel))
            face = None if image is None else detect_and_align(
                image, size, detector=detector, retry_pad=PAD, pad_first=cropped)
            if face is None:
                row = {"path": rel, "split": split,
                       "reason": "unreadable" if image is None else rejection_reason(image, detector)}
                writer.writerow(row)
                stream.flush()
                removed[rel] = row
                processed.append(None)
                continue

            out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out), face.crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                raise OSError(f"Falha ao gravar {out}")
            processed.append(out.relative_to(REPO).as_posix())

    df[column] = processed
    df.to_csv(splits_csv, index=False)

    kept = df[column].notna()
    print(f"{name} {size}: {kept.sum()}/{len(df)} crops ({kept.mean():.1%}), "
          f"{(~kept).sum()} eliminadas -> {removed_csv.relative_to(REPO)}")
    print(df.assign(kept=kept).groupby("split")["kept"].agg(["sum", "count"]).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--size", type=int, choices=(112, 224), default=112,
                        help="112 para ArcFace, 224 para ViT (DINOv2/CLIP)")
    args = parser.parse_args()

    with open(REPO / "config.yaml", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)["detector"]
    cfg["model_path"] = str(REPO / cfg["model_path"])
    preprocess(args.dataset, args.size, FaceDetector(**cfg))


if __name__ == "__main__":
    main()
