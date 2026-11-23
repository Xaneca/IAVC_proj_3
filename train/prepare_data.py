"""Alinha os datasets com o MESMO detetor e alinhamento da aplicação.

Para cada imagem listada no CSV de splits do dataset, corre
app.align.detect_and_align (SCRFD + template ArcFace) e grava o crop em
data/aligned/<dataset>/<caminho relativo original>. Os splits não são criados
aqui: vêm de scripts/data/prepare_utkface.py e
scripts/data/prepare_expression_datasets.py.

Os datasets descarregados (UTKFace "Aligned & Cropped", AffectNet Aligned) já
vêm recortados, mas com outro alinhamento. Para a app e o treino verem
exatamente o mesmo tipo de crop, são realinhados. Como a face ocupa a imagem
toda, usa-se retry_pad: sem deteção, repete com uma margem preta à volta.

Cada dataset gera data/aligned/<dataset>/report.csv (path, status, score),
com status ok | no_face | unreadable. As imagens já processadas
são saltadas, por isso o script pode ser interrompido e retomado.

Correr: python -m train.prepare_data [--size 112|224] [--datasets utkface affectnet]
"""
import argparse
import csv
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from app.align import detect_and_align
from app.detector import FaceDetector

ALIGNED = Path("data/aligned")
RETRY_PAD = 0.5
JPEG_QUALITY = 95

# nome → (pasta onde os caminhos do CSV são relativos, CSV, coluna do caminho,
#          já recortado?). Nos recortados a deteção sem margem falha sempre,
#          por isso começa-se logo com margem (metade do tempo).
DATASETS = {
    "utkface": (Path("data/UTKFace/UTKFace"), Path("data/UTKFace/all_labels.csv"), "filename", True),
    "affectnet": (Path("data/AffectNET"), Path("data/splits/affectnet.csv"), "path", True),
    "btfer7": (Path("data/BTFER7"), Path("data/splits/btfer7.csv"), "path", False),
}
REPORT_FIELDS = ["path", "status", "score"]


def read_paths(listing: Path, column: str) -> list[str]:
    with listing.open(encoding="utf-8", newline="") as stream:
        return list(dict.fromkeys(row[column] for row in csv.DictReader(stream)))


def read_report(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["path"]: row for row in csv.DictReader(stream)}


def align_dataset(name: str, detector: FaceDetector, size: int) -> None:
    root, listing, column, cropped = DATASETS[name]
    if not root.is_dir() or not listing.is_file():
        print(f"{name}: missing ({root} or {listing}), skipped")
        return
    dst = ALIGNED / name if size == 112 else ALIGNED / f"{name}_{size}"
    report_path = dst / "report.csv"
    report = read_report(report_path)
    todo = [p for p in read_paths(listing, column) if p not in report]

    dst.mkdir(parents=True, exist_ok=True)
    new_file = not report_path.is_file()
    with report_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, REPORT_FIELDS)
        if new_file:
            writer.writeheader()
        for rel in tqdm(todo, desc=name):
            image = cv2.imread(str(root / rel))
            if image is None:
                row = {"path": rel, "status": "unreadable", "score": ""}
            else:
                face = detect_and_align(image, size, detector=detector,
                                        retry_pad=RETRY_PAD, pad_first=cropped)
                if face is None:
                    row = {"path": rel, "status": "no_face", "score": ""}
                else:
                    out = (dst / rel).with_suffix(".jpg")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(out), face.crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                        raise OSError(f"Failed to write {out}")
                    row = {"path": rel, "status": "ok", "score": f"{face.detection.score:.3f}"}
            writer.writerow(row)
            report[rel] = row
            stream.flush()

    counts: dict[str, int] = {}
    for row in report.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    total = len(report)
    summary = ", ".join(f"{k}={v} ({v / total:.1%})" for k, v in sorted(counts.items()))
    print(f"{name}: {total} images → {summary}  [{report_path}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, choices=(112, 224), default=112,
                        help="112 for ArcFace/ResNet, 224 for ViT (folder <dataset>_224)")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    args = parser.parse_args()

    with open("config.yaml", encoding="utf-8") as stream:
        detector = FaceDetector(**yaml.safe_load(stream)["detector"])
    for name in args.datasets:
        align_dataset(name, detector, args.size)


if __name__ == "__main__":
    main()
