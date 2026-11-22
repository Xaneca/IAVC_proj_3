"""Gera uma grelha de alinhamento SCRFD com variações controladas da imagem."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.align import detect_and_align


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Imagem ilegível: {args.image}")

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rows = []
    for angle in (-20, 0, 20):
        matrix = cv2.getRotationMatrix2D(center, angle, 0.9)
        frame = cv2.warpAffine(image, matrix, (width, height))
        face112 = detect_and_align(frame, 112)
        face224 = detect_and_align(frame, 224)
        if face112 is None or face224 is None:
            parser.error(f"Sem face analisável para a inclinação de {angle:+d} graus")
        preview = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.putText(preview, f"{angle:+d} graus", (8, 213), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        # Ampliação para comparar os dois tamanhos na mesma escala visual.
        enlarged = cv2.resize(face112.crop, (224, 224), interpolation=cv2.INTER_NEAREST)
        rows.append(np.hstack((preview, enlarged, face224.crop)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), np.vstack(rows)):
        parser.error(f"Erro ao gravar {args.output}")


if __name__ == "__main__":
    main()
