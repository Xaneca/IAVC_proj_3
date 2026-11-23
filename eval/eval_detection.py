"""Mede a latência por frame do SCRFD-10G e SCRFD-500M em 320 e 640."""

import argparse
import csv
from pathlib import Path
from time import perf_counter_ns

import cv2
import numpy as np
import yaml

from app.detector import SCRFDDetector
from scripts.download_detection_models import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30, help="Medições por configuração")
    parser.add_argument("--output", type=Path, default=Path("results/detection/timings.csv"))
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("--warmup tem de ser não negativo e --iterations tem de ser positivo")
    frames = []
    for path in args.images:
        frame = cv2.imread(str(path))
        if frame is None:
            parser.error(f"Não foi possível ler a imagem: {path}")
        frames.append(frame)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["detector"]
    rows = []
    for name in MODELS.values():
        for size in (320, 640):
            detector = SCRFDDetector(
                str(Path(config["model_path"]).parent / name),
                input_size=size,
                min_score=config["min_score"],
                nms_threshold=config.get("nms_threshold", 0.4),
            )
            for index in range(args.warmup):
                detector.detect(frames[index % len(frames)])
            times = []
            for index in range(args.iterations):
                frame = frames[index % len(frames)]
                start = perf_counter_ns()
                detector.detect(frame)
                times.append((perf_counter_ns() - start) / 1e6)
            row = {
                "model": name, "input_size": size, "provider": detector.provider,
                "frames": args.iterations, "mean_ms": round(float(np.mean(times)), 3),
                "median_ms": round(float(np.median(times)), 3),
                "p95_ms": round(float(np.percentile(times, 95)), 3),
            }
            rows.append(row)
            print(f"{name} {size} {detector.provider}: {row['median_ms']:.3f} ms (mediana)")
            del detector
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
