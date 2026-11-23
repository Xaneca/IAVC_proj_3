"""Demonstração de deteção facial com webcam ou vídeo."""

import argparse

import cv2
import yaml

from app.detector import SCRFDDetector
from app.ui import FPSCounter, draw_detection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", help="Índice da webcam ou caminho para um vídeo")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    source = config["source"] if args.source is None else args.source
    if isinstance(source, str) and source.isdecimal():
        source = int(source)

    detector = SCRFDDetector(**config["detector"])
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise SystemExit(f"Não foi possível abrir a fonte de vídeo: {source}")
        fps = FPSCounter()
        first_frame = True
        while True:
            ok, frame = cap.read()
            if not ok:
                if first_frame or isinstance(source, int):
                    raise SystemExit(f"Não foi possível ler um frame da fonte: {source}")
                break
            first_frame = False
            frame = cv2.flip(frame, 1)
            detections = detector.detect(frame)
            for detection in detections:
                draw_detection(frame, detection)
            cv2.putText(frame, f"FPS {fps.tick():.1f}  Faces {len(detections)}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Detecao facial SCRFD", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
