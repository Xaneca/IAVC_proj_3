"""Ponto de entrada da aplicação: python -m app.main"""
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import yaml

from app.align import align_faces, detect_and_align
from app.backbone import SharedBackbone
from app.detector import Detection, FaceDetector
from app.gallery import Gallery
from app.heads import EXPRESSIONS, GENDERS, AgeGenderHead, ExpressionHead, age_to_range
from app.ui import FPSCounter, draw


@dataclass
class FaceResult:
    detection: Detection
    identity: str | None = None
    labels: list[str] = field(default_factory=list)   # idade, género, expressão


def load_head(head, path, device):
    head.load_state_dict(torch.load(path, map_location=device))
    return head.eval().to(device)


def analyse(frame, cfg, detector, backbone, age_gender, expression, gallery, modules):
    detections = detector.detect(frame)
    results = [FaceResult(d) for d in detections]
    crops = align_faces(frame, detections)
    visible = [(result, face.crop) for result, face in zip(results, crops, strict=True)
               if face.analysable]
    if not visible:
        return results

    out = backbone([crop for _, crop in visible])
    min_conf = cfg["heads"]["min_confidence"]

    with torch.no_grad():
        if modules["age_gender"]:
            ages, g_logits = age_gender(out["layer3"])
            g_probs = g_logits.softmax(1)
        if modules["expression"]:
            e_probs = expression(out["layer3"]).softmax(1)

    for i, (r, _) in enumerate(visible):
        if modules["age_gender"]:
            conf, idx = g_probs[i].max(0)
            gender = GENDERS[idx] if conf >= min_conf else "?"
            r.labels.append(f"~{age_to_range(ages[i].item())} | {gender}")
        if modules["expression"]:
            conf, idx = e_probs[i].max(0)
            label = EXPRESSIONS[idx] if conf >= min_conf else "?"
            r.labels.append(f"{label} ({conf:.2f})")
        if modules["identity"]:
            name, score = gallery.match(out["embedding"][i].cpu().numpy())
            r.identity = f"{name} ({score:.2f})"
    return results


def enrol(cap, detector, backbone, gallery, n_frames):
    """Captura n frames da maior cara visível e regista o template médio."""
    name = input("Nome da pessoa a registar (com consentimento): ").strip()
    if not name:
        return
    embeddings = []
    while len(embeddings) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        face = detect_and_align(frame, 112, detector=detector)
        if face is None:
            continue
        embeddings.append(backbone([face.crop])["embedding"][0].cpu().numpy())
    if embeddings:
        gallery.enrol(name, np.stack(embeddings))
        print(f"Registado: {name} ({len(embeddings)} frames)")


def main(config_path: str = "config.yaml"):
    cfg = yaml.safe_load(open(config_path))
    device = cfg["backbone"]["device"]

    detector = FaceDetector(**cfg["detector"])
    backbone = SharedBackbone(cfg["backbone"]["weights"], device)
    age_gender = load_head(AgeGenderHead(), cfg["heads"]["age_gender"], device)
    expression = load_head(ExpressionHead(), cfg["heads"]["expression"], device)
    gallery = Gallery(cfg["gallery"]["dir"], cfg["gallery"]["threshold"])
    modules = dict(cfg["modules"])

    cap = cv2.VideoCapture(cfg["source"])
    fps = FPSCounter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = analyse(frame, cfg, detector, backbone, age_gender, expression, gallery, modules)
        draw(frame, results, fps.tick(), modules)
        cv2.imshow("Facial Analysis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            enrol(cap, detector, backbone, gallery, cfg["gallery"]["enrol_frames"])
        elif key == ord("d"):
            gallery.delete(input("Nome a apagar: ").strip())
        elif key == ord("c"):
            gallery.clear()
            print("Galeria apagada.")
        elif key in (ord("1"), ord("2"), ord("3")):
            m = ["age_gender", "expression", "identity"][key - ord("1")]
            modules[m] = not modules[m]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
