"""Overlay no vídeo: caixas, identidade, atributos e estatísticas."""
import time

import cv2

GREEN, ORANGE, WHITE = (0, 200, 0), (0, 165, 255), (255, 255, 255)


def draw(frame, results, fps: float, modules: dict) -> None:
    for r in results:
        x1, y1, x2, y2 = r.detection.bbox.astype(int)
        known = r.identity not in (None, "Unknown")
        color = GREEN if known else ORANGE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        lines = [r.identity or ""] + r.labels
        for i, text in enumerate(l for l in lines if l):
            y = y1 - 8 - 18 * i
            cv2.putText(frame, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    status = "  ".join(f"[{k[0]}]{'on' if v else 'off'}" for k, v in modules.items())
    cv2.putText(frame, f"FPS {fps:.1f}  {status}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)


class FPSCounter:
    def __init__(self, alpha: float = 0.9):
        self.alpha, self.fps, self.last = alpha, 0.0, time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        self.fps = self.alpha * self.fps + (1 - self.alpha) / max(now - self.last, 1e-6)
        self.last = now
        return self.fps
