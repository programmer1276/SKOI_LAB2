#!/usr/bin/env python3
"""Record the student's own video with a textured object from a webcam."""

import argparse
from pathlib import Path
import time

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Запись собственного видео для ЛР2")
    parser.add_argument("--output", type=Path, default=Path("data/my_video.mp4"))
    parser.add_argument("--duration", type=float, default=10.0, help="длительность записи в секундах")
    parser.add_argument("--camera", type=int, default=0, help="номер камеры")
    args = parser.parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("камера не открылась; проверьте разрешение macOS на доступ к камере")
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    if not 1 <= fps <= 120:
        fps = 30.0
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    print("Поместите крупный текстурный предмет в центр кадра.")
    print("Пробел — начать запись, Esc — отменить.")
    while True:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("не удалось получить кадр с камеры")
        preview = frame.copy()
        cv2.putText(preview, "PLACE OBJECT - SPACE TO RECORD", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow("Own video for Lab 2", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            break
        if key == 27:
            camera.release()
            writer.release()
            cv2.destroyAllWindows()
            print("Запись отменена.")
            return

    started = time.monotonic()
    print(f"Идёт запись {args.duration:g} секунд. Плавно двигайте и поворачивайте предмет.")
    while time.monotonic() - started < args.duration:
        ok, frame = camera.read()
        if not ok:
            break
        writer.write(frame)
        remaining = max(0.0, args.duration - (time.monotonic() - started))
        cv2.putText(frame, f"REC {remaining:04.1f}s - Q to stop", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Own video for Lab 2", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    writer.release()
    camera.release()
    cv2.destroyAllWindows()
    print(f"Сохранено: {output}")


if __name__ == "__main__":
    main()
