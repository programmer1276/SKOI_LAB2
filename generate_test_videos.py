#!/usr/bin/env python3
"""Create reproducible videos and ground truth for the laboratory work."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT, FPS, FRAMES = 640, 360, 30, 120
OBJ_W, OBJ_H = 180, 120


def object_texture(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((OBJ_H, OBJ_W, 3), (45, 95, 205), np.uint8)
    for row in range(0, OBJ_H, 15):
        for col in range(0, OBJ_W, 15):
            color = rng.integers(25, 235, 3).tolist()
            cv2.rectangle(image, (col, row), (col + 13, row + 13), color, -1)
    cv2.rectangle(image, (4, 4), (OBJ_W - 5, OBJ_H - 5), (255, 255, 255), 4)
    cv2.rectangle(image, (22, 34), (OBJ_W - 22, 88), (25, 25, 25), -1)
    cv2.putText(image, "ORB", (36, 75), cv2.FONT_HERSHEY_DUPLEX, 1.35, (255, 255, 255), 3, cv2.LINE_AA)
    return image


def background(frame_index: int, rng: np.random.Generator) -> np.ndarray:
    y = np.linspace(35, 95, HEIGHT, dtype=np.uint8)[:, None]
    base = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    base[:, :, 0] = y
    base[:, :, 1] = np.clip(y + 20, 0, 255)
    base[:, :, 2] = np.clip(y + 8, 0, 255)
    for x in range(0, WIDTH, 80):
        cv2.line(base, (x, 0), (x, HEIGHT), (70, 85, 75), 1)
    for yy in range(0, HEIGHT, 60):
        cv2.line(base, (0, yy), (WIDTH, yy), (70, 85, 75), 1)
    noise = rng.normal(0, 2.0, base.shape).astype(np.int16)
    return np.uint8(np.clip(base.astype(np.int16) + noise, 0, 255))


def quad_for(case: str, i: int) -> np.ndarray:
    t = i / (FRAMES - 1)
    if case == "01_translation":
        cx, cy, scale, angle = 120 + 390 * t, 180 + 55 * np.sin(t * 2 * np.pi), 1.0, 0
    elif case == "02_scale_rotation":
        cx, cy = 320 + 115 * np.sin(t * 2 * np.pi), 180
        # The object starts axis-aligned, so its initial ROI tightly matches its borders.
        scale, angle = 0.70 + 0.60 * t, 70 * t
    elif case == "03_occlusion":
        cx, cy, scale, angle = 120 + 390 * t, 170 + 35 * np.sin(t * 4 * np.pi), 0.92, 8 * np.sin(t * 2 * np.pi)
    else:
        cx = 320 + 170 * np.sin(t * 2 * np.pi)
        cy = 180 + 65 * np.sin(t * 4 * np.pi)
        scale, angle = 0.8 + 0.28 * np.sin(t * 2 * np.pi + .5), 18 * np.sin(t * 3 * np.pi)
    rectangle = np.array([[-OBJ_W / 2, -OBJ_H / 2], [OBJ_W / 2, -OBJ_H / 2],
                          [OBJ_W / 2, OBJ_H / 2], [-OBJ_W / 2, OBJ_H / 2]], np.float32) * scale
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]], np.float32)
    quad = rectangle @ rotation.T + [cx, cy]
    if case == "04_personal_demo":
        quad += np.array([[-8, 3], [6, -5], [10, 5], [-5, 7]], np.float32) * np.sin(t * 4 * np.pi)
    return quad.astype(np.float32)


def render_case(case: str, root: Path) -> tuple[int, int, int, int]:
    root.mkdir(parents=True, exist_ok=True)
    video_path, gt_path = root / f"{case}.mp4", root / f"{case}_gt.csv"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {video_path}")
    texture = object_texture()
    source = np.array([[0, 0], [OBJ_W - 1, 0], [OBJ_W - 1, OBJ_H - 1], [0, OBJ_H - 1]], np.float32)
    rng = np.random.default_rng(100 + sum(map(ord, case)))
    rows, first_box = [], None
    for i in range(FRAMES):
        frame = background(i, rng)
        quad = quad_for(case, i)
        transform = cv2.getPerspectiveTransform(source, quad)
        warped = cv2.warpPerspective(texture, transform, (WIDTH, HEIGHT))
        alpha = cv2.warpPerspective(np.full((OBJ_H, OBJ_W), 255, np.uint8), transform, (WIDTH, HEIGHT))
        frame[alpha > 0] = warped[alpha > 0]
        if case == "03_occlusion" and 48 <= i <= 76:
            cover_x = int(250 + (i - 48) * 5)
            cv2.rectangle(frame, (cover_x, 85), (cover_x + 115, 290), (35, 35, 35), -1)
            cv2.putText(frame, "OCCLUDER", (cover_x + 7, 190), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, (230, 230, 230), 1, cv2.LINE_AA)
        if case == "04_personal_demo":
            gain = 0.72 + 0.34 * (0.5 + 0.5 * np.sin(i * .08))
            frame = cv2.convertScaleAbs(frame, alpha=gain, beta=7)
            frame = cv2.GaussianBlur(frame, (3, 3), 0.45)
        x, y, w, h = cv2.boundingRect(quad)
        rows.append((i, x, y, w, h, *quad.reshape(-1).tolist()))
        first_box = first_box or (x, y, w, h)
        writer.write(frame)
    writer.release()
    with gt_path.open("w", newline="", encoding="utf-8") as stream:
        out = csv.writer(stream)
        out.writerow(("frame", "x", "y", "w", "h",
                      "p0x", "p0y", "p1x", "p1y", "p2x", "p2y", "p3x", "p3y"))
        out.writerows(rows)
    return first_box


def main() -> None:
    root = Path("data")
    cases = ("01_translation", "02_scale_rotation", "03_occlusion", "04_personal_demo")
    for case in cases:
        box = render_case(case, root)
        print(f"{case}: bbox={','.join(map(str, box))}")
    print("Важно: 04_personal_demo — техническая имитация. Для сдачи замените её своей записью.")


if __name__ == "__main__":
    main()
