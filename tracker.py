#!/usr/bin/env python3
"""Object tracking by ORB keypoints, descriptor matching and RANSAC homography."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameResult:
    frame: int
    found: bool
    matches: int
    inliers: int
    iou: float | None


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(v.strip()) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("рамка задаётся как x,y,w,h") from exc
    if len(result) != 4 or result[2] <= 0 or result[3] <= 0:
        raise argparse.ArgumentTypeError("нужны четыре числа x,y,w,h; w и h > 0")
    return result


def load_ground_truth(path: Path | None) -> dict[int, tuple[float, ...]]:
    if path is None:
        return {}
    result = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            polygon_keys = tuple(f"p{point}{axis}" for point in range(4) for axis in ("x", "y"))
            keys = polygon_keys if all(row.get(key) not in (None, "") for key in polygon_keys) else ("x", "y", "w", "h")
            result[int(row["frame"])] = tuple(float(row[k]) for k in keys)
    return result


def polygon_iou(polygon: np.ndarray, truth_data: tuple[float, ...]) -> float:
    if len(truth_data) == 8:
        truth = np.array(truth_data, np.float32).reshape(4, 2)
    else:
        x, y, w, h = truth_data
        truth = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
    predicted = cv2.convexHull(polygon.astype(np.float32))
    truth = cv2.convexHull(truth)
    intersection, _ = cv2.intersectConvexConvex(predicted, truth)
    union = abs(cv2.contourArea(predicted)) + abs(cv2.contourArea(truth)) - intersection
    return float(intersection / union) if union > 0 else 0.0


def valid_polygon(polygon: np.ndarray, width: int, height: int, initial_area: float) -> bool:
    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon.astype(np.float32)):
        return False
    area = abs(cv2.contourArea(polygon.astype(np.float32)))
    margin = max(width, height) * 0.25
    inside = ((polygon[:, 0] >= -margin) & (polygon[:, 0] < width + margin)
              & (polygon[:, 1] >= -margin) & (polygon[:, 1] < height + margin)).all()
    return inside and initial_area * 0.15 < area < initial_area * 6.0


def draw_overlay(frame: np.ndarray, polygon: np.ndarray, points: np.ndarray,
                 found: bool, matches: int, inliers: int, frame_index: int) -> None:
    color = (40, 220, 40) if found else (40, 40, 230)
    cv2.polylines(frame, [np.int32(polygon)], True, color, 3, cv2.LINE_AA)
    for px, py in points:
        cv2.circle(frame, (round(float(px)), round(float(py))), 3, (0, 220, 255), -1, cv2.LINE_AA)
    status = "TRACKED" if found else "LOST"
    cv2.rectangle(frame, (8, 8), (390, 72), (20, 20, 20), -1)
    cv2.putText(frame, f"Object: {status}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.68, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"frame={frame_index} matches={matches} inliers={inliers}",
                (18, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)


def track_video(input_path: Path, output_path: Path, bbox: tuple[int, int, int, int] | None,
                gt_path: Path | None, metrics_path: Path | None, show: bool,
                max_features: int, ratio: float, min_inliers: int) -> dict:
    capture = cv2.VideoCapture(str(input_path))
    ok, first = capture.read()
    if not ok:
        raise RuntimeError(f"не удалось прочитать видео: {input_path}")

    height, width = first.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if bbox is None:
        if not show:
            raise ValueError("в режиме без окна необходимо передать --bbox x,y,w,h")
        bbox = tuple(map(int, cv2.selectROI("Select object, ENTER to confirm", first, False, False)))
        cv2.destroyWindow("Select object, ENTER to confirm")
    x, y, w, h = bbox
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"рамка {bbox} выходит за границы кадра {width}x{height}")

    gray_first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_first)
    mask[y:y + h, x:x + w] = 255
    orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=15, patchSize=31, fastThreshold=7)
    reference_kp, reference_desc = orb.detectAndCompute(gray_first, mask)
    if reference_desc is None or len(reference_kp) < min_inliers:
        raise RuntimeError("на объекте слишком мало ключевых точек; выберите более текстурный объект")

    reference_points = np.float32([kp.pt for kp in reference_kp])
    initial_polygon = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    initial_area = float(w * h)
    current_polygon = initial_polygon.copy()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    ground_truth = load_ground_truth(gt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"не удалось открыть файл результата: {output_path}")

    rows: list[FrameResult] = []
    started = time.perf_counter()
    frame_index = 0
    frame = first
    try:
        while True:
            if frame_index == 0:
                found, matches_count, inliers_count = True, len(reference_kp), len(reference_kp)
                inlier_points = reference_points
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_kp, current_desc = orb.detectAndCompute(gray, None)
                good = []
                if current_desc is not None:
                    pairs = matcher.knnMatch(reference_desc, current_desc, k=2)
                    good = [a for pair in pairs if len(pair) == 2
                            for a, b in [pair] if a.distance < ratio * b.distance]
                matches_count = len(good)
                found, inliers_count = False, 0
                inlier_points = np.empty((0, 2), np.float32)
                if matches_count >= 4:
                    src = np.float32([reference_kp[m.queryIdx].pt for m in good])
                    dst = np.float32([current_kp[m.trainIdx].pt for m in good])
                    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                    if homography is not None and inlier_mask is not None:
                        candidate = cv2.perspectiveTransform(initial_polygon.reshape(1, -1, 2), homography)[0]
                        inliers_count = int(inlier_mask.sum())
                        if inliers_count >= min_inliers and valid_polygon(candidate, width, height, initial_area):
                            current_polygon = candidate
                            found = True
                            inlier_points = dst[inlier_mask.ravel().astype(bool)]

            iou = polygon_iou(current_polygon, ground_truth[frame_index]) if frame_index in ground_truth else None
            rows.append(FrameResult(frame_index, found, matches_count, inliers_count, iou))
            draw_overlay(frame, current_polygon, inlier_points, found, matches_count, inliers_count, frame_index)
            writer.write(frame)
            if show:
                cv2.imshow("Keypoint tracker (Q to exit)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    ious = [r.iou for r in rows if r.iou is not None]
    successful = [r for r in rows if r.found]
    summary = {
        "input": str(input_path), "output": str(output_path), "bbox": list(bbox),
        "reference_keypoints": len(reference_kp), "frames": len(rows),
        "successful_frames": len(successful),
        "tracking_success_percent": round(100 * len(successful) / len(rows), 2),
        "mean_matches": round(float(np.mean([r.matches for r in rows])), 2),
        "mean_inliers": round(float(np.mean([r.inliers for r in rows])), 2),
        "processing_fps": round(len(rows) / elapsed, 2),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "precision_iou_0_5_percent": round(100 * sum(v >= .5 for v in ious) / len(ious), 2) if ious else None,
        "parameters": {"max_features": max_features, "ratio": ratio, "min_inliers": min_inliers},
        "per_frame": [asdict(row) for row in rows],
    }
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Трекинг объекта по ключевым точкам ORB")
    parser.add_argument("input", type=Path, help="входное видео")
    parser.add_argument("--output", type=Path, default=Path("outputs/tracked.mp4"))
    parser.add_argument("--bbox", type=parse_bbox, help="начальная рамка x,y,w,h")
    parser.add_argument("--ground-truth", type=Path, help="CSV с колонками frame,x,y,w,h")
    parser.add_argument("--metrics", type=Path, help="куда сохранить JSON с метриками")
    parser.add_argument("--show", action="store_true", help="показывать окно; без --bbox предложит выбрать ROI")
    parser.add_argument("--max-features", type=int, default=1800)
    parser.add_argument("--ratio", type=float, default=0.78, help="порог ratio test Лоу")
    parser.add_argument("--min-inliers", type=int, default=7)
    args = parser.parse_args()
    result = track_video(args.input, args.output, args.bbox, args.ground_truth,
                         args.metrics, args.show, args.max_features, args.ratio, args.min_inliers)
    print(json.dumps({k: v for k, v in result.items() if k != "per_frame"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
