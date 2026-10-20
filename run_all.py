#!/usr/bin/env python3
"""Generate all test data, run the tracker, and print a compact results table."""

from pathlib import Path

from generate_test_videos import render_case
from tracker import track_video


CASES = ("01_translation", "02_scale_rotation", "03_occlusion", "04_personal_demo")


def main() -> None:
    data, outputs = Path("data"), Path("outputs")
    outputs.mkdir(exist_ok=True)
    print("case                  success     mean IoU    IoU>=0.5   FPS")
    print("-" * 68)
    for case in CASES:
        bbox = render_case(case, data)
        result = track_video(data / f"{case}.mp4", outputs / f"{case}_tracked.mp4", bbox,
                             data / f"{case}_gt.csv", outputs / f"{case}_metrics.json",
                             False, 1800, .78, 7)
        print(f"{case:21} {result['tracking_success_percent']:7.2f}%"
              f" {result['mean_iou']:12.4f} {result['precision_iou_0_5_percent']:9.2f}%"
              f" {result['processing_fps']:7.2f}")


if __name__ == "__main__":
    main()

