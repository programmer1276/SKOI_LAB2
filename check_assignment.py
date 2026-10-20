#!/usr/bin/env python3
"""Check the parts of the Lab 2 assignment that can be verified automatically."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def exists(relative: str) -> bool:
    return (ROOT / relative).is_file()


def main() -> None:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Исходный код трекера", exists("tracker.py"), "tracker.py"))
    checks.append(("Три варианта тестового видео",
                   all(exists(f"data/{i:02d}_{name}.mp4") for i, name in
                       ((1, "translation"), (2, "scale_rotation"), (3, "occlusion"))),
                   "data/01..03_*.mp4"))
    checks.append(("Результаты трёх тестов",
                   all(exists(f"outputs/{i:02d}_{name}_tracked.mp4") for i, name in
                       ((1, "translation"), (2, "scale_rotation"), (3, "occlusion"))),
                   "outputs/01..03_*_tracked.mp4"))
    required_sections = ("Теоретическая база", "Описание разработанной системы",
                         "Результаты работы и тестирования системы", "Выводы по работе",
                         "Использованные источники")
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if exists("README.md") else ""
    checks.append(("README: пять обязательных частей",
                   all(section in readme for section in required_sections), "README.md"))
    checks.append(("Собственное видео студента", exists("data/my_video.mp4"), "data/my_video.mp4"))
    checks.append(("Результат на собственном видео", exists("outputs/my_video_tracked.mp4"),
                   "outputs/my_video_tracked.mp4"))
    try:
        remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT,
                                capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        remote = ""
    checks.append(("GitHub remote origin", "github.com" in remote.lower(), remote or "не настроен"))

    print("Проверка соответствия заданию ЛР2\n")
    for title, passed, detail in checks:
        print(f"[{'OK' if passed else 'НУЖНО'}] {title}: {detail}")

    metrics = []
    for path in sorted((ROOT / "outputs").glob("0[1-3]_*_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics.append((path.stem, data.get("mean_iou"), data.get("tracking_success_percent")))
    if metrics:
        print("\nФактические результаты тестов:")
        for name, iou, success in metrics:
            print(f"- {name}: mean IoU={iou}, успешные кадры={success}%")

    failed = sum(not passed for _, passed, _ in checks)
    print(f"\nИтог: {len(checks) - failed}/{len(checks)} проверок выполнено.")
    if failed:
        print("Пункты с пометкой НУЖНО требуют завершения.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
