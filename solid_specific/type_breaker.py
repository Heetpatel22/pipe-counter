import os
from functools import lru_cache

import cv2
import numpy as np
from ultralytics import YOLO


@lru_cache(maxsize=2)
def _load_model(trained_weights_path):
    return YOLO(trained_weights_path)


def _ring_boxes(image_rgb, boxes, color):
    annotated = image_rgb.copy()
    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)
        radius = int(max(xmax - xmin, ymax - ymin) / 2)
        cv2.circle(annotated, (cx, cy), radius, color, 3)
    return annotated


def _has_hollow_center(image_gray, boxes):
    hollow_count = 0

    for box in boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        crop = image_gray[ymin:ymax, xmin:xmax]
        if crop.size == 0:
            continue

        h, w = crop.shape
        center_mask = np.zeros_like(crop)
        cv2.circle(center_mask, (w // 2, h // 2), int(min(w, h) * 0.25), 255, -1)

        center_brightness = cv2.mean(crop, mask=center_mask)[0]
        outer_brightness = cv2.mean(crop, mask=cv2.bitwise_not(center_mask))[0]

        if center_brightness < (outer_brightness * 0.65) and center_brightness < 100:
            hollow_count += 1

    return hollow_count


def analyze_inventory(image_path, trained_weights_path, conf=0.60, iou=0.45):
    model = _load_model(trained_weights_path)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return {
            "passed": False,
            "status": "Cannot read image",
            "category": "Solid Steel Rod Stack",
            "count": 0,
            "total": 0,
            "hollow_count": 0,
            "hollow_ratio": 0.0,
            "avg_confidence": 0.0,
            "reason": f"Cannot read the image file at {image_path}.",
            "annotated_image": None,
        }

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    results = model(image_path, conf=conf, iou=iou, verbose=False)
    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
    confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []

    total = len(boxes)
    if total == 0:
        return {
            "passed": False,
            "status": "No targets recorded",
            "category": "Solid Steel Rod Stack",
            "count": 0,
            "total": 0,
            "hollow_count": 0,
            "hollow_ratio": 0.0,
            "avg_confidence": 0.0,
            "reason": "No rod targets were detected above the confidence threshold.",
            "annotated_image": image_rgb,
        }

    hollow_count = _has_hollow_center(image_gray, boxes)
    hollow_ratio = hollow_count / total
    avg_confidence = float(np.mean(confidences) * 100) if len(confidences) else 0.0

    if hollow_ratio >= 0.10:
        return {
            "passed": False,
            "status": "Rejected",
            "category": "Hollow Pipe Detected",
            "count": 0,
            "total": total,
            "hollow_count": hollow_count,
            "hollow_ratio": hollow_ratio,
            "avg_confidence": avg_confidence,
            "reason": f"Hollow pipe geometry detected ({hollow_ratio * 100:.1f}% targets flagged as hollow). Limit 10.0%.",
            "annotated_image": image_rgb,
        }

    return {
        "passed": True,
        "status": "Passed",
        "category": "Solid Steel Rod Stack",
        "count": total,
        "total": total,
        "hollow_count": 0,
        "hollow_ratio": 0.0,
        "avg_confidence": avg_confidence,
        "reason": "Batch verified as solid rod inventory.",
        "annotated_image": _ring_boxes(image_rgb, boxes, (26, 54, 140)),
    }


def run_production_inventory_with_type_breaker(image_path, trained_weights_path):
    report = analyze_inventory(image_path, trained_weights_path)

    print(f"\nBatch Scan -> {os.path.basename(image_path)}")
    print(f"Status   : {report['status']}")
    print(f"Category : {report['category']}")
    print(f"Count    : {report['count']} Units")
    print(f"Reason   : {report['reason']}\n")

    return report["passed"]
