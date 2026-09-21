from pathlib import Path

import cv2
import numpy as np

from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES


SEQUENCE_NAME = "v_00HRwkvvjtQ_c001"
FRAME_ID = 1

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.4

SEQUENCE_DIR = Path("datasets/sportsmot/val") / SEQUENCE_NAME

IMAGE_PATH = (
    SEQUENCE_DIR
    / "img1"
    / f"{FRAME_ID:06d}.jpg"
)

GT_PATH = SEQUENCE_DIR / "gt" / "gt.txt"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / f"{SEQUENCE_NAME}_comparison_{FRAME_ID:06d}.jpg"
)


def load_gt_boxes(gt_path, frame_id):
    boxes = []

    with gt_path.open("r") as file:
        for line in file:
            values = [value.strip() for value in line.split(",")]

            current_frame = int(values[0])

            if current_frame != frame_id:
                continue

            track_id = int(values[1])

            x = float(values[2])
            y = float(values[3])
            w = float(values[4])
            h = float(values[5])

            boxes.append(
                {
                    "track_id": track_id,
                    "box": np.array(
                        [x, y, x + w, y + h],
                        dtype=np.float32,
                    ),
                }
            )

    return boxes


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area_a = (
        (box_a[2] - box_a[0])
        * (box_a[3] - box_a[1])
    )

    area_b = (
        (box_b[2] - box_b[0])
        * (box_b[3] - box_b[1])
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def match_boxes(gt_boxes, prediction_boxes):
    candidates = []

    for gt_index, gt in enumerate(gt_boxes):
        for pred_index, pred in enumerate(prediction_boxes):

            iou = compute_iou(
                gt["box"],
                pred,
            )

            if iou >= IOU_THRESHOLD:
                candidates.append(
                    (
                        iou,
                        gt_index,
                        pred_index,
                    )
                )

    # Partiamo dagli accoppiamenti con IoU maggiore.
    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gt_index, pred_index in candidates:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_pred:
            continue

        matched_gt.add(gt_index)
        matched_pred.add(pred_index)

        matches.append(
            (
                gt_index,
                pred_index,
                iou,
            )
        )

    return matches, matched_gt, matched_pred


def draw_box(image, box, color, label):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    cv2.putText(
        image,
        label,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
    )


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Immagine non trovata: {IMAGE_PATH}"
        )

    # -------------------------------------------------
    # Ground truth
    # -------------------------------------------------

    gt_boxes = load_gt_boxes(
        GT_PATH,
        FRAME_ID,
    )

    print(f"Ground truth: {len(gt_boxes)} giocatori")

    # -------------------------------------------------
    # RF-DETR
    # -------------------------------------------------

    print("Caricamento RF-DETR Nano...")

    model = RFDETRNano()

    detections = model.predict(
        str(IMAGE_PATH),
        threshold=CONFIDENCE_THRESHOLD,
    )

    person_class_id = next(
        class_id
        for class_id, class_name in COCO_CLASSES.items()
        if class_name == "person"
    )

    detections = detections[
        detections.class_id == person_class_id
    ]

    prediction_boxes = [
        np.array(box, dtype=np.float32)
        for box in detections.xyxy
    ]

    print(
        f"RF-DETR person detection: "
        f"{len(prediction_boxes)}"
    )

    # -------------------------------------------------
    # Matching GT ↔ prediction
    # -------------------------------------------------

    matches, matched_gt, matched_pred = match_boxes(
        gt_boxes,
        prediction_boxes,
    )

    true_positives = len(matches)

    false_negatives = (
        len(gt_boxes)
        - true_positives
    )

    false_positives = (
        len(prediction_boxes)
        - true_positives
    )

    precision = (
        true_positives
        / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )

    recall = (
        true_positives
        / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )

    matched_ious = [
        match[2]
        for match in matches
    ]

    mean_iou = (
        float(np.mean(matched_ious))
        if matched_ious
        else 0
    )

    print()
    print("===== RISULTATI =====")
    print(f"TP: {true_positives}")
    print(f"FP: {false_positives}")
    print(f"FN: {false_negatives}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"Mean IoU matched: {mean_iou:.3f}")

    # -------------------------------------------------
    # Disegno
    # -------------------------------------------------

    annotated = image.copy()

    # Ground truth = verde
    for gt in gt_boxes:
        draw_box(
            annotated,
            gt["box"],
            (0, 255, 0),
            f"GT ID {gt['track_id']}",
        )

    # Prediction = rosso
    for index, pred_box in enumerate(prediction_boxes):

        if index in matched_pred:
            label = "RF-DETR match"
        else:
            label = "RF-DETR FP"

        draw_box(
            annotated,
            pred_box,
            (0, 0, 255),
            label,
        )

    cv2.imwrite(
        str(OUTPUT_PATH),
        annotated,
    )

    print()
    print(f"Output salvato: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()