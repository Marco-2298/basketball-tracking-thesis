from pathlib import Path

import cv2
import supervision as sv

from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES


SEQUENCE_NAME = "v_00HRwkvvjtQ_c001"
FRAME_ID = 1

IMAGE_PATH = (
    Path("datasets/sportsmot/val")
    / SEQUENCE_NAME
    / "img1"
    / f"{FRAME_ID:06d}.jpg"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / f"{SEQUENCE_NAME}_rfdetr_frame_{FRAME_ID:06d}.jpg"


def main():
    print("Caricamento RF-DETR Nano...")
    model = RFDETRNano()

    print(f"Immagine: {IMAGE_PATH}")

    detections = model.predict(
        str(IMAGE_PATH),
        threshold=0.4,
    )

    # RF-DETR pretrained usa le classi COCO.
    # Per ora ci interessa soltanto la classe "person".
    person_class_id = next(
        class_id
        for class_id, class_name in COCO_CLASSES.items()
        if class_name == "person"
    )

    person_mask = detections.class_id == person_class_id
    detections = detections[person_mask]

    print(f"Persone rilevate: {len(detections)}")

    labels = [
        f"person {confidence:.2f}"
        for confidence in detections.confidence
    ]

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(f"Immagine non trovata: {IMAGE_PATH}")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = box_annotator.annotate(
        scene=image.copy(),
        detections=detections,
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels,
    )

    cv2.imwrite(str(OUTPUT_PATH), annotated)

    print(f"Output salvato in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()