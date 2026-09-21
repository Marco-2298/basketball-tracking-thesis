from pathlib import Path
import json
import random
import cv2


DATASET_DIR = Path("datasets/sportsmot_coco/train")
ANNOTATION_PATH = DATASET_DIR / "_annotations.coco.json"

OUTPUT_DIR = Path("outputs/coco_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_IMAGES = 10


def main():
    with ANNOTATION_PATH.open("r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    annotations_by_image = {}

    for ann in annotations:
        annotations_by_image.setdefault(
            ann["image_id"], []
        ).append(ann)

    selected_images = random.sample(
        images,
        min(NUM_IMAGES, len(images)),
    )

    for image_info in selected_images:
        image_path = (
            DATASET_DIR
            / image_info["file_name"]
        )

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Immagine non trovata: {image_path}")
            continue

        image_id = image_info["id"]

        for ann in annotations_by_image.get(
            image_id, []
        ):
            x, y, w, h = ann["bbox"]

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

        output_path = (
            OUTPUT_DIR
            / image_info["file_name"]
        )

        cv2.imwrite(
            str(output_path),
            image,
        )

        print(f"Salvato: {output_path}")


if __name__ == "__main__":
    main()
    