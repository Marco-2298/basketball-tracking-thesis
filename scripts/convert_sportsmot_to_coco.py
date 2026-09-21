from pathlib import Path
import json
import os


SPORTSMOT_ROOT = Path("datasets/sportsmot")
OUTPUT_ROOT = Path("datasets/sportsmot_coco")

CATEGORY = {
    "id": 1,
    "name": "basketball_player",
    "supercategory": "person",
}


def convert_split(source_split: str, target_split: str):
    source_dir = SPORTSMOT_ROOT / source_split
    target_dir = OUTPUT_ROOT / target_split

    target_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [CATEGORY],
    }

    image_id = 1
    annotation_id = 1

    basketball_file = SPORTSMOT_ROOT / "splits_txt" / "basketball.txt"

    basketball_sequences = {
        line.strip()
        for line in basketball_file.read_text().splitlines()
        if line.strip()
    }

    sequences = sorted(
        p
        for p in source_dir.iterdir()
        if p.is_dir() and p.name in basketball_sequences
    )

    print(f"\nConversione {source_split} -> {target_split}")
    print(f"Sequenze: {len(sequences)}")

    for sequence_dir in sequences:
        sequence_name = sequence_dir.name

        img_dir = sequence_dir / "img1"
        gt_path = sequence_dir / "gt" / "gt.txt"

        if not img_dir.exists():
            print(f"SKIP: manca img1 in {sequence_name}")
            continue

        if not gt_path.exists():
            print(f"SKIP: manca gt.txt in {sequence_name}")
            continue

        # Leggiamo prima tutte le annotazioni,
        # indicizzate per frame.
        annotations_by_frame = {}

        with gt_path.open("r") as file:
            for line in file:
                values = [
                    value.strip()
                    for value in line.split(",")
                ]

                frame_id = int(values[0])
                track_id = int(values[1])

                x = float(values[2])
                y = float(values[3])
                width = float(values[4])
                height = float(values[5])

                annotations_by_frame.setdefault(
                    frame_id, []
                ).append(
                    {
                        "track_id": track_id,
                        "bbox": [
                            x,
                            y,
                            width,
                            height,
                        ],
                    }
                )

        frame_paths = sorted(img_dir.glob("*.jpg"))

        for frame_path in frame_paths:
            frame_id = int(frame_path.stem)

            # Nome unico: evita collisioni tra 000001.jpg
            # appartenenti a sequenze differenti.
            new_filename = (
                f"{sequence_name}_{frame_path.name}"
            )

            target_image_path = (
                target_dir / new_filename
            )

            # Hard link:
            # non duplica fisicamente l'immagine sul disco.
            if not target_image_path.exists():
                os.link(
                    frame_path.resolve(),
                    target_image_path,
                )

            # Ricaviamo dimensioni dall'header JPEG con OpenCV
            # solo quando necessario.
            import cv2

            image = cv2.imread(str(frame_path))

            if image is None:
                print(
                    f"Impossibile leggere: {frame_path}"
                )
                continue

            height, width = image.shape[:2]

            current_image_id = image_id

            coco["images"].append(
                {
                    "id": current_image_id,
                    "file_name": new_filename,
                    "width": width,
                    "height": height,
                }
            )

            for ann in annotations_by_frame.get(
                frame_id, []
            ):
                x, y, w, h = ann["bbox"]

                if w <= 0 or h <= 0:
                    continue

                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,

                        # Campo extra utile a noi.
                        # COCO lo ignora.
                        "track_id": ann["track_id"],
                    }
                )

                annotation_id += 1

            image_id += 1

    annotation_path = (
        target_dir / "_annotations.coco.json"
    )

    with annotation_path.open("w") as file:
        json.dump(coco, file)

    print(f"Immagini: {len(coco['images'])}")
    print(
        f"Bounding box: "
        f"{len(coco['annotations'])}"
    )
    print(f"COCO JSON: {annotation_path}")


def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    convert_split(
        source_split="train",
        target_split="train",
    )

    convert_split(
        source_split="val",
        target_split="valid",
    )


if __name__ == "__main__":
    main()