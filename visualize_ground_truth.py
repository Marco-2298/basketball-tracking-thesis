from pathlib import Path

import cv2


SEQUENCE = Path(
    "datasets/sportsmot/val/v_G-vNjfx1GGc_c601"
)

FRAME_ID = 1


# --------------------------------------------------
# Load frame
# --------------------------------------------------

image_path = SEQUENCE / "img1" / f"{FRAME_ID:06d}.jpg"

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Immagine non trovata: {image_path}")


# --------------------------------------------------
# Load ground truth
# --------------------------------------------------

gt_path = SEQUENCE / "gt" / "gt.txt"

with gt_path.open("r") as file:

    for line in file:

        values = [
            value.strip()
            for value in line.split(",")
        ]

        frame_id = int(values[0])

        if frame_id != FRAME_ID:
            continue

        track_id = int(values[1])

        x = int(float(values[2]))
        y = int(float(values[3]))
        width = int(float(values[4]))
        height = int(float(values[5]))

        x2 = x + width
        y2 = y + height

        cv2.rectangle(
            image,
            (x, y),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"ID {track_id}",
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


# --------------------------------------------------
# Save
# --------------------------------------------------

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "sportsmot_ground_truth.jpg"

cv2.imwrite(
    str(output_path),
    image,
)

print(f"Salvato: {output_path}")