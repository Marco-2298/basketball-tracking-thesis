from pathlib import Path
import cv2

SEQUENCE_NAME = "v_00HRwkvvjtQ_c001"

SEQUENCE = Path("datasets/sportsmot/val") / SEQUENCE_NAME
IMG_DIR = SEQUENCE / "img1"
GT_PATH = SEQUENCE / "gt" / "gt.txt"
SEQINFO_PATH = SEQUENCE / "seqinfo.ini"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / f"{SEQUENCE_NAME}_gt.mp4"


def read_seqinfo(path):
    info = {}

    with path.open("r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("["):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                info[key.strip()] = value.strip()

    return info


def load_ground_truth(path):
    annotations = {}

    with path.open("r") as file:
        for line in file:
            values = [value.strip() for value in line.split(",")]

            frame_id = int(values[0])
            track_id = int(values[1])

            x = int(float(values[2]))
            y = int(float(values[3]))
            width = int(float(values[4]))
            height = int(float(values[5]))

            annotations.setdefault(frame_id, []).append(
                {
                    "track_id": track_id,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )

    return annotations


def main():
    seqinfo = read_seqinfo(SEQINFO_PATH)

    fps = int(seqinfo["frameRate"])
    width = int(seqinfo["imWidth"])
    height = int(seqinfo["imHeight"])
    seq_length = int(seqinfo["seqLength"])
    extension = seqinfo["imExt"]

    annotations = load_ground_truth(GT_PATH)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height),
    )

    for frame_id in range(1, seq_length + 1):

        image_path = IMG_DIR / f"{frame_id:06d}{extension}"
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Frame non trovato: {image_path}")
            continue

        for annotation in annotations.get(frame_id, []):

            track_id = annotation["track_id"]

            x = annotation["x"]
            y = annotation["y"]
            w = annotation["width"]
            h = annotation["height"]

            x2 = x + w
            y2 = y + h

            cv2.rectangle(
                frame,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        writer.write(frame)

    writer.release()

    print()
    print("Video creato:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()