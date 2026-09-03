"""
detect.py — YOLO26 Object Detection
Supports three modes:
  image  : run detection on a single image file
  video  : run detection on a video file
  webcam : run detection live from your webcam

Usage examples:
  python detect.py --mode image  --source inputs/photo.jpg
  python detect.py --mode video  --source inputs/clip.mp4
  python detect.py --mode webcam
  python detect.py --mode image  --source inputs/photo.jpg --model yolo26s.pt --conf 0.4
"""

import argparse
import os
import sys
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEFAULT_MODEL  = os.getenv("YOLO_MODEL", "yolo26s.pt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str) -> YOLO:
    """Load a YOLO model. Downloads automatically if a built-in name is given."""
    print(f"[INFO] Loading model: {model_path}")
    return YOLO(model_path)


def ensure_output_dir(path: str) -> str:
    """Create the outputs directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def draw_boxes(frame, results, conf_threshold: float):
    """
    Draw bounding boxes and labels on a frame in-place.
    Returns the annotated frame.
    """
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < conf_threshold:
                continue

            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Class label
            class_id = int(box.cls[0])
            label = result.names[class_id]
            text = f"{label} {confidence:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w, y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                frame,
                text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
    return frame


# ---------------------------------------------------------------------------
# Detection modes
# ---------------------------------------------------------------------------

def detect_image(model: YOLO, source: str, output_dir: str, conf: float) -> None:
    """Run detection on a single image and save the result."""
    if not os.path.isfile(source):
        print(f"[ERROR] Image not found: {source}")
        sys.exit(1)

    frame = cv2.imread(source)
    if frame is None:
        print(f"[ERROR] Could not read image: {source}")
        sys.exit(1)

    results = model(frame, conf=conf, verbose=False)
    annotated = draw_boxes(frame, results, conf)

    # Count detections
    total = sum(len(r.boxes) for r in results)
    print(f"[INFO] Detected {total} object(s) in '{source}'")

    # Save output
    basename = os.path.basename(source)
    out_path = os.path.join(output_dir, f"detected_{basename}")
    cv2.imwrite(out_path, annotated)
    print(f"[INFO] Saved result to: {out_path}")

    # Show result (press any key to close)
    cv2.imshow("Detection Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_video(model: YOLO, source: str, output_dir: str, conf: float) -> None:
    """Run detection on every frame of a video and save the annotated video."""
    if not os.path.isfile(source):
        print(f"[ERROR] Video not found: {source}")
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {source}")
        sys.exit(1)

    # Video writer setup
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    basename = os.path.splitext(os.path.basename(source))[0]
    out_path = os.path.join(output_dir, f"detected_{basename}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"[INFO] Processing video: {source}  ({total_frames} frames)")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results  = model(frame, conf=conf, verbose=False)
        annotated = draw_boxes(frame, results, conf)
        writer.write(annotated)

        # Live preview (press 'q' to abort early)
        cv2.imshow("Video Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Interrupted by user.")
            break

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[INFO]  {frame_idx}/{total_frames} frames processed...")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved annotated video to: {out_path}")


def detect_webcam(model: YOLO, conf: float, camera_index: int = 0) -> None:
    """Run real-time detection on webcam feed. Press 'q' to quit."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}.")
        sys.exit(1)

    print("[INFO] Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame.")
            break

        results   = model(frame, conf=conf, verbose=False)
        annotated = draw_boxes(frame, results, conf)

        # FPS overlay
        cv2.putText(
            annotated,
            "Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

        cv2.imshow("Webcam Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8 Object Detection — image / video / webcam"
    )
    parser.add_argument(
        "--mode",
        choices=["image", "video", "webcam"],
        required=True,
        help="Detection mode: image | video | webcam",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to input image or video (not needed for webcam mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="YOLO model weights file or name (default: from .env → yolo26s.pt). "
             "Options: yolo26n.pt  yolo26s.pt  yolo26m.pt  yolo26l.pt  yolo26x.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Directory to save detection results (default: outputs/)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index for webcam mode (default: 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate source for image/video modes
    if args.mode in ("image", "video") and args.source is None:
        print(f"[ERROR] --source is required for mode '{args.mode}'")
        sys.exit(1)

    model      = load_model(args.model)
    output_dir = ensure_output_dir(args.output)

    if args.mode == "image":
        detect_image(model, args.source, output_dir, args.conf)
    elif args.mode == "video":
        detect_video(model, args.source, output_dir, args.conf)
    elif args.mode == "webcam":
        detect_webcam(model, args.conf, args.camera)


if __name__ == "__main__":
    main()
