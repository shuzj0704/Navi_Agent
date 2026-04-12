"""YOLOE real-time segmentation test for indoor objects."""
import os, sys
_VLN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_SRC_ROOT = os.path.join(_VLN_ROOT, "Navi_Agent", "src")
sys.path.insert(0, _VLN_ROOT)
sys.path.insert(0, _SRC_ROOT)
os.chdir(_VLN_ROOT)

import cv2
from ultralytics import YOLOE

# Target classes
CLASSES = [
    "refrigerator", "chair", "table", "computer", "monitor", "laptop",
    "cup", "pillow", "sofa", "door", "cardboard box", "trash can",
]

def main():
    # Load YOLOE segmentation model
    model = YOLOE("Navi_Agent/models/yoloe-11l-seg.pt")
    model.set_classes(CLASSES)

    # Open default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera, trying test image instead...")
        # Fallback: use a test image
        results = model.predict(
            source="https://ultralytics.com/images/bus.jpg",
            show=True,
            conf=0.3,
        )
        cv2.waitKey(0)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"YOLOE loaded. Detecting: {CLASSES}")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("YOLOE Segmentation", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
