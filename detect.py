import cv2
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0", help="Video file path or 0 for webcam")
parser.add_argument("--weights", default="runs/detect/ppe_detector/weights/best.pt")
args = parser.parse_args()

model = YOLO(args.weights)

source = int(args.source) if args.source == "0" else args.source
cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("PPE Detector", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
