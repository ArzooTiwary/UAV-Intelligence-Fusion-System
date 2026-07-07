import sys
import json
from ultralytics import YOLO

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PEOPLE_CLASSES = {"person"}
BOAT_CLASSES = {"boat"}

model = YOLO("yolov8n.pt")


def detect_counts(image_path, conf_threshold=0.4):
    results = model.predict(source=image_path, conf=conf_threshold, verbose=False)
    result = results[0]
    counts = {"vehicles": 0, "people": 0, "boats": 0}
    if result.boxes is None:
     return counts
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    for cls_id in class_ids:
        class_name = result.names[cls_id]
        if class_name in VEHICLE_CLASSES:
            counts["vehicles"] += 1
        elif class_name in PEOPLE_CLASSES:
            counts["people"] += 1
        elif class_name in BOAT_CLASSES:
            counts["boats"] += 1

    return counts


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "bus.jpg"
    result = detect_counts(image_path)
    print(json.dumps(result, indent=2))

    with open("last_detection.json", "w") as f:
        json.dump(result, f, indent=2)