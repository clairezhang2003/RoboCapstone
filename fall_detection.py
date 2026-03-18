import cv2
from ultralytics import YOLO
from collections import deque

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("test_video.mp4")

history = deque(maxlen=10)
prev_center_y = None
fall_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=416, verbose=False)

    for box in results[0].boxes:
        cls = int(box.cls[0])

        if cls == 0:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height

            center_y = (y1 + y2) / 2

            # motion
            if prev_center_y is not None:
                delta_y = center_y - prev_center_y
            else:
                delta_y = 0

            prev_center_y = center_y

            # smoothing
            history.append(aspect_ratio)
            avg_ratio = sum(history) / len(history)

            # fall logic
            if avg_ratio > 1.2 and delta_y > 15:
                fall_counter += 1
            else:
                fall_counter = 0

            if fall_counter > 5:
                label = "FALL DETECTED"
                color = (0, 0, 255)
            else:
                label = "Person"
                color = (0, 255, 0)

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Fall Detection Test", frame)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()