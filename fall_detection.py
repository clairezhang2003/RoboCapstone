import cv2
from ultralytics import YOLO
from collections import deque

# Load TensorRT YOLO model
model = YOLO("yolov8n.engine", task="detect")

# GStreamer pipeline for CSI camera
def gstreamer_pipeline(w=640, h=480, fps=30, flip=0):
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1640, height=1232, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink" % (fps, flip, w, h)
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("MAV Vision System Active. Press 'q' to quit.")

# Fall detection state
aspect_history = deque(maxlen=10)
prev_center_y = None
fall_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference (optimized)
    results = model(frame, imgsz=320, conf=0.25)
    r = results[0]
    annotated_frame = r.plot()

    # -----------------------------
    # Select largest person
    # -----------------------------
    largest_box = None
    max_area = 0

    for box in r.boxes:
        if int(box.cls[0]) != 0:  # person only
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area > max_area:
            max_area = area
            largest_box = (x1, y1, x2, y2)

    # -----------------------------
    # Fall detection logic
    # -----------------------------
    if largest_box is not None:
        x1, y1, x2, y2 = largest_box

        width = x2 - x1
        height = y2 - y1

        if height > 0:
            aspect_ratio = width / height
            center_y = (y1 + y2) / 2

            # Motion
            if prev_center_y is not None:
                delta_y = center_y - prev_center_y
            else:
                delta_y = 0

            prev_center_y = center_y

            # Normalize motion
            delta_y_norm = delta_y / height

            # Smooth aspect ratio
            aspect_history.append(aspect_ratio)
            avg_ratio = sum(aspect_history) / len(aspect_history)

            # -----------------------------
            # Improved fall condition
            # -----------------------------
            if (avg_ratio > 1.0 and delta_y_norm > 0.02) or (avg_ratio > 1.3):
                fall_counter += 1
            else:
                fall_counter = max(0, fall_counter - 1)

            # -----------------------------
            # Trigger detection
            # -----------------------------
            if fall_counter > 3:
                label = "FALL DETECTED"
                color = (0, 0, 255)

                # Optional: console alert (one-time trigger)
                if fall_counter == 4:
                    print("⚠️ FALL DETECTED")

            else:
                label = "Person"
                color = (0, 255, 0)

            # Draw result
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            # Debug (optional)
            # print(f"AR={aspect_ratio:.2f}, avg={avg_ratio:.2f}, dY={delta_y:.1f}, norm={delta_y_norm:.3f}, count={fall_counter}")

    # Display
    cv2.imshow("Jetson Nano MAV Feed", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()