import cv2
from ultralytics import YOLO
from collections import deque
import math

# -----------------------------
# Load YOLOv8 Pose TensorRT model
# -----------------------------
# Replace with your engine file for pose
model = YOLO("yolov8n-pose.engine", task="pose")  

# -----------------------------
# GStreamer pipeline for CSI camera
# -----------------------------
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

# -----------------------------
# Fall detection variables
# -----------------------------
aspect_history = deque(maxlen=10)
prev_head_y = None
fall_counter = 0
last_box = None
lost_frames = 0
MAX_LOST = 10

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # YOLOv8 Pose inference
    # -----------------------------
    results = model(frame, imgsz=320, conf=0.25)  # smaller size = faster
    r = results[0]
    annotated_frame = r.plot()

    # -----------------------------
    # Select largest person
    # -----------------------------
    largest_box = None
    max_area = 0

    for box in r.boxes:
        if int(box.cls[0]) != 0:  # person class
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest_box = (x1, y1, x2, y2)

    # -----------------------------
    # Tracking fallback
    # -----------------------------
    if largest_box is not None:
        last_box = largest_box
        lost_frames = 0
    else:
        lost_frames += 1

    if largest_box is None and last_box is not None and lost_frames < MAX_LOST:
        largest_box = last_box  # use fallback box

    # -----------------------------
    # Fall detection logic
    # -----------------------------
    if largest_box is not None and r.keypoints is not None:
        # Use keypoints: 0=head, 8=mid-hip
        kpts = r.keypoints[0].cpu().numpy()  # shape [17,3] xy + confidence
        head_y = kpts[0][1]
        hip_y = kpts[8][1]
        torso_height = hip_y - head_y

        # Body angle (horizontal = fallen)
        shoulder_x = (kpts[5][0] + kpts[6][0]) / 2
        hip_x = (kpts[11][0] + kpts[12][0]) / 2
        torso_dx = hip_x - shoulder_x
        torso_dy = hip_y - head_y
        angle_deg = abs(math.degrees(math.atan2(torso_dy, torso_dx)))

        # Motion: head drop
        if prev_head_y is not None:
            delta_y = head_y - prev_head_y
        else:
            delta_y = 0
        prev_head_y = head_y

        delta_y_norm = delta_y / torso_height if torso_height > 0 else 0

        # -----------------------------
        # Smoothing + heuristics
        # -----------------------------
        aspect_ratio = torso_dx / torso_dy if torso_dy > 0 else 0
        aspect_history.append(aspect_ratio)
        avg_ratio = sum(aspect_history) / len(aspect_history)

        # Fall condition
        if (avg_ratio > 1.0 and delta_y_norm > 0.02) or (avg_ratio > 1.2) or (angle_deg < 45):
            fall_counter += 1
        else:
            fall_counter = max(0, fall_counter - 1)

        # -----------------------------
        # Trigger detection
        # -----------------------------
        x1, y1, x2, y2 = largest_box
        if fall_counter > 3:
            label = "FALL DETECTED"
            color = (0, 0, 255)
            if fall_counter == 4:
                print("⚠️ FALL DETECTED")
        else:
            label = "Person"
            color = (0, 255, 0)

        # Draw
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # -----------------------------
    # Display
    # -----------------------------
    cv2.imshow("Jetson Nano MAV Feed", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()