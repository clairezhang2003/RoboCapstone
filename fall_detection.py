import cv2
from ultralytics import YOLO
from collections import deque
import math
import time

def gstreamer_pipeline(w=640, h=480, fps=30, flip=0):
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=640, height=480, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink" % (fps, flip, w, h)
    )


model = YOLO("yolov8n-pose.engine", task="pose")

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

aspect_history = deque(maxlen=10)
prev_head_y = None
fall_counter = 0
prev_time = 0

print("MAV Vision System Active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference with lower image size for speed
    # verbose=False reduces console spam which can slow down Python loops
    results = model(frame, imgsz=160, conf=0.25, verbose=False)
    r = results[0]
    annotated_frame = r.plot()

    # Select largest person
    largest_box = None
    max_area = 0
    person_idx = -1

    for i, box in enumerate(r.boxes):
        if int(box.cls[0]) == 0:  # person class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_box = (x1, y1, x2, y2)
                person_idx = i

    if person_idx != -1 and r.keypoints is not None:
        # FIX: Access the xy coordinates for the specific person detected
        # r.keypoints.xy is a tensor of shape [Num_Persons, 17, 2]
        kpts = r.keypoints.xy[person_idx].cpu().numpy()

        # Ensure we have the minimum required keypoints 
        if len(kpts) > 12:
            head_y = kpts[0][1]
            # Average hip position for stability
            hip_y = (kpts[11][1] + kpts[12][1]) / 2
            torso_height = hip_y - head_y

            # Body angle (horizontal = fallen)
            shoulder_x = (kpts[5][0] + kpts[6][0]) / 2
            hip_x = (kpts[11][0] + kpts[12][0]) / 2
            torso_dx = hip_x - shoulder_x
            torso_dy = hip_y - head_y
            angle_deg = abs(math.degrees(math.atan2(torso_dy, torso_dx)))

            # Head drop motion
            delta_y_norm = 0
            if prev_head_y is not None and torso_height > 0:
                delta_y_norm = (head_y - prev_head_y) / torso_height
            prev_head_y = head_y

            # Smoothing aspect ratio
            aspect_ratio = abs(torso_dx / torso_dy) if torso_dy != 0 else 0
            aspect_history.append(aspect_ratio)
            avg_ratio = sum(aspect_history) / len(aspect_history)

            # Condition Check
            if (avg_ratio > 1.2) or (angle_deg < 45 and delta_y_norm > 0.02):
                fall_counter += 1
            else:
                fall_counter = max(0, fall_counter - 1)

            # UI Feedback
            x1, y1, x2, y2 = largest_box
            if fall_counter > 3:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated_frame, "FALL DETECTED", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Jetson Nano MAV Feed", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
