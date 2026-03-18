import cv2
from ultralytics import YOLO
import time

# -----------------------------
# Load model
# -----------------------------
print("Loading YOLOv8n model...")
model = YOLO("yolov8n.pt")  # or "yolov8n.engine" if using TensorRT
print("Model loaded!")

# -----------------------------
# GStreamer pipeline (CSI camera)
# -----------------------------
def gstreamer_pipeline(width=640, height=480, fps=30):
    return (
        "nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! appsink"
    )

# -----------------------------
# Open camera
# -----------------------------
print("Opening camera...")
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("❌ ERROR: Camera failed to open")
    exit()

print("✅ Camera opened successfully")

# -----------------------------
# Inference loop
# -----------------------------
prev_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run inference (reduced size for Nano)
    results = model(frame, imgsz=416, verbose=False)

    # Draw detections
    for box in results[0].boxes:
        cls = int(box.cls[0])

        if cls == 0:  # person class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

    cv2.imshow("YOLOv8n Jetson Nano", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()