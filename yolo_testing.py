import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.engine", task="detect")

def gstreamer_pipeline(w=640, h=480, fps=30, flip=0):
    return (
        "nvarguscamerasrc sensor-id=0 ! " # Verified Sensor ID
        "video/x-raw(memory:NVMM), width=(int)1640, height=(int)1232, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink" % (fps, flip, w, h)
    )

cap = cv2.VideoCapture(gstreamer_pipeline(w=640, h=640, fps=30), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("MAV Vision System Active. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, conf=0.25, stream=True)

    for r in results:
        annotated_frame = r.plot()
        
        # FALL DETECTION LOGIC HERE
        
        cv2.imshow("Jetson Nano MAV Feed", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()