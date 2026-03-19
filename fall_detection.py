import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2
from ultralytics import YOLO

Gst.init(None)

model = YOLO("yolov8n.pt")

pipeline = Gst.parse_launch(
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink name=sink max-buffers=1 drop=true sync=false"
)

appsink = pipeline.get_by_name("sink")

pipeline.set_state(Gst.State.PLAYING)

print("Pipeline started...")

while True:
    sample = appsink.try_pull_sample(100000000)  # 100ms timeout

    if sample is None:
        print("No frame yet...")
        continue

    buf = sample.get_buffer()
    caps = sample.get_caps()

    height = caps.get_structure(0).get_value("height")
    width = caps.get_structure(0).get_value("width")

    success, map_info = buf.map(Gst.MapFlags.READ)
    if not success:
        continue

    frame = np.frombuffer(map_info.data, dtype=np.uint8)
    frame = frame.reshape((height, width, 3))

    buf.unmap(map_info)

    print("Frame received")

    # YOLO inference
    results = model(frame, imgsz=320, verbose=False)

    annotated = results[0].plot()

    cv2.imshow("YOLO CSI", annotated)

    if cv2.waitKey(1) == 27:
        break

pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()