import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from collections import deque
import math
import time

class FallDetectionNode(Node):
    def __init__(self):
        super().__init__('fall_detection_node')
        self.publisher_ = self.create_publisher(Image, 'vision/annotated_frame', 10)
        self.bridge = CvBridge()
        
        # model setup
        self.model = YOLO("yolov8n-pose.engine", task="pose")
        
        # GStreamer pipeline
        self.cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera.")
        
        self.aspect_history = deque(maxlen=10)
        self.prev_head_y = None
        self.fall_counter = 0
        self.prev_time = time.time()

        self.timer = self.create_timer(0.03, self.process_frame)

    def gstreamer_pipeline(w=640, h=480, fps=30, flip=0):
        return (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1640, height=1232, framerate=%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=%d, height=%d, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink" % (fps, flip, w, h)
        )
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            break
    
        # Inference with lower image size for speed
        # verbose=False reduces console spam which can slow down Python loops
        results = self.model(frame, imgsz=160, conf=0.25, verbose=False)
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
                if self.prev_head_y is not None and torso_height > 0:
                    delta_y_norm = (head_y - self.prev_head_y) / torso_height
                self.prev_head_y = head_y
    
                # Smoothing aspect ratio
                current_aspect = abs(torso_dx / torso_dy) if torso_dy != 0 else 0
                self.aspect_history.append(current_aspect)
                avg_ratio = sum(self.aspect_history) / len(self.aspect_history)
    
                # Condition Check
                if (avg_ratio > 1.2) or (angle_deg < 45 and delta_y_norm > 0.02):
                    self.fall_counter += 1
                else:
                    self.fall_counter = max(0, self.fall_counter - 1)
    
                # UI Feedback
                x1, y1, x2, y2 = largest_box
                color = (0, 0, 255) if self.fall_counter > 3 else (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                if self.fall_counter > 3:
                    cv2.putText(annotated_frame, "FALL DETECTED", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
        ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.publisher_.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    node = FallDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
