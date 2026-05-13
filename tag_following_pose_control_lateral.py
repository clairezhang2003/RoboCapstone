import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
import rclpy.qos
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import copy
from std_srvs.srv import Trigger
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool
from geometry_msgs.msg import PoseStamped
import numpy as np
import threading
import cv2
from cv2 import aruco
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

TEST_MODE = False

GROUND = 'GROUND'
CONNECT = 'CONNECT'
TAKEOFF = 'TAKEOFF'
HOVER = 'HOVER'
FOLLOW = 'FOLLOW'
LAND = 'LAND'
ABORT = 'ABORT'

LOCAL_GOAL_TOLERANCE = 0.15
GOAL_TOLERANCE = 0.05
TAKEOFF_INCREMENT = 0.15
LANDING_INCREMENT = 0.15

TAG_SIZE = 0.15              # [m] physical side length of the AprilTag
FLYING_HEIGHT = 0.78         # [m] hover altitude above ground
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOLLOW_DISTANCE = 1.5        # [m] desired stand-off distance from the tag
DISTANCE_ALPHA = 0.2
DETECTION_TIMEOUT = 1.0
MIN_BOX_HEIGHT_PX = 5.0

K_LATERAL = 0.5              # [m] lateral movement per normalized pixel error
K_DISTANCE = 0.3             # proportional gain for distance error
MAX_LATERAL = 2.0            # [m] don't chase more than this far sideways

COMMAND = 'ground'
MODE = GROUND
land_initialized = False

# Load camera calibration
try:
    camera_matrix = np.load('camera_matrix.npy')
    dist_coeffs = np.load('dist_coeffs.npy')
    FOCAL_LENGTH_PX = camera_matrix[0, 0]
    print(f"Camera calibration loaded. fx={FOCAL_LENGTH_PX:.2f}px")
except FileNotFoundError:
    print("ERROR: camera_matrix.npy or dist_coeffs.npy not found.")
    print("Run calibration_run.py first.")
    exit(1)


def handle_launch():
    global COMMAND
    COMMAND = 'launch'
    print('Launch Requested. Your drone should take off.')


def handle_test():
    global COMMAND
    COMMAND = 'test'
    print('Test Requested. Following AprilTag.')


def handle_land():
    global COMMAND
    COMMAND = 'land'
    print('Land Requested. Your drone should land.')


def handle_abort():
    global COMMAND
    COMMAND = 'abort'
    print('Abort Requested. Your drone should land immediately.')


def callback_launch(request, response):
    handle_launch()
    response.success = True
    return response


def callback_test(request, response):
    handle_test()
    response.success = True
    return response


def callback_land(request, response):
    handle_land()
    response.success = True
    return response


def callback_abort(request, response):
    handle_abort()
    response.success = True
    return response


class CommNode(Node):
    def __init__(self):
        super().__init__('rob498_drone_2')

        self.srv_launch = self.create_service(
            Trigger, 'rob498_drone_2/comm/launch', callback_launch
        )
        self.srv_test = self.create_service(
            Trigger, 'rob498_drone_2/comm/test', callback_test
        )
        self.srv_land = self.create_service(
            Trigger, 'rob498_drone_2/comm/land', callback_land
        )
        self.srv_abort = self.create_service(
            Trigger, 'rob498_drone_2/comm/abort', callback_abort
        )

        self.image_pub = None
        self.bridge = None
        if TEST_MODE:
            self.image_pub = self.create_publisher(
                Image, 'rob498_drone_2/camera/annotated_feed', 10
            )
            self.bridge = CvBridge()

        self.rate = self.create_rate(30)

        qos_mavros = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.state_sub = self.create_subscription(
            State,
            'mavros/state',
            callback=self.state_callback,
            qos_profile=qos_mavros
        )
        self.state = State()

        self.odom_sub = self.create_subscription(
            PoseStamped,
            'mavros/local_position/pose',
            callback=self.odom_callback,
            qos_profile=rclpy.qos.qos_profile_system_default
        )
        self.odom_pose = None
        self.ground_z = None

        self.set_mode_cli = self.create_client(SetMode, 'mavros/set_mode')
        while not self.set_mode_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_mode service not available')

        self.arm_cli = self.create_client(CommandBool, 'mavros/cmd/arming')
        while not self.arm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('arming service not available')

        self.pose_pub = self.create_publisher(
            PoseStamped, 'mavros/setpoint_position/local', 10
        )

        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
        self.detector = aruco.ArucoDetector(self.aruco_dict)

        self.pipeline = (
            "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1640, height=1232, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, width=640, height=480, format=BGRx ! videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera pipeline!")

        self.frame_lock = threading.Lock()
        self.annotated_lock = threading.Lock()
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.latest_detection = None
        self.last_detection_time = None
        self.filtered_distance = None
        self.prev_filtered_distance = None

        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

        self.display_thread = None
        if TEST_MODE:
            self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
            self.display_thread.start()

    def state_callback(self, msg):
        self.state = msg

    def odom_callback(self, msg):
        self.odom_pose = msg
        if self.ground_z is None:
            self.ground_z = msg.pose.position.z

    def capture_loop(self):
        while rclpy.ok():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame

    def detect_tag(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        annotated = frame_undistorted.copy()

        corners_list, ids, _ = self.detector.detectMarkers(frame_undistorted)

        largest_tag = None
        max_area = 0

        if ids is not None:
            for i, tag_id in enumerate(ids):
                corners = corners_list[i][0]
                x_coords = corners[:, 0].astype(int)
                y_coords = corners[:, 1].astype(int)

                x1 = int(np.min(x_coords))
                y1 = int(np.min(y_coords))
                x2 = int(np.max(x_coords))
                y2 = int(np.max(y_coords))

                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_tag = (x1, y1, x2, y2, int(tag_id[0]), corners)

        if largest_tag is not None:
            x1, y1, x2, y2, tag_id, corners = largest_tag
            self.latest_detection = (x1, y1, x2, y2)
            self.last_detection_time = self.get_clock().now()

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            tag_height = abs(y2 - y1)
            tag_width = abs(x2 - x1)

            corners_int = corners.astype(int)
            cv2.polylines(annotated, [corners_int], True, (0, 255, 0), 3)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"Tag ID={tag_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )
            cv2.circle(annotated, (int(center_x), int(center_y)), 6, (0, 255, 0), -1)
            cv2.line(
                annotated,
                (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2),
                (int(center_x), int(center_y)),
                (255, 255, 0),
                1
            )

            if self.filtered_distance is not None:
                cv2.putText(
                    annotated,
                    f"dist={self.filtered_distance:.2f}m",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

            cv2.putText(
                annotated,
                f"tag_h={tag_height}px tag_w={tag_width}px",
                (x1, y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )
        else:
            self.latest_detection = None

        cv2.line(
            annotated,
            (IMAGE_WIDTH // 2 - 20, IMAGE_HEIGHT // 2),
            (IMAGE_WIDTH // 2 + 20, IMAGE_HEIGHT // 2),
            (255, 255, 255),
            1
        )
        cv2.line(
            annotated,
            (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2 - 20),
            (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2 + 20),
            (255, 255, 255),
            1
        )

        mode_label = f"{'[TEST MODE] ' if TEST_MODE else ''}MODE: {MODE}"
        cv2.putText(
            annotated,
            mode_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255) if TEST_MODE else (255, 255, 255),
            2
        )

        tag_count = len(ids) if ids is not None else 0
        cv2.putText(
            annotated,
            f"AprilTags: {tag_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )

        if TEST_MODE:
            cv2.putText(
                annotated,
                "NO FLIGHT COMMANDS SENT",
                (10, IMAGE_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        with self.annotated_lock:
            self.latest_annotated_frame = annotated

    def display_loop(self):
        while rclpy.ok():
            with self.annotated_lock:
                if self.latest_annotated_frame is None:
                    continue
                display_frame = self.latest_annotated_frame.copy()

            try:
                img_msg = self.bridge.cv2_to_imgmsg(display_frame, encoding="bgr8")
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = "camera_link"
                self.image_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish image: {e}")

            import time
            time.sleep(0.033)


def run_test_mode(node):
    print("TEST MODE")
    rate = node.create_rate(10)

    while rclpy.ok():
        node.detect_tag()

        detection_lost = (
            node.latest_detection is None or
            node.last_detection_time is None or
            (node.get_clock().now() - node.last_detection_time) > Duration(seconds=DETECTION_TIMEOUT)
        )

        if detection_lost:
            print("NO APRILTAG DETECTED; would hold position")
            node.filtered_distance = None
            node.prev_filtered_distance = None
            rate.sleep()
            continue

        x1, y1, x2, y2 = node.latest_detection
        center_x = (x1 + x2) / 2.0
        box_height_px = float(y2 - y1)

        if box_height_px < MIN_BOX_HEIGHT_PX:
            print(f"BOX TOO SMALL: {box_height_px:.1f}px; would ignore detection")
            rate.sleep()
            continue

        raw_distance = (TAG_SIZE * FOCAL_LENGTH_PX) / box_height_px

        if node.filtered_distance is None:
            node.filtered_distance = raw_distance
            node.prev_filtered_distance = None
        else:
            node.prev_filtered_distance = node.filtered_distance
            node.filtered_distance = (
                DISTANCE_ALPHA * raw_distance +
                (1.0 - DISTANCE_ALPHA) * node.filtered_distance
            )

        if node.prev_filtered_distance is not None:
            distance_change = node.filtered_distance - node.prev_filtered_distance
            direction = (
                "MOVING AWAY" if distance_change > 0.01 else
                "MOVING CLOSER" if distance_change < -0.01 else
                "STATIONARY"
            )
        else:
            distance_change = 0.0
            direction = "INITIALISING"

        pixel_error = center_x - (IMAGE_WIDTH / 2.0)
        norm_error = pixel_error / (IMAGE_WIDTH / 2.0)
        lateral_offset = norm_error * K_LATERAL

        distance_error = node.filtered_distance - FOLLOW_DISTANCE
        forward_offset = float(np.clip(distance_error * K_DISTANCE, -0.3, 0.3))

        print(
            f"APRILTAG DETECTED | "
            f"tag height={box_height_px:.0f}px | "
            f"raw distance={raw_distance:.2f}m filtered distance={node.filtered_distance:.2f}m | "
            f"change={distance_change:+.3f}m ({direction}) | "
            f"distance_error={distance_error:+.2f}m | "
            f"lateral_error={norm_error:+.2f} | "
            f"target offsets=(x:{forward_offset:+.2f}, y:{-lateral_offset:+.2f})"
        )

        rate.sleep()


def main(args=None):
    global COMMAND, MODE, land_initialized

    COMMAND = 'ground'
    MODE = GROUND
    land_initialized = False

    rclpy.init(args=args)
    node = CommNode()

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    node.get_logger().info("Node online.")

    if TEST_MODE:
        node.get_logger().info("TEST MODE")
        MODE = FOLLOW
        try:
            run_test_mode(node)
        except KeyboardInterrupt:
            pass
        rclpy.shutdown()
        thread.join()
        cv2.destroyAllWindows()
        return

    while rclpy.ok() and (node.odom_pose is None or node.ground_z is None):
        node.rate.sleep()

    goal_pos = PoseStamped()
    goal_pos.pose.position.x = node.odom_pose.pose.position.x
    goal_pos.pose.position.y = node.odom_pose.pose.position.y
    goal_pos.pose.position.z = node.ground_z + FLYING_HEIGHT
    goal_pos.pose.orientation = node.odom_pose.pose.orientation

    node.get_logger().info(
        f"Initial pose received. Ground z={node.ground_z:.3f}, "
        f"goal z={goal_pos.pose.position.z:.3f} (AprilTag height={TAG_SIZE}m)"
    )

    cmd = PoseStamped()
    cmd.pose = copy.deepcopy(node.odom_pose.pose)
    cmd.pose.position.z = node.ground_z
    cmd.header.frame_id = "map"
    cmd.header.stamp = node.get_clock().now().to_msg()

    while rclpy.ok() and not node.state.connected:
        node.rate.sleep()
    node.get_logger().info("Node connected.")

    offb_set_mode = SetMode.Request()
    offb_set_mode.custom_mode = "OFFBOARD"

    arm_cmd = CommandBool.Request()
    arm_cmd.value = True

    prev_request = node.get_clock().now()
    counter = 0
    counter_total = 100

    node.get_logger().info("Starting loop.")

    while rclpy.ok():
        if COMMAND == 'abort' and MODE != GROUND:
            MODE = ABORT
            COMMAND = 'ground'
        elif COMMAND == 'launch' and MODE == GROUND:
            MODE = CONNECT
            COMMAND = 'ground'
        elif COMMAND == 'test':
            if MODE == HOVER or MODE == FOLLOW:
                MODE = FOLLOW
                node.filtered_distance = None
                node.prev_filtered_distance = None
                node.get_logger().info("FOLLOW mode active.")
            else:
                node.get_logger().warning(f"Test command ignored - not ready (MODE={MODE})")
            COMMAND = 'ground'
        elif COMMAND == 'land' and MODE != GROUND:
            MODE = LAND
            land_initialized = False
            COMMAND = 'ground'

        node.detect_tag()

        detection_lost = (
            node.latest_detection is None or
            node.last_detection_time is None or
            (node.get_clock().now() - node.last_detection_time) > Duration(seconds=DETECTION_TIMEOUT)
        )

        tag_valid = False
        center_x = None
        box_height_px = None
        raw_distance = None
        distance_change = 0.0
        direction = "INITIALISING"

        if detection_lost:
            node.filtered_distance = None
            node.prev_filtered_distance = None
        else:
            x1, y1, x2, y2 = node.latest_detection
            center_x = (x1 + x2) / 2.0
            box_height_px = float(y2 - y1)

            if box_height_px >= MIN_BOX_HEIGHT_PX:
                raw_distance = (TAG_SIZE * FOCAL_LENGTH_PX) / box_height_px

                if node.filtered_distance is None:
                    node.filtered_distance = raw_distance
                    node.prev_filtered_distance = None
                else:
                    node.prev_filtered_distance = node.filtered_distance
                    node.filtered_distance = (
                        DISTANCE_ALPHA * raw_distance +
                        (1.0 - DISTANCE_ALPHA) * node.filtered_distance
                    )

                if node.prev_filtered_distance is not None:
                    distance_change = node.filtered_distance - node.prev_filtered_distance
                    direction = (
                        "MOVING AWAY" if distance_change > 0.01 else
                        "MOVING CLOSER" if distance_change < -0.01 else
                        "STATIONARY"
                    )

                tag_valid = True

        if detection_lost:
            print(f"MODE={MODE} | NO APRILTAG DETECTED; holding position")
        elif not tag_valid:
            print(f"MODE={MODE} | BOX TOO SMALL: {box_height_px:.1f}px; ignoring detection")
        else:
            print(
                f"MODE={MODE} | APRILTAG DETECTED | "
                f"tag height={box_height_px:.0f}px | "
                f"raw distance={raw_distance:.2f}m filtered distance={node.filtered_distance:.2f}m | "
                f"change={distance_change:+.3f}m ({direction})"
            )

        if MODE == CONNECT:
            if node.state.armed and node.state.mode == "OFFBOARD":
                MODE = TAKEOFF
                cmd.pose.position.z = node.odom_pose.pose.position.z
            else:
                if counter >= counter_total and node.get_clock().now() - prev_request > Duration(seconds=2.0):
                    if not node.state.armed:
                        if node.arm_cli.call(arm_cmd).success:
                            node.get_logger().info("Vehicle armed")
                    if node.state.armed and node.state.mode != "OFFBOARD":
                        if node.set_mode_cli.call(offb_set_mode).mode_sent:
                            node.get_logger().info("OFFBOARD enabled")
                    prev_request = node.get_clock().now()
                counter += 1

            cmd.pose.position.x = node.odom_pose.pose.position.x
            cmd.pose.position.y = node.odom_pose.pose.position.y
            cmd.pose.orientation = node.odom_pose.pose.orientation

        elif MODE == TAKEOFF:
            if np.abs(goal_pos.pose.position.z - node.odom_pose.pose.position.z) < GOAL_TOLERANCE:
                cmd.pose.position.z = goal_pos.pose.position.z
                goal_pos.pose.position.x = node.odom_pose.pose.position.x
                goal_pos.pose.position.y = node.odom_pose.pose.position.y
                node.get_logger().info("Takeoff complete. Hovering - send test to begin following.")
                MODE = HOVER
            elif np.abs(cmd.pose.position.z - node.odom_pose.pose.position.z) < LOCAL_GOAL_TOLERANCE:
                cmd.pose.position.z = min(
                    cmd.pose.position.z + TAKEOFF_INCREMENT,
                    goal_pos.pose.position.z
                )

        elif MODE == HOVER:
            cmd.pose.position.x = goal_pos.pose.position.x
            cmd.pose.position.y = goal_pos.pose.position.y
            cmd.pose.position.z = goal_pos.pose.position.z

        elif MODE == FOLLOW:
            if detection_lost:
                cmd.pose.position.x = node.odom_pose.pose.position.x
                cmd.pose.position.y = node.odom_pose.pose.position.y
                cmd.pose.position.z = node.ground_z + FLYING_HEIGHT
                node.get_logger().warning(
                    "AprilTag not detected. Holding position.",
                    throttle_duration_sec=2.0,
                    clock=node.get_clock()
                )
            elif not tag_valid:
                cmd.pose.position.x = node.odom_pose.pose.position.x
                cmd.pose.position.y = node.odom_pose.pose.position.y
                cmd.pose.position.z = node.ground_z + FLYING_HEIGHT
                node.get_logger().warning(
                    f"Box height {box_height_px:.1f}px too small — ignoring detection.",
                    throttle_duration_sec=1.0,
                    clock=node.get_clock()
                )
            else:
                pixel_error = center_x - (IMAGE_WIDTH / 2.0)
                norm_error = pixel_error / (IMAGE_WIDTH / 2.0)
                lateral_offset = norm_error * K_LATERAL

                distance_error = node.filtered_distance - FOLLOW_DISTANCE
                forward_offset = np.clip(distance_error * K_DISTANCE, -0.3, 0.3)

                target_x = node.odom_pose.pose.position.x + forward_offset
                target_y = node.odom_pose.pose.position.y - lateral_offset
                target_z = node.ground_z + FLYING_HEIGHT

                target_x = np.clip(
                    target_x,
                    node.odom_pose.pose.position.x - MAX_LATERAL,
                    node.odom_pose.pose.position.x + MAX_LATERAL
                )
                target_y = np.clip(
                    target_y,
                    node.odom_pose.pose.position.y - MAX_LATERAL,
                    node.odom_pose.pose.position.y + MAX_LATERAL
                )

                cmd.pose.position.x = target_x
                cmd.pose.position.y = target_y
                cmd.pose.position.z = target_z

                print(
                    f"MODE={MODE} | FOLLOWING | "
                    f"distance={node.filtered_distance:.2f}m (error={distance_error:+.2f}m) | "
                    f"lateral_error={norm_error:+.2f} | "
                    f"target=({target_x:.2f}, {target_y:.2f}, {target_z:.2f})"
                )

        elif MODE == LAND:
            if not land_initialized:
                cmd.pose.position.z = node.odom_pose.pose.position.z - LANDING_INCREMENT
                land_initialized = True

            if np.abs(cmd.pose.position.z - node.odom_pose.pose.position.z) < LOCAL_GOAL_TOLERANCE:
                cmd.pose.position.z -= LANDING_INCREMENT

            if abs(node.odom_pose.pose.position.z - node.ground_z) < LOCAL_GOAL_TOLERANCE:
                cmd.pose.position.z = node.ground_z

            if not node.state.armed and node.get_clock().now() - prev_request > Duration(seconds=5.0):
                offb_set_mode.custom_mode = "AUTO.LOITER"
                if node.set_mode_cli.call(offb_set_mode).mode_sent:
                    node.get_logger().info("Landed, switching to GROUND")
                    MODE = GROUND
                prev_request = node.get_clock().now()

        elif MODE == ABORT:
            offb_set_mode.custom_mode = "AUTO.LAND"
            if node.state.mode != "AUTO.LAND" and node.get_clock().now() - prev_request > Duration(seconds=0.5):
                if node.set_mode_cli.call(offb_set_mode).mode_sent:
                    node.get_logger().info("Landing mode enabled (ABORT)")
                prev_request = node.get_clock().now()

            if (
                node.state.mode == "AUTO.LAND" and
                not node.state.armed and
                node.get_clock().now() - prev_request > Duration(seconds=5.0)
            ):
                offb_set_mode.custom_mode = "STABILIZED"
                if node.set_mode_cli.call(offb_set_mode).mode_sent:
                    node.get_logger().info("Landed from ABORT, switching to GROUND")
                    MODE = GROUND
                prev_request = node.get_clock().now()

        elif MODE == GROUND:
            cmd.pose.position.x = node.odom_pose.pose.position.x
            cmd.pose.position.y = node.odom_pose.pose.position.y
            cmd.pose.position.z = node.ground_z

        cmd.header.stamp = node.get_clock().now().to_msg()
        node.pose_pub.publish(cmd)
        node.rate.sleep()

    rclpy.shutdown()
    thread.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()