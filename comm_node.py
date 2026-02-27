import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty, Trigger

#added new
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import threading
import time

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
TAKEOFF_ALTITUDE   = 0.50   # metres (primary altitude goal)
HOVER_DURATION     = 30.0   # seconds (test duration)
LAND_DESCENT_RATE  = 0.10   # m/s downward
LAND_ALTITUDE      = 0.05   # consider landed below this height (metres)
OFFBOARD_RATE      = 20.0   # Hz for setpoint publishing

# ====================================================
# LAUNCH handler
# ====================================================
def handle_launch(self):
    self.get_logger().info('LAUNCH received – ascending to %.2f m' % TAKEOFF_ALTITUDE)

    # Build takeoff setpoint (hold current x,y; rise to target altitude)
    with self._lock:
        takeoff_pose = PoseStamped()
        takeoff_pose.header.frame_id = 'map'
        takeoff_pose.pose.position.x = self.current_pose.pose.position.x
        takeoff_pose.pose.position.y = self.current_pose.pose.position.y
        takeoff_pose.pose.position.z = TAKEOFF_ALTITUDE
        takeoff_pose.pose.orientation = self.current_pose.pose.orientation
        self.hold_pose = takeoff_pose

    # Pre-stream setpoints for a short while before requesting OFFBOARD
    time.sleep(0.5)

    # Switch to OFFBOARD and arm (only if not already)
    if self.drone_state.mode != 'OFFBOARD':
        if not self._set_mode('OFFBOARD'):
            self.get_logger().warn('Could not set OFFBOARD mode – pilot may need to switch manually')

    if not self.drone_state.armed:
        if not self._set_arm(True):
            self.get_logger().warn('Could not arm – is the vehicle already armed?')

    # Wait until target altitude is reached (±5 cm)
    self.get_logger().info('Climbing …')
    deadline = time.time() + 15.0   # 15 s timeout
    while time.time() < deadline:
        with self._lock:
            z = self.current_pose.pose.position.z
        if abs(z - TAKEOFF_ALTITUDE) < 0.05:
            self.get_logger().info('Target altitude reached (%.3f m)' % z)
            break
        time.sleep(0.1)
    else:
        self.get_logger().warn('Altitude not reached within timeout – proceeding anyway')

# ====================================================
# TEST handler
# ====================================================
def handle_test(self):
    self.get_logger().info('TEST received – stationkeeping for %.0f s' % HOVER_DURATION)

    # Latch current pose as the hold target
    with self._lock:
        self.hold_pose = PoseStamped()
        self.hold_pose.header.frame_id = 'map'
        self.hold_pose.pose = self.current_pose.pose
        self.mission_active = True

    start = time.time()
    while time.time() - start < HOVER_DURATION:
        elapsed = time.time() - start
        if int(elapsed) % 5 == 0:   # log every ~5 s
            with self._lock:
                p = self.current_pose.pose.position
            self.get_logger().info(
                'Stationkeeping … %.1f s  pos=(%.3f, %.3f, %.3f)' %
                (elapsed, p.x, p.y, p.z))
        time.sleep(1.0)

    self.get_logger().info('Test duration elapsed.')
    with self._lock:
        self.mission_active = False

# ====================================================
# LAND handler
# ====================================================
def handle_land(self):
    self.get_logger().info('LAND received – descending for soft landing')

    # Gradually lower the setpoint until on ground
    with self._lock:
        land_pose = PoseStamped()
        land_pose.header.frame_id = 'map'
        land_pose.pose.position.x = self.current_pose.pose.position.x
        land_pose.pose.position.y = self.current_pose.pose.position.y
        land_pose.pose.position.z = self.current_pose.pose.position.z
        land_pose.pose.orientation = self.current_pose.pose.orientation

    step = LAND_DESCENT_RATE / OFFBOARD_RATE   # metres per timer tick

    while True:
        with self._lock:
            current_z = self.current_pose.pose.position.z
            land_pose.pose.position.z = max(
                land_pose.pose.position.z - step, 0.0)
            self.hold_pose = land_pose

        if current_z < LAND_ALTITUDE:
            self.get_logger().info('Landed (z = %.3f m)' % current_z)
            break
        time.sleep(1.0 / OFFBOARD_RATE)

    # Disarm after landing
    time.sleep(1.0)
    self._set_arm(False)
    with self._lock:
        self.hold_pose = None

# ====================================================
# ABORT handler
# ====================================================
def handle_abort(self):
    self.get_logger().warn('ABORT received – landing immediately!')
    with self._lock:
        self.mission_active = False
    self.handle_land()

# ====================================================
# Service callbacks (called in the ROS executor thread)
# Spawn handlers in background threads so services return quickly
# ====================================================
def callback_launch(self, request, response):
    threading.Thread(target=self.handle_launch, daemon=True).start()
    response.success = True
    response.message = 'Launch command accepted'
    return response

def callback_test(self, request, response):
    threading.Thread(target=self.handle_test, daemon=True).start()
    response.success = True
    response.message = 'Test command accepted'
    return response

def callback_land(self, request, response):
    threading.Thread(target=self.handle_land, daemon=True).start()
    response.success = True
    response.message = 'Land command accepted'
    return response

def callback_abort(self, request, response):
    threading.Thread(target=self.handle_abort, daemon=True).start()
    response.success = True
    response.message = 'Abort command accepted'
    return response


# ====================================================
# Pose / state callbacks
# ====================================================
def callback_pose(self, msg: PoseStamped):
    with self._lock:
        self.current_pose = msg

def callback_state(self, msg: State):
    self.drone_state = msg

# ====================================================
# Setpoint publisher  (runs at OFFBOARD_RATE Hz)
# ====================================================
def _publish_setpoint(self):
    with self._lock:
        if self.hold_pose is not None:
            sp = PoseStamped()
            sp.header.stamp    = self.get_clock().now().to_msg()
            sp.header.frame_id = 'map'
            sp.pose            = self.hold_pose.pose
            self.pub_setpoint.publish(sp)
        else:
            # Publish current position so OFFBOARD mode can be entered
            sp = PoseStamped()
            sp.header.stamp    = self.get_clock().now().to_msg()
            sp.header.frame_id = 'map'
            sp.pose            = self.current_pose.pose
            self.pub_setpoint.publish(sp)

# ====================================================
# Helper – set FCU mode
# ====================================================
def _set_mode(self, mode: str) -> bool:
    req = SetMode.Request()
    req.custom_mode = mode
    if not self.cli_set_mode.wait_for_service(timeout_sec=3.0):
        self.get_logger().error('set_mode service not available')
        return False
    future = self.cli_set_mode.call_async(req)
    rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
    return future.result() is not None and future.result().mode_sent

# ====================================================
# Helper – arm / disarm
# ====================================================
def _set_arm(self, arm: bool) -> bool:
    req = CommandBool.Request()
    req.value = arm
    if not self.cli_arming.wait_for_service(timeout_sec=3.0):
        self.get_logger().error('arming service not available')
        return False
    future = self.cli_arming.call_async(req)
    rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
    return future.result() is not None and future.result().success


# -------------------------------------------------------
# CommNode
# -------------------------------------------------------
class CommNode(Node):

    def __init__(self):
        super().__init__('rob498_drone_02')

        # ── Internal state ──────────────────────────────
        self.current_pose   = PoseStamped()   # latest Vicon / EKF pose
        self.hold_pose      = None            # pose to hold during stationkeeping
        self.drone_state    = State()         # MAVROS FCU state
        self.mission_active = False
        self._lock          = threading.Lock()

        # ── MAVROS subscribers ───────────────────────────
        self.sub_pose  = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose',
            self.callback_pose, 10)
        self.sub_state = self.create_subscription(
            State, '/mavros/state',
            self.callback_state, 10)

        # ── MAVROS setpoint publisher ────────────────────
        self.pub_setpoint = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)

        # ── MAVROS service clients ───────────────────────
        self.cli_arming   = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.cli_set_mode = self.create_client(SetMode,     '/mavros/set_mode')

        # ── Ground-control service servers ──────────────
        self.srv_launch = self.create_service(
            Trigger, 'rob498_drone_02/comm/launch', self.callback_launch)
        self.srv_test   = self.create_service(
            Trigger, 'rob498_drone_02/comm/test',   self.callback_test)
        self.srv_land   = self.create_service(
            Trigger, 'rob498_drone_02/comm/land',   self.callback_land)
        self.srv_abort  = self.create_service(
            Trigger, 'rob498_drone_02/comm/abort',  self.callback_abort)

        # ── Setpoint timer (must be publishing before OFFBOARD) ─
        self.timer = self.create_timer(1.0 / OFFBOARD_RATE, self._publish_setpoint)

        self.get_logger().info('CommNode initialised. Waiting for commands.')

# -------------------------------------------------------
# Entry point
# -------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()