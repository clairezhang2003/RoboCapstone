import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
import rclpy.time
import rclpy.qos
import copy
from std_srvs.srv import Trigger
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool
from geometry_msgs.msg import PoseStamped, Pose
import numpy as np
import threading

GROUND = 'GROUND'
WAIT = 'WAIT'
CONNECT = 'CONNECT'
TAKEOFF = 'TAKEOFF'
HOVER = 'HOVER'
LAND = 'LAND'
ABORT = 'ABORT'

LOCAL_GOAL_TOLERANCE = 0.15  # [m]
GOAL_TOLERANCE = 0.05
TAKEOFF_INCREMENT = 0.15     # [m]
LANDING_INCREMENT = 0.15

GOAL_HEIGHT = 0.3           # [m] above ground reference

COMMAND = 'ground'
MODE = GROUND

# Callback handlers
def handle_launch():
    global COMMAND
    COMMAND = 'launch'
    print('Launch Requested. Your drone should take off.')

def handle_test():
    global COMMAND
    COMMAND = 'test'
    print('Test Requested. Your drone should perform the required tasks. Recording starts now.')

def handle_land():
    global COMMAND
    COMMAND = 'land'
    print('Land Requested. Your drone should land.')

def handle_abort():
    global COMMAND
    COMMAND = 'abort'
    print('Abort Requested. Your drone should land immediately due to safety considerations')

# Service callbacks
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
        self.srv_launch = self.create_service(Trigger, 'rob498_drone_2/comm/launch', callback_launch)
        self.srv_test = self.create_service(Trigger, 'rob498_drone_2/comm/test', callback_test)
        self.srv_land = self.create_service(Trigger, 'rob498_drone_2/comm/land', callback_land)
        self.srv_abort = self.create_service(Trigger, 'rob498_drone_2/comm/abort', callback_abort)

        self.rate = self.create_rate(30)

        # state and odom
        self.state_sub = self.create_subscription(State, 'mavros/state', callback=self.state_callback, qos_profile=10)
        self.state = State()

        self.odom_sub = self.create_subscription(
            PoseStamped,
            'mavros/local_position/pose',
            callback=self.odom_callback,
            qos_profile=rclpy.qos.qos_profile_system_default
        )
        self.odom_pose = None     # will be set in odom_callback
        self.ground_z = None      # reference z when on ground

        # set mode client
        self.set_mode_cli = self.create_client(SetMode, 'mavros/set_mode')
        while not self.set_mode_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('set_mode service not available')

        # arming client
        self.arm_cli = self.create_client(CommandBool, 'mavros/cmd/arming')
        while not self.arm_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('arming service not available')

        # position setpoint publisher
        self.pose_pub = self.create_publisher(PoseStamped, 'mavros/setpoint_position/local', 10)

    # state callback
    def state_callback(self, msg):
        self.state = msg
        self.get_logger().debug(f"Received state {msg}")

    # odom callback
    def odom_callback(self, msg):
        self.odom_pose = msg
        # latch a ground reference once (assumed on ground at startup)
        if self.ground_z is None:
            self.ground_z = msg.pose.position.z
            #self.get_logger().info(f"Ground reference z set to {self.ground_z:.3f}")
        self.get_logger().debug(f"Received odom {msg}")


def main(args=None):
    global COMMAND, MODE

    COMMAND = 'ground'
    MODE = GROUND

    rclpy.init(args=args)
    node = CommNode()

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    node.get_logger().info("Node online.")

    # wait for first odom and ground_z
    while rclpy.ok() and (node.odom_pose is None or node.ground_z is None):
        node.rate.sleep()

    # hover goal: GOAL_HEIGHT above ground reference
    goal_pos = PoseStamped()
    goal_pos.pose = copy.deepcopy(node.odom_pose.pose)
    goal_pos.pose.position.z = node.ground_z + GOAL_HEIGHT
    node.get_logger().info(
        f"Initial pose received. Ground z={node.ground_z:.3f}, goal z={goal_pos.pose.position.z:.3f}"
    )

    # initial command pose: at ground reference
    cmd = PoseStamped()
    cmd.pose = copy.deepcopy(node.odom_pose.pose)
    cmd.pose.position.z = node.ground_z
    cmd.header.frame_id = "map"
    cmd.header.stamp = node.get_clock().now().to_msg()

    # wait to connect
    while rclpy.ok() and not node.state.connected:
        node.rate.sleep()
    node.get_logger().info("Node connected.")

    # for arm and offboard
    offb_set_mode = SetMode.Request()
    offb_set_mode.custom_mode = "OFFBOARD"
    arm_cmd = CommandBool.Request()
    arm_cmd.value = True

    prev_request = node.get_clock().now()
    counter = 0
    counter_total = 100

    node.get_logger().info("Starting loop.")

    while rclpy.ok():
        # state machine commands
        if COMMAND == 'abort' and MODE != GROUND:
            MODE = ABORT
            COMMAND = 'ground'
        elif COMMAND == 'launch' and MODE == GROUND:
            MODE = CONNECT
            COMMAND = 'ground'
        elif COMMAND == 'test':
            MODE = HOVER
            COMMAND = 'ground'
        elif COMMAND == 'land' and MODE != GROUND:
            MODE = LAND
            COMMAND = 'ground'

        #node.get_logger().info(f"Mode: {MODE}")

        if MODE == WAIT:
            pass

        elif MODE == CONNECT:
            # check if armed and offboard
            if node.state.armed and node.state.mode == "OFFBOARD":
                # once set, proceed to TAKEOFF
                MODE = TAKEOFF
                # reset local setpoint to current z (should be near ground_z)
                cmd.pose.position.z = node.odom_pose.pose.position.z
            else:
                if counter >= counter_total and node.get_clock().now() - prev_request > Duration(seconds=2.0):
                    node.get_logger().debug(f"current mode: {node.state.mode}")
                    if not node.state.armed:
                        node.get_logger().debug("attempting to arm")
                        if node.arm_cli.call(arm_cmd).success:
                            node.get_logger().info("Vehicle armed")
                    if node.state.armed and node.state.mode != "OFFBOARD":
                        node.get_logger().debug("attempting to offboard")
                        if node.set_mode_cli.call(offb_set_mode).mode_sent:
                            node.get_logger().info("OFFBOARD enabled")
                    prev_request = node.get_clock().now()
                counter += 1

            # keep x,y at current, z at current (near ground_z)
            cmd.pose.position.x = node.odom_pose.pose.position.x
            cmd.pose.position.y = node.odom_pose.pose.position.y
            cmd.pose.orientation = node.odom_pose.pose.orientation
            # z is whatever we initialized; you can also lock to ground_z here if desired
            # cmd.pose.position.z = node.ground_z

        elif MODE == TAKEOFF:
            # distance to final goal
            #node.get_logger().info(
                #f"local goal distance z: {np.abs(cmd.pose.position.z - node.odom_pose.pose.position.z)}"
            #)
            if np.abs(goal_pos.pose.position.z - node.odom_pose.pose.position.z) < GOAL_TOLERANCE:
                cmd.pose.position.z = goal_pos.pose.position.z
                MODE = HOVER
            elif np.abs(cmd.pose.position.z - node.odom_pose.pose.position.z) < LOCAL_GOAL_TOLERANCE:
                cmd.pose.position.z = min(
                    cmd.pose.position.z + TAKEOFF_INCREMENT,
                    goal_pos.pose.position.z
                )

        elif MODE == HOVER:
            # hold at goal_pos altitude, follow x,y from odom if you want
            cmd.pose.position.x = node.odom_pose.pose.position.x
            cmd.pose.position.y = node.odom_pose.pose.position.y
            cmd.pose.position.z = goal_pos.pose.position.z
            node.get_logger().info("Hovering")
        elif MODE == LAND:
            # first time entering LAND from hover, step down from current altitude
            if cmd.pose.position.z == goal_pos.pose.position.z:
                cmd.pose.position.z = node.odom_pose.pose.position.z - LANDING_INCREMENT

            # update local landing goal downward in increments
            if np.abs(cmd.pose.position.z - node.odom_pose.pose.position.z) < LOCAL_GOAL_TOLERANCE:
                cmd.pose.position.z -= LANDING_INCREMENT

            # clamp near ground reference
            if abs(node.odom_pose.pose.position.z - node.ground_z) < LOCAL_GOAL_TOLERANCE:
                cmd.pose.position.z = node.ground_z

            # when landed (disarmed) switch to AUTO.LOITER and GROUND
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

            if node.state.mode == "AUTO.LAND" and not node.state.armed and node.get_clock().now() - prev_request > Duration(seconds=5.0):
                offb_set_mode.custom_mode = "STABILIZED"
                if node.set_mode_cli.call(offb_set_mode).mode_sent:
                    node.get_logger().info("Landed from ABORT, switching to GROUND")
                    MODE = GROUND
                prev_request = node.get_clock().now()

        elif MODE == GROUND:
            # lock commanded altitude to ground reference
            cmd.pose.position.x = node.odom_pose.pose.position.x
            cmd.pose.position.y = node.odom_pose.pose.position.y
            cmd.pose.position.z = node.ground_z

        # publish setpoint
        cmd.header.stamp = node.get_clock().now().to_msg()
        node.pose_pub.publish(cmd)
        node.rate.sleep()

    rclpy.shutdown()
    thread.join()


if __name__ == '__main__':
    main()
