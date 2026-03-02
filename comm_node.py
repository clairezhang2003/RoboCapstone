import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
import rclpy.time
import rclpy.qos
from std_srvs.srv import Trigger
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool
from geometry_msgs.msg import PoseStamped
import numpy as np
import threading

# Constants
GROUND = 'GROUND'
CONNECT = 'CONNECT'
TAKEOFF = 'TAKEOFF'
HOVER = 'HOVER'
LAND = 'LAND'
ABORT = 'ABORT'

LOCAL_GOAL_TOLERANCE = 0.15 
GOAL_TOLERANCE = 0.05
TAKEOFF_INCREMENT = 0.05  # Reduced for smoother ascent
LANDING_INCREMENT = 0.1
GOAL_HEIGHT = 0.5

# Global variables for state management
COMMAND = 'ground'
MODE = GROUND

# Callback handlers
def handle_launch():
    global COMMAND
    COMMAND = 'launch'
    print('Launch Requested.')

def handle_test():
    global COMMAND
    COMMAND = 'test'
    print('Test Requested.')

def handle_land():
    global COMMAND 
    COMMAND = 'land'
    print('Land Requested.')

def handle_abort():
    global COMMAND 
    COMMAND = 'abort'
    print('Abort Requested.')

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
        super().__init__('rob498_drone_08')
        
        # Services
        self.srv_launch = self.create_service(Trigger, 'rob498_drone_08/comm/launch', callback_launch)
        self.srv_test = self.create_service(Trigger, 'rob498_drone_08/comm/test', callback_test)
        self.srv_land = self.create_service(Trigger, 'rob498_drone_08/comm/land', callback_land)
        self.srv_abort = self.create_service(Trigger, 'rob498_drone_08/comm/abort', callback_abort)

        # Clients
        self.set_mode_cli = self.create_client(SetMode, 'mavros/set_mode')
        self.arm_cli = self.create_client(CommandBool, 'mavros/cmd/arming')

        # Pub/Sub
        self.state_sub = self.create_subscription(State, 'mavros/state', self.state_callback, 10)
        self.odom_sub = self.create_subscription(PoseStamped, 'mavros/local_position/pose', self.odom_callback, rclpy.qos.qos_profile_system_default)
        self.pose_pub = self.create_publisher(PoseStamped, 'mavros/setpoint_position/local', 10)

        self.state = State()
        self.odom_pose = None
        self.rate = self.create_rate(20) # 20Hz is ideal for MAVROS offboard

    def state_callback(self, msg):
        self.state = msg

    def odom_callback(self, msg):
        self.odom_pose = msg

def main(args=None):
    global COMMAND, MODE 

    rclpy.init(args=args)
    node = CommNode()

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    # Wait for heartbeat and Odometry
    node.get_logger().info("Waiting for MAVROS connection and Odom...")
    while rclpy.ok() and (not node.state.connected or node.odom_pose is None):
        node.rate.sleep()
    
    node.get_logger().info("Connected! Ready.")

    # Initialize goals
    goal_pos = PoseStamped()
    goal_pos.pose.position.z = GOAL_HEIGHT
    
    cmd = PoseStamped()
    cmd.header.frame_id = "map"
    
    offb_set_mode = SetMode.Request()
    offb_set_mode.custom_mode = "OFFBOARD"
    
    arm_cmd = CommandBool.Request()
    arm_cmd.value = True 

    prev_request = node.get_clock().now()

    while rclpy.ok():
        # 1. Update Global State Machine Transitions
        if COMMAND == 'abort' and MODE != GROUND:
            MODE = ABORT
        elif COMMAND == 'launch' and MODE == GROUND:
            MODE = CONNECT
        elif COMMAND == 'test' and MODE == HOVER:
            # Only allow test if we are already hovering
            pass 
        elif COMMAND == 'land' and MODE not in [GROUND, LAND]:
            MODE = LAND

        # 2. Mode Logic
        if MODE == GROUND:
            # Keep setpoint at current ground position
            cmd.pose.position = node.odom_pose.pose.position

        elif MODE == CONNECT:
            # STEP A: Send setpoints BEFORE switching to offboard
            cmd.pose.position = node.odom_pose.pose.position
            
            if node.state.mode != "OFFBOARD":
                if (node.get_clock().now() - prev_request) > Duration(seconds=2.0):
                    node.get_logger().info("Setting OFFBOARD...")
                    node.set_mode_cli.call_async(offb_set_mode)
                    prev_request = node.get_clock().now()
            
            elif not node.state.armed:
                if (node.get_clock().now() - prev_request) > Duration(seconds=2.0):
                    node.get_logger().info("Arming...")
                    node.arm_cli.call_async(arm_cmd)
                    prev_request = node.get_clock().now()
            
            else:
                node.get_logger().info("Armed and Offboard ready.")
                MODE = TAKEOFF

        elif MODE == TAKEOFF:
            # Incrementally increase altitude setpoint
            if np.abs(GOAL_HEIGHT - node.odom_pose.pose.position.z) < GOAL_TOLERANCE:
                MODE = HOVER
                node.get_logger().info("Hover altitude reached.")
            else:
                # Smoothly ramp up the setpoint
                cmd.pose.position.z = min(cmd.pose.position.z + TAKEOFF_INCREMENT, GOAL_HEIGHT)

        elif MODE == HOVER:
            cmd.pose.position.z = GOAL_HEIGHT

        elif MODE == LAND:
            # Decrement altitude
            if node.odom_pose.pose.position.z < 0.1: # Close to ground
                if node.state.armed:
                    # Request Disarm or switch to Auto-Land
                    offb_set_mode.custom_mode = "AUTO.LAND"
                    node.set_mode_cli.call_async(offb_set_mode)
                else:
                    MODE = GROUND
                    COMMAND = 'ground'
            else:
                cmd.pose.position.z = max(cmd.pose.position.z - LANDING_INCREMENT, 0.0)

        elif MODE == ABORT:
            offb_set_mode.custom_mode = "AUTO.LAND"
            node.set_mode_cli.call_async(offb_set_mode)
            if not node.state.armed:
                MODE = GROUND

        # 3. Final Publication
        cmd.header.stamp = node.get_clock().now().to_msg()
        node.pose_pub.publish(cmd)
        
        try:
            node.rate.sleep()
        except rclpy.errors.ROSInterruptException:
            break

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
