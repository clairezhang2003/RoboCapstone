import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class RealSenseRelayNode(Node):
    def __init__(self):
        super().__init__('rob498_realsense_relay')
        self.output_frame_id = 'map'
        self.latest_pose = None  # Store latest data to decouple frequency

        self.vision_pose_pub = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10  # Reduced from 10 to 1 to minimize cache/latency
        )

        self.realsense_sub = self.create_subscription(
            Odometry,
            '/camera/pose/sample',
            self.realsense_pose_callback,
            qos_profile_sensor_data
        )

        # Create a timer to publish at a fixed rate (e.g., 30Hz)
        # 30Hz is a stable frequency for PX4 EKF2 fusion
        publish_period = 1.0 / 30.0 
        self.timer = self.create_timer(publish_period, self.publish_at_rate)

#        self.get_logger().info('Relaying RealSense pose -> /mavros/vision_pose/pose at 30Hz')

    def realsense_pose_callback(self, msg: Odometry) -> None:
        # Just store the message to be picked up by the timer
        self.latest_pose = msg

    def publish_at_rate(self) -> None:
        # Only publish if we have received at least one message
        if self.latest_pose is not None:
            pose_msg = PoseStamped()
            pose_msg.header = self.latest_pose.header
            pose_msg.header.frame_id = self.output_frame_id
            pose_msg.pose = self.latest_pose.pose.pose

            self.vision_pose_pub.publish(pose_msg)
            # Log removed from here to prevent console/cache flooding

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseRelayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
