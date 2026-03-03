import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class RealSenseRelayNode(Node):
    def __init__(self):
        super().__init__('rob498_realsense_relay')

        self.output_frame_id = 'map'
        self.received_count = 0

        # RealSense camera frame (x right, y down, z forward) -> Vicon-like ENU (x forward, y left, z up)
        # Position mapping: (x, y, z) -> (z, -x, -y)
        # Equivalent frame-rotation quaternion used for orientation basis change.

        self.vision_pose_pub = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 10)
        self.realsense_sub = self.create_subscription(
            Odometry,
            '/camera/pose/sample',
            self.realsense_pose_callback,
            qos_profile_sensor_data,
        )
        self.health_timer = self.create_timer(2.0, self._health_check)

        self.get_logger().info('Relaying RealSense pose from /camera/pose/sample to /mavros/vision_pose/pose')
        self.get_logger().info('Subscription QoS set to sensor_data (best_effort) for RealSense compatibility')
        
    def realsense_pose_callback(self, msg: Odometry) -> None:
        self.received_count += 1
        out = PoseStamped()
        out.header = msg.header
        out.header.frame_id = self.output_frame_id
        out.pose = msg.pose.pose
        self.vision_pose_pub.publish(out)

        if self.received_count == 1:
            self.get_logger().info('First /camera/pose/sample message received and relayed to /mavros/vision_pose/pose')

    def _health_check(self) -> None:
        if self.received_count == 0:
            self.get_logger().warn('No /camera/pose/sample messages received yet (check topic name and QoS)')

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseRelayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
