import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_conjugate(q):
    x, y, z, w = q
    return -x, -y, -z, w


def quat_normalize(q):
    x, y, z, w = q
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm == 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


class RealSenseRelayNode(Node):
    def __init__(self):
        super().__init__('rob498_realsense_relay')

        self.output_frame_id = 'map'

        # RealSense camera frame (x right, y down, z forward) -> Vicon-like ENU (x forward, y left, z up)
        # Position mapping: (x, y, z) -> (z, -x, -y)
        # Equivalent frame-rotation quaternion used for orientation basis change.
        #self.rs_to_enu_quat = (-0.5, 0.5, -0.5, 0.5)
        # self.rs_to_enu_quat_conj = quat_conjugate(self.rs_to_enu_quat)

        self.vision_pose_pub = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 10)
        self.get_logger().info('test0')
        self.realsense_sub = self.create_subscription(
            Odometry,
            '/camera/pose/sample',
            self.realsense_pose_callback,
            10,
        )

        self.get_logger().info(
            f'Relaying RealSense pose from /camera/pose/sample to /mavros/vision_pose/pose '
        )

    def realsense_pose_callback(self, msg: Odometry) -> None:
 	
        self.vision_pose_pub.publish(msg)
        
        self.get_logger().info('test')


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseRelayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
