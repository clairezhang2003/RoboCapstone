import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


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

        output_frame_id = self.declare_parameter('output_frame_id', 'map').value
        self.apply_frame_transform = self.declare_parameter('apply_frame_transform', True).value

        self.output_frame_id = output_frame_id

        # RealSense camera frame (x right, y down, z forward) -> Vicon-like ENU (x forward, y left, z up)
        # Position mapping: (x, y, z) -> (z, -x, -y)
        # Equivalent frame-rotation quaternion used for orientation basis change.
        self.rs_to_enu_quat = (-0.5, 0.5, -0.5, 0.5)
        self.rs_to_enu_quat_conj = quat_conjugate(self.rs_to_enu_quat)

        self.vision_pose_pub = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 10)
        self.realsense_sub = self.create_subscription(
            PoseStamped,
            '/camera/pose/sample',
            self.realsense_pose_callback,
            10,
        )

        self.get_logger().info(
            f'Relaying RealSense pose from /camera/pose/sample to /mavros/vision_pose/pose '
            f'(transform={self.apply_frame_transform}, frame_id={self.output_frame_id})'
        )

    def realsense_pose_callback(self, msg: PoseStamped) -> None:
        if not self.apply_frame_transform:
            self.vision_pose_pub.publish(msg)
            return

        out = PoseStamped()
        out.header = msg.header
        out.header.frame_id = self.output_frame_id

        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z

        out.pose.position.x = pz
        out.pose.position.y = -px
        out.pose.position.z = -py

        q = msg.pose.orientation
        q_in = (q.x, q.y, q.z, q.w)
        q_out = quat_multiply(quat_multiply(self.rs_to_enu_quat, q_in), self.rs_to_enu_quat_conj)
        qx, qy, qz, qw = quat_normalize(q_out)

        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw

        self.vision_pose_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseRelayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
