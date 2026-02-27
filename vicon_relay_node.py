import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class ViconRelayNode(Node):
    def __init__(self):
        super().__init__('rob498_vicon_relay')
        self.vision_pose_pub = self.create_publisher(PoseStamped, '/mavros/vision_pose/pose', 10)
        self.vicon_sub = self.create_subscription(
            PoseStamped,
            '/vicon/ROB498_Drone/ROB498_Drone',
            self.vicon_pose_callback,
            10,
        )

    def vicon_pose_callback(self, msg: PoseStamped) -> None:
        self.vision_pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ViconRelayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
