import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

import math 

class RealSenseRelayNode(Node):
    def __init__(self):
        super().__init__('rob498_realsense_relay')
        self.output_frame_id = 'map'
        self.publish_rate =  30.0
        timer_period = 1.0 / self.publish_rate
	self.vision_pose_pub = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10
        )

        self.realsense_sub = self.create_subscription(
            Odometry,
            '/camera/pose/sample',
            self.realsense_pose_callback,
            qos_profile_sensor_data  # fixes the silent QoS mismatch
        )

        self.timer = self.create_timer(timer_period, self.publish_callback)
	#self.get_logger().info('Relaying RealSense pose -> /mavros/vision_pose/pose')

    def realsense_callback(self, msg: Odometry) -> None:
	self.latest_pose = msg 

    def realsen_pose_callback(self, msg: Odometry) -> None:
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.header.frame_id = self.output_frame_id
        pose_msg.pose = msg.pose.pose  # extract Pose from PoseWithCovariance

        self.vision_pose_pub.publish(pose_msg)
        #self.get_logger().info('Published vision pose')

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseRelayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
