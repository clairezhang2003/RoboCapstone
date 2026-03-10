#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Header
from builtin_interfaces.msg import Time


class WaypointPublisher(Node):
    def __init__(self):
        super().__init__('waypoint_publisher')

        self.publisher_ = self.create_publisher(PoseArray, 'rob498_drone_2/comm/waypoints', 10)
        self.timer = self.create_timer(1.0, self.publish_waypoints)  # Publish at 1 Hz

        self.get_logger().info('Waypoint publisher started, publishing to /com/waypoint')

    def publish_waypoints(self):
        msg = PoseArray()

        # Header
        msg.header = Header()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.header.frame_id = 'vicon/world'

        # Define waypoints: (x, y, z, qx, qy, qz, qw)
        waypoints = [
            (-1.5,  1.5, 0.5,  0.0, 0.0, 0.0, 1.0),
            ( 0.0,  1.3, 0.5,  0.0, 0.0, 0.0, 1.0),
            ( 1.5,  0.1, 0.5,  0.0, 0.0, 0.0, 1.0),
            ( 1.0, -1.8, 0.5,  0.0, 0.0, 0.0, 1.0),
        ]

        for (x, y, z, qx, qy, qz, qw) in waypoints:
            pose = Pose()
            pose.position = Point(x=x, y=y, z=z)
            pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            msg.poses.append(pose)

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published {len(msg.poses)} waypoints')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
