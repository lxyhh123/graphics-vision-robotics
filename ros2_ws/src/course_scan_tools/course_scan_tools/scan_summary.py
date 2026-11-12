import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

from .geometry import front_distance


class ScanSummary(Node):
    def __init__(self):
        super().__init__("course_scan_summary")
        self.summary = self.create_publisher(Float32, "course/front_distance", 10)
        self.scan = self.create_subscription(
            LaserScan, "scan", self.on_scan, qos_profile_sensor_data
        )

    def on_scan(self, scan):
        distance = front_distance(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, scan.range_max,
        )
        self.summary.publish(Float32(data=float(distance)))


def main(args=None):
    rclpy.init(args=args)
    node = ScanSummary()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
