#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class PID(Node):
    def __init__(self):
        super().__init__("pid_controller")
        self.get_logger().info("PID Controller Node has been started.")


def main(args=None):
    rclpy.init(args=args)
    node = PID()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()