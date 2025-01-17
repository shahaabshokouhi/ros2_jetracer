#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile
import math
import numpy as np

class PID(Node):
    def __init__(self):
        # Initialize the Node
        super().__init__("pid_controller")
        self.get_logger().info("PID Controller Node has been started.")

        # Variables to store position and yaw
        self.position = None
        self.yaw = None
        self.x_speed = 0.5
        self.cmd = Twist()
        self.prev_error = 0
        self.integral = 0
        self.kp = 0.8
        self.ki = 0.0
        self.kd = 0.02

        qos_profile = QoSProfile(depth=10)
        self.cmd_publish = self.create_publisher(Twist, "cmd_vel", qos_profile)

        # Subscribe to the odometry topic
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/model/jetracer/odometry',
            self.odom_callback,
            qos_profile
        )

        # self.timer = self.create_timer(0.2, self.print_pose)

    def pid_controller(self):

        # PID contoller
        target_x = 5
        target_y = -5
        
        x_robot = self.position.x
        y_robot = self.position.y
        yaw_robot = self.yaw            
        
        error_x = target_x - x_robot
        error_y = target_y - y_robot
        distance_to_target = math.sqrt(error_x ** 2 + error_y ** 2)
                        
        target_angle = math.atan2(error_y, error_x)
        # current_angle = 2 * math.atan2(self.pose.orientation.z, self.pose.orientation.w)
        error_heading = target_angle - yaw_robot
        
        if error_heading > np.pi:
            error_heading -= 2 * np.pi
        elif error_heading < -np.pi:
            error_heading += 2 * np.pi
            
        angular_velocity = self.kp * error_heading + self.ki * self.integral + self.kd * (error_heading - self.prev_error)
        
        self.prev_error = error_heading
        self.integral += error_heading
        
        if distance_to_target < 0.1:
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.0
        else:
            self.cmd.linear.x = self.x_speed
            self.cmd.angular.z = angular_velocity

        self.cmd_publish.publish(self.cmd) 
                # Print the current position and yaw
        self.get_logger().info(
            f"Position: x={x_robot}, y={y_robot} | Yaw: {math.degrees(yaw_robot):.2f} degrees"
        )

    def odom_callback(self, msg):

        self.position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # Convert quaternion to yaw
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        self.pid_controller()



def main(args=None):
    rclpy.init(args=args)
    node = PID()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
