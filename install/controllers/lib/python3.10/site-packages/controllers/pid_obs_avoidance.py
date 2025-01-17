#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile
from tf2_msgs.msg import TFMessage
# from network import Actor, Critic
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

def get_angle(x1, y1, x2, y2):
	# return np.arctan2(y2 - y1, x2 - x1)
	return np.arctan2(y2 - y1, x2 - x1)


def normalize_angle(angle):
    """
    Normalize the angle to be within the range [-pi, pi].
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def get_angle_difference(angle1, angle2):
    """
    Calculate the shortest path difference between two angles,
    ensuring the result is within [-pi, pi].
    """
    difference = angle1 - angle2
    return normalize_angle(difference)

class Actor(nn.Module):
    def __init__(self, grid_size):
        super(Actor, self).__init__()
        n_input = grid_size * grid_size + 1
        self.fc1 = nn.Linear(n_input, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        if isinstance(x, tuple):
            x = tuple(torch.tensor(item, dtype=torch.float) for item in x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = x * torch.pi / 2.0
        return x
    
class PID_OBS(Node):
    def __init__(self):

        # Initialize the Node
        super().__init__("pid_obs_avoidance")
        self.get_logger().info("PID Controller with Obstacle Avoidance Node has been started.")

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
        self.grid_size = 10
        self.theta1 = 0
        self.theta2 = 120
        self.radius = 3.5
        self.num_slices_angular = self.grid_size
        self.num_slices_radial = self.grid_size + 1
        self.rotation_angle = -self.theta2 / 2
        self.error_heading = 0.0
        self.calculate_grid_centers()
        self.actor = Actor(grid_size=self.grid_size)
        self.final_target = np.array([5, 1], dtype=np.float32)
        self.obs = None

		# Load the state_dict into the model
        # Finding the path to the networks
        package_name = 'controllers'
        package_path = get_package_share_directory(package_name)
        model_relative_path = os.path.join(package_path, 'ppo_actor.pth')
        self.actor.load_state_dict(torch.load(model_relative_path))

        # Create a publisher to send the Twist commands
        qos_profile = QoSProfile(depth=10)
        self.cmd_publish = self.create_publisher(Twist, "cmd_vel", qos_profile)

        # Subscribe to the odometry topic
        # self.odom_subscriber = self.create_subscription(
        #     Odometry,
        #     '/model/jetracer/odometry',
        #     self.odom_callback,
        #     qos_profile
        # )

        # Subscribe to the lidar topic
        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/lidar',
            self.create_state,
            qos_profile
        )
        
        # Subscribe to ground truth pose of the robot
        self.tf_subscriber = self.create_subscription(
            TFMessage,
            '/world/jetracer_world/dynamic_pose/info',
            self.tf_callback,
            qos_profile
        )

        self.timer = self.create_timer(0.1, self.pid_controller)

    def tf_callback(self, msg):

        second_tansform = msg.transforms[1]
        self.position = second_tansform.transform.translation
        orientation = second_tansform.transform.rotation

        # Convert quaternion to yaw
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

        self.get_logger().info(f"Position: x={self.position.x}, y={self.position.y}, Yaw: {math.degrees(self.yaw):.2f} degrees")

        
    def pid_controller(self):

        theta = self.actor(self.obs)
        theta = torch.clamp(theta, min= -0.6, max= 0.6)

        # PID contoller
        target_x = self.final_target[0]
        target_y = self.final_target[1]
        theta = theta.detach().numpy()
        x_robot = self.position.x
        y_robot = self.position.y
        yaw_robot = self.yaw
        # print(f"yaw: {yaw_robot}")
        # print(f"theta: {theta}")
        theta += yaw_robot
        temp_local_coord = np.array([self.radius * np.cos(theta), self.radius * np.sin(theta)])
        temp_global_coord = np.array([x_robot, y_robot]) + temp_local_coord.reshape(2,)

        # if np.any(self.obs == 0):
        #     print("Obstacle detected")
        target_x = temp_global_coord[0]
        target_y = temp_global_coord[1]

        error_x = target_x - x_robot
        error_y = target_y - y_robot
        distance_to_target = math.sqrt((self.final_target[0] - x_robot) ** 2 + (self.final_target[1] - y_robot) ** 2)
                        
        target_angle = math.atan2(error_y, error_x)
        error_heading = target_angle - yaw_robot

        while error_heading > np.pi:
            error_heading -= 2 * np.pi
        while error_heading < -np.pi:
            error_heading += 2 * np.pi
            
        self.error_heading = error_heading

        angular_velocity = self.kp * error_heading + self.ki * self.integral + self.kd * (error_heading - self.prev_error)
        
        self.prev_error = error_heading
        self.integral += error_heading
        
        if distance_to_target < 1:
            # self.get_logger().info(f"Target reached at x={x_robot}, y={y_robot}, Distance to target: {distance_to_target}, target_x: {self.final_target[0]}, target_y: {self.final_target[1]}")
            self.cmd.linear.x = 0.0
            self.cmd.angular.z = 0.0
        else:
            self.cmd.linear.x = self.x_speed
            self.cmd.angular.z = angular_velocity

        self.cmd_publish.publish(self.cmd) 
        
        # Print the current position and yaw
        # self.get_logger().info(
        #     f"Position: x={x_robot}, y={y_robot} | Yaw: {math.degrees(yaw_robot):.2f} degrees"
        # )

    def odom_callback(self, msg):

        self.position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # Convert quaternion to yaw
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        # if self.obs is not None:
        # self.get_logger().info(f"Position: x={self.position.x}, y={self.position.y} | Yaw: {math.degrees(self.yaw):.2f} degrees")
        #     desired_theta = self.actor(self.obs)
            # self.pid_controller(desired_theta)

    def create_state(self, msg):
        
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = msg.range_max
        r_theta = []
        
        for i in range(len(msg.ranges)):
            distance = 0.0 if msg.ranges[i] == float('inf') else msg.ranges[i]
            theta = angle_min + i * angle_increment
            if distance != 0.0:
                r_theta.append((distance, theta))
        
        r_theta = np.array(r_theta)

        # Create the grid
        grid = np.ones((self.grid_size, self.grid_size))
        
        cell_height_theta = np.radians((self.theta2 - self.theta1) / self.num_slices_angular)
        cell_width_r = self.radius / self.num_slices_radial
        
        for point in r_theta:
            r, theta = point[0], point[1]
            theta = theta - np.radians(self.rotation_angle)
            # theta = np.radians(theta)

            cell_y_theta = self.grid_size - 1 - int(theta // cell_height_theta)
            if cell_y_theta == self.grid_size:
                cell_y_theta = self.grid_size - 1
            cell_x_r = int((r - cell_width_r) // cell_width_r)
            if cell_x_r == self.grid_size:
                cell_x_r = self.grid_size - 1
            grid[cell_y_theta, cell_x_r] = 0

        # Make all cells after the first zero in each row also zero
        for row in range(self.grid_size):
            zero_found = False
            for col in range(self.grid_size):
                if grid[row, col] == 0:
                    zero_found = True  # Start marking the cells after first zero
                if zero_found:
                    grid[row, col] = 0  # Set cells after first zero to zero
        
        positioning_array = self.grid_centers_polar[:,:,0] + (self.grid_centers_polar[:,:,1])
        obs = grid * positioning_array
        obs = obs.reshape(self.grid_size * self.grid_size)
        body_to_goal = get_angle(self.position.x, self.position.y , self.final_target[0], self.final_target[1])
        error_angle = get_angle_difference(body_to_goal, self.yaw)
        # adding error_heading to the end of the array
        self.obs = np.append(obs, [error_angle], axis=0)

        
        # self.get_logger().info(f"LIDAR points: {grid}")

    def calculate_grid_centers(self):

        theta1 = self.theta1
        theta2 = self.theta2
        radius = self.radius
        theta1 = 0
        theta2 = 120
        radius = 3.5

        # Function to calculate midpoints in polar coordinates
        def midpoint_polar(r1, r2, theta1, theta2):
            r_mid = (r1 + r2) / 2
            theta_mid = (theta1 + theta2) / 2
            return r_mid, theta_mid

        # Rotate a point in polar coordinates
        def rotate_point_polar(r, theta, angle_deg):
            theta_rad = np.radians(angle_deg)
            theta_rot = theta + theta_rad
            return r, theta_rot

        grid_centers = np.zeros((self.grid_size + 1, self.grid_size, 2), dtype=np.float32)
        grid_centers_polar = np.zeros((self.grid_size + 1, self.grid_size, 2), dtype=np.float32)

        # grid_centers=[]
        radial_intervals = np.linspace(0, radius, self.num_slices_radial + 1)
        angular_intervals = np.linspace(np.radians(theta1), np.radians(theta2), self.num_slices_angular + 1)

        # Calculate midpoints for each grid cell
        for i in range(self.num_slices_radial):
            for j in reversed(range(self.num_slices_angular)):
                r_mid, theta_mid = midpoint_polar(radial_intervals[i], radial_intervals[i + 1],
                                                  angular_intervals[j], angular_intervals[j + 1])
                # Apply rotation
                r_rot, theta_rot = rotate_point_polar(r_mid, theta_mid, self.rotation_angle)
                # Convert to Cartesian coordinates
                x_rot = r_rot * np.cos(theta_rot)
                y_rot = r_rot * np.sin(theta_rot)
                grid_centers[i, self.grid_size - 1 - j] = [x_rot, y_rot]
                grid_centers_polar[i, self.grid_size - 1 - j] = [r_rot, theta_rot]

        self.grid_centers_1 = grid_centers[1:, :, :]
        self.grid_centers_polar = np.transpose(grid_centers_polar[1:, :, :], axes=(1, 0, 2))


        return grid_centers[1:, :, :], grid_centers_polar[1:, :, :]
    


def main(args=None):
    rclpy.init(args=args)
    node = PID_OBS()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
