from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch the ROS-Ignition bridge for /cmd_vel
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_ign_bridge',
            arguments=['/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist'],
            output='screen'
        ),
        
        # Launch the PID controller node
        Node(
            package='controllers',  # Replace with your package name
            executable='pid',
            name='pid_controller',
            output='screen'
        ),
        
        # Launch the ROS-Ignition bridge for robot position (odometry)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_ign_bridge_position',
            arguments=['/model/jetracer/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry'],
            output='screen'
        ),

        # Launch the ROS-Ignition bridge for LIDAR data
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_ign_bridge_lidar',
            arguments=['/lidar@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan'],
            output='screen'
        )
    ])
