from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Bridge for /cmd_vel topic (geometry_msgs/Twist <-> ignition.msgs.Twist)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_ign_bridge_cmd_vel',
            arguments=['/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist'],
            output='screen'
        ),
        
        # PID Controller Node
        Node(
            package='controllers',  # Replace with your package name
            executable='pid_obs_avoidance',
            name='pid_obs_avoidance',
            output='screen'
        ),
        
        # Bridge for robot position (odometry: nav_msgs/Odometry <-> ignition.msgs.Odometry)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_ign_bridge_odometry',
            arguments=['/model/jetracer/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry'],
            output='screen'
        ),

        # Bridge for robot TF (Pose_V: tf2_msgs/TFMessage <-> ignition.msgs.Pose_V)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_ign_bridge_tf',
            arguments=['/world/jetracer_world/dynamic_pose/info@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V'],
            output='screen'
        ),

        # Bridge for LIDAR data (sensor_msgs/LaserScan <-> ignition.msgs.LaserScan)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_ign_bridge_lidar',
            arguments=['/lidar@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan'],
            output='screen'
        )
    ])
