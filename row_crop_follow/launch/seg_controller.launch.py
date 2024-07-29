from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    package_name = "row_crop_follow"
    package_share_path = FindPackageShare(package_name)
    default_config_path = PathJoinSubstitution([package_share_path, "config", "seg_controller_params.yaml"])

    params_file = LaunchConfiguration("params_file", default=default_config_path)
    declare_arg_params_file = DeclareLaunchArgument(
        "params_file", default_value=default_config_path, description="Path to the params file to load"
    )

    node = Node(
        name="seg_controller",
        package=package_name,
        executable="seg_controller_node",
        parameters=[params_file],
        output="screen",
    )

    return LaunchDescription(
        [
            declare_arg_params_file,
            node,
        ]
    )
