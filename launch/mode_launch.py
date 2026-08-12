from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    mode_node = Node(
        package="dre",
        executable="mode_node",
        name="mode_node",
        output="screen",
        parameters=[{
            "mode": LaunchConfiguration("mode"),
            "headless": LaunchConfiguration("headless"),
        }],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mode",
                default_value="dro",
                description="Operational mode to start in (see mode_node.py's MODE_TABLE)",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="If true, skip UI nodes (e.g. rviz) for the chosen mode",
            ),
            mode_node,
        ]
    )
