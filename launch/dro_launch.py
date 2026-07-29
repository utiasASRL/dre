from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessIO
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    rviz_file = PathJoinSubstitution(
        [FindPackageShare("dre"), "config", "rviz_pogo.rviz"])
    output_path = PathJoinSubstitution(
        [FindPackageShare("dre"), "output"]
    )

    dro_node = Node(
        package="dre",
        executable="dro_node",
        name="dro_node",
        output="screen",
        parameters=[
            {"output_path": output_path}
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_file]
    )

    # Launch rviz once dro_node logs that it's actually ready, rather than a fixed
    # guessed delay: startup time varies a lot (near-instant, or ~30s if torch
    # compile is enabled), and launching rviz too early just means it spams TF
    # warnings until dro_node catches up.
    rviz_launched = []

    def launch_rviz_when_ready(event):
        if rviz_launched or b"DRO ready" not in event.text:
            return None
        rviz_launched.append(True)
        return [rviz_node]

    return LaunchDescription(
        [
            dro_node,
            RegisterEventHandler(
                OnProcessIO(
                    target_action=dro_node,
                    on_stdout=launch_rviz_when_ready,
                    on_stderr=launch_rviz_when_ready,
                )
            ),
        ]
    )
