from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessIO
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    default_config = PathJoinSubstitution(
        [FindPackageShare("dre"), "config", "config_loc.yaml"]
    )
    rviz_file = PathJoinSubstitution(
        [FindPackageShare("dre"), "config", "rviz_drl.rviz"]
    )
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

    loc_node = Node(
        package="dre",
        executable="loc_node",
        name="loc_node",
        output="screen",
        parameters=[
            {"config_file": LaunchConfiguration("config_file")}
        ],
    )

    initial_pose_selector = Node(
        package="dre",
        executable="initial_pose_selector",
        name="initial_pose_selector",
        output="screen",
    )

    loc_viz_node = Node(
        package="dre",
        executable="loc_viz_node",
        name="loc_viz_node",
        output="screen",
    )

    # loc_node also publishes /map_path at boot (for the static map it loaded),
    # so map_viz_node works here too: it shows the overall map with the live
    # /drl_estimate dot, complementing loc_viz_node's close-up scan overlay.
    # rviz_drl.rviz has displays for both nodes' output topics.
    map_viz_node = Node(
        package="dre",
        executable="map_viz_node",
        name="map_viz_node",
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_file]
    )

    static_transform_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "odom", "map"]
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
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to loc_node config yaml (set map_path and initial_pose inside)",
            ),
            dro_node,
            loc_node,
            initial_pose_selector,
            loc_viz_node,
            map_viz_node,
            static_transform_publisher,
            RegisterEventHandler(
                OnProcessIO(
                    target_action=dro_node,
                    on_stdout=launch_rviz_when_ready,
                    on_stderr=launch_rviz_when_ready,
                )
            ),
        ]
    )
