import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

WS = os.path.expanduser('~/ROB498-flight/resources/code/ros2_ws')

def generate_launch_description():

    # MAVROS launch
    mavros_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('px4_autonomy_modules'),
                'launch',
                'mavros.launch.py'
            )
        )
    )

    # RealSense launch
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        )
    )

    # RealSense relay node (standalone script)
    realsense_relay = ExecuteProcess(
        cmd=['python3', os.path.join(WS, 'realsense_relay_node.py')],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        mavros_launch,
        realsense_launch,
        realsense_relay,
    ])
