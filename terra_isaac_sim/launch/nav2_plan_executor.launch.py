#!/usr/bin/env python3
"""
Launch file for the Nav2 plan executor.
This launch file starts the nav2_plan_executor node with configurable parameters.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate the launch description for nav2_plan_executor."""
    
    # Declare launch arguments
    plan_path_arg = DeclareLaunchArgument(
        'plan_path',
        description='Path to the plan .pkl file'
    )
    
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='map',
        description='Frame ID for navigation goals'
    )
    
    # Create the node
    nav2_plan_executor_node = Node(
        package='terra_nav',  # You might need to adjust this package name
        executable='nav2_plan_executor.py',
        name='nav2_plan_executor',
        output='screen',
        parameters=[{
            'use_sim_time': True,  # Set to False for real robot
        }],
        arguments=[
            '--plan_path', LaunchConfiguration('plan_path'),
            '--frame_id', LaunchConfiguration('frame_id')
        ]
    )
    
    return LaunchDescription([
        plan_path_arg,
        frame_id_arg,
        nav2_plan_executor_node,
    ])
