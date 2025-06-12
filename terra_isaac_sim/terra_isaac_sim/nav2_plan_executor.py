import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import pickle
import argparse
from pathlib import Path

from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from tf_transformations import quaternion_from_euler

from terra.config import AgentConfig
from terra.utils import angle_idx_to_rad


class Nav2PlanExecutor(Node):
    """ROS 2 node that executes a plan using Nav2 navigation."""

    def __init__(self, plan_path: str, frame_id: str = "map"):
        super().__init__('nav2_plan_executor')

        # Configuration
        self.frame_id = frame_id
        self.current_goal_index = 0
        self.plan = []
        self.executing_goal = False

        # Digger configuration
        self.angles_base = AgentConfig().angles_base
        self.angles_cabin = AgentConfig().angles_cabin

        # Load the plan
        self.load_plan(plan_path)

        # Create Nav2 action client
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        # Wait for Nav2 to be available
        self.get_logger().info("Waiting for Nav2 action server...")
        self.nav_client.wait_for_server()
        self.get_logger().info("Nav2 action server is available!")

        # Start executing the plan after a short delay
        self.start_timer = self.create_timer(2.0, self.start_plan_execution)

    def load_plan(self, plan_path: str):
        """Load the plan from a pickle file."""
        try:
            with open(plan_path, 'rb') as f:
                self.plan = pickle.load(f)

            # Validate plan format
            if not isinstance(self.plan, list):
                raise ValueError(f"Expected plan to be a list, got {type(self.plan)}")
            if len(self.plan) == 0:
                raise ValueError("Plan is empty")

            # Sort plan by timestep to ensure correct order
            self.plan.sort(key=lambda x: x['step'])
            self.get_logger().info(f"Loaded plan with {len(self.plan)} waypoints from {plan_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to load plan from {plan_path}: {e}")
            raise

    def start_plan_execution(self):
        """Start executing the plan by sending the first goal."""
        if len(self.plan) == 0:
            self.get_logger().warn("No plan to execute!")
            return

        # Cancel the timer - we only want to start once
        self.destroy_timer(self.start_timer)

        self.get_logger().info("Starting plan execution...")
        self.send_next_goal()

    def send_next_goal(self):
        """Send the next goal in the plan to Nav2."""
        if self.current_goal_index >= len(self.plan):
            self.get_logger().info("Plan execution completed!")
            return

        if self.executing_goal:
            self.get_logger().warn("Already executing a goal, skipping...")
            return

        # Get current waypoint
        waypoint = self.plan[self.current_goal_index]
        agent_state = waypoint['agent_state']

        # TODO: Is there no transform between Terra and Isaac environments?
        x_base = agent_state['pos_base'][0]
        y_base = agent_state['pos_base'][1]
        angle_base = angle_idx_to_rad(
            agent_state['angle_base'],
            self.angles_base
        )
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.create_pose_stamped(
            x_base, y_base, angle_base
        )

        self.get_logger().info(
            f"Sending goal {self.current_goal_index + 1}/{len(self.plan)}: "
            f"pos=({agent_state['pos_base'][0]:.2f}, {agent_state['pos_base'][1]:.2f}), "
            f"angle={agent_state['angle_base']:.2f} rad"
        )

        # Send goal
        self.executing_goal = True
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def create_pose_stamped(self, x: float, y: float, yaw: float) -> PoseStamped:
        """Create a PoseStamped message from x, y, yaw."""
        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        quat = quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation = Quaternion(
            x=quat[0], y=quat[1], z=quat[2], w=quat[3]
        )

        return pose

    def goal_response_callback(self, future):
        """Handle the response when a goal is sent."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected!")
            self.executing_goal = False
            return

        self.get_logger().info("Goal accepted, waiting for result...")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """Handle the result when a goal is completed."""
        result = future.result().result
        self.executing_goal = False

        if result:
            self.get_logger().info(
                f"Goal {self.current_goal_index + 1} completed successfully!"
            )
        else:
            self.get_logger().error(
                f"Goal {self.current_goal_index + 1} failed!"
            )
        self.current_goal_index += 1
        self.create_timer(1.0, self.send_next_goal_delayed)

    def send_next_goal_delayed(self):
        """Send the next goal after a delay (to be called by timer)."""
        self.send_next_goal()
        # Destroy the timer since it's only meant to fire once
        try:
            self.destroy_timer(self.next_goal_timer)
        except:
            pass

    def navigation_feedback_callback(self, feedback_msg):
        """Handle navigation feedback."""
        feedback = feedback_msg.feedback
        current_pose = feedback.current_pose.pose

        # Log progress occasionally
        if hasattr(self, '_last_feedback_time'):
            current_time = self.get_clock().now()
            time_diff = (current_time - self._last_feedback_time).nanoseconds / 1e9
            if time_diff < 2.0:  # Log every 2 seconds
                return

        self._last_feedback_time = self.get_clock().now()

        self.get_logger().info(
            f"Current position: ({current_pose.position.x:.2f}, "
            f"{current_pose.position.y:.2f}), "
            f"Distance remaining: {feedback.distance_remaining:.2f}m"
        )


def main():
    """Main function to run the Nav2 plan executor."""
    parser = argparse.ArgumentParser(description="Execute a plan using Nav2")
    parser.add_argument(
        "-p", "--plan_path",
        type=str,
        required=True,
        help="Path to the plan .pkl file"
    )
    parser.add_argument(
        "-f", "--frame_id",
        type=str,
        default="map",
        help="Frame ID for navigation goals (default: map)"
    )

    args = parser.parse_args()

    # Verify plan file exists
    plan_path = Path(args.plan_path)
    if not plan_path.exists():
        print(f"Error: Plan file {plan_path} does not exist!")
        return 1

    rclpy.init()

    try:
        node = Nav2PlanExecutor(str(plan_path), args.frame_id)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        try:
            node.destroy_node()
        except:
            pass
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    exit(main())
